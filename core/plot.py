"""
Benchmark results bar chart: one panel per benchmark family, headline metrics only.

* 95% CI error bars where the summary carries <key>_ci_lo / <key>_ci_hi
* sample size n under each bar
* dashed chance level where one is defined (1/#options, 50% yes/no)
* incomplete benchmarks (summary complete == False) are hatched and labelled
"""
import os

# (benchmark, metric key, bar label, chance level or None); the benchmark's
# display name (_DISPLAY) is written once under its group of bars.
_PANELS = [
    ("Text MCQ", "#2a78d6", [
        ("medqa",    "accuracy_pct", "Accuracy", 25.0),   # 4 options
        ("rar",      "accuracy_pct", "Accuracy", 20.0),   # 5 options
        ("radiorag", "accuracy_pct", "Accuracy", 25.0),   # 4 options
    ]),
    ("Image VQA", "#eb6834", [
        ("radbench",        "mcq_accuracy_pct",        "MCQ",           None),  # 2–12 options
        ("radbench",        "yes_no_accuracy_pct",     "Yes/No",        50.0),
        ("radbench",        "open_judge_accuracy_pct", "Open\n(judge)", None),
        ("vqa_med_2019",    "open_exact_match_pct",    "Exact\nmatch",  None),
        ("vqa_med_2019",    "open_judge_accuracy_pct", "Open\n(judge)", None),
        ("radimagenet_vqa", "mcq_accuracy_pct",        "MCQ",           25.0),  # 4 options
        ("radimagenet_vqa", "yes_no_accuracy_pct",     "Yes/No",        50.0),
        ("radimagenet_vqa", "open_judge_accuracy_pct", "Open\n(judge)", None),
    ]),
    ("German report extraction", "#1baf7a", [
        ("label_extraction_mamma", "menopause_accuracy_pct",  "Meno-\npause acc.", None),
        ("label_extraction_mamma", "birads_li_accuracy_pct",  "BI-RADS\nL acc.",   None),
        ("label_extraction_mamma", "birads_re_accuracy_pct",  "BI-RADS\nR acc.",   None),
        ("label_extraction_mamma", "acr_li_accuracy_pct",     "BPE L\nacc.",       None),
        ("label_extraction_mamma", "acr_re_accuracy_pct",     "BPE R\nacc.",       None),
        ("label_extraction_mamma", "lesions_li_micro_f1_pct", "Lesions\nL F1",     None),
        ("label_extraction_mamma", "lesions_re_micro_f1_pct", "Lesions\nR F1",     None),
        ("label_extraction_arm",   "micro_f1_pct",            "Micro-F1",          None),
        ("label_extraction_arm",   "macro_f1_pct",            "Macro-F1",          None),
        ("label_extraction",       "micro_f1_pct",            "Micro-F1",          None),
    ]),
]

_DISPLAY = {
    "medqa": "MedQA", "rar": "RaR", "radiorag": "RadioRAG",
    "radbench": "RadBench", "vqa_med_2019": "VQA-Med-2019", "radimagenet_vqa": "RadImageNet-VQA",
    "label_extraction_mamma": "Mamma-MRT", "label_extraction_arm": "Arm X-ray",
    "label_extraction": "NER",
}

_INK = "#222222"
_MUTED = "#666666"
_INCOMPLETE = "#c0392b"


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f   # NaN -> None


def sample_size(metrics: dict, key: str):
    """Best available n for one metric of the summary dict."""
    candidates = [f"{key}_n"]
    for subset in ("yes_no", "mcq", "open"):          # VQA subsets: <subset>_rows
        if key.startswith(subset + "_"):
            candidates.append(f"{subset}_rows")
            break
    for suffix in ("_accuracy_pct", "_macro_f1_pct"):  # Mamma fields: <field>_n_bewertet
        if key.endswith(suffix):
            candidates.append(key[: -len(suffix)] + "_n_bewertet")
    candidates += ["rows", "n_gesamt", "n_items"]
    for c in candidates:
        n = _num(metrics.get(c))
        if n is not None:
            return int(n)
    return None


def collect_bars(summary: list) -> list:
    """Panels with their bars: [(title, color, [bar dict, ...]), ...] (empty panels dropped)."""
    by_bench = {name.lower(): m for name, m, err in summary if not err and m}
    panels = []
    for title, color, spec in _PANELS:
        bars = []
        for bench, key, label, chance in spec:
            m = by_bench.get(bench)
            if not m:
                continue
            value = _num(m.get(key))
            if value is None:
                continue
            lo, hi = _num(m.get(f"{key}_ci_lo")), _num(m.get(f"{key}_ci_hi"))
            bars.append({
                "bench": bench, "key": key, "label": label, "value": value,
                "ci": (lo, hi) if lo is not None and hi is not None else None,
                "n": sample_size(m, key), "chance": chance,
                "incomplete": m.get("complete") is False,
                "n_items": m.get("n_items"), "n_expected": m.get("n_expected"),
            })
        if bars:
            panels.append((title, color, bars))
    return panels


def _groups(bars: list) -> list:
    """Contiguous runs of bars of the same benchmark: [(bench, first_idx, last_idx)]."""
    out = []
    for i, b in enumerate(bars):
        if out and out[-1][0] == b["bench"]:
            out[-1][2] = i
        else:
            out.append([b["bench"], i, i])
    return out


def generate_results_chart(summary: list, model_name: str, out_path: str) -> None:
    """
    Generate the results chart.

    summary: list of (benchmark_name, metrics_dict, error_or_None); metrics may carry
             <key>_ci_lo/_ci_hi, <key>_n, n_items, n_expected and complete.
    model_name: string shown in the chart subtitle
    out_path: file path for the saved PNG (written atomically)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        print("  matplotlib not installed — skipping chart generation.")
        return

    panels = collect_bars(summary)
    if not panels:
        print("  No metrics to plot.")
        return

    widths = [max(len(bars), 2) for _, _, bars in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 9), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    if len(panels) == 1:
        axes = [axes]

    any_ci = any_chance = any_incomplete = False
    for ax, (title, color, bars) in zip(axes, panels):
        xs = list(range(len(bars)))
        for x, b in zip(xs, bars):
            ax.bar(x, b["value"], width=0.72, color=color,
                   alpha=0.45 if b["incomplete"] else 0.9,
                   hatch="///" if b["incomplete"] else None,
                   edgecolor=_INCOMPLETE if b["incomplete"] else "white",
                   linewidth=1.0, zorder=3)
            top = b["value"]
            if b["ci"]:
                lo, hi = b["ci"]
                ax.errorbar(x, b["value"], yerr=[[max(b["value"] - lo, 0)], [max(hi - b["value"], 0)]],
                            fmt="none", ecolor=_INK, elinewidth=1.2, capsize=4, zorder=4)
                top = max(top, hi)
                any_ci = True
            if b["chance"] is not None:
                ax.hlines(b["chance"], x - 0.42, x + 0.42, colors=_INK, linewidths=1.3,
                          linestyles=(0, (4, 3)), zorder=5)
                any_chance = True
            ax.text(x, top + 1.2, f"{b['value']:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=_INK, zorder=6)
            any_incomplete |= b["incomplete"]

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b['label']}\nn={b['n'] if b['n'] is not None else '?'}" for b in bars],
                           fontsize=8.5, color=_INK)

        # Benchmark name once per group, separators between groups
        for gi, (bench, first, last) in enumerate(_groups(bars)):
            b = bars[first]
            name = _DISPLAY.get(bench, bench)
            if b["incomplete"]:
                name += f"\nINCOMPLETE ({b['n_items']}/{b['n_expected']})"
            ax.annotate(name, xy=((first + last) / 2, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -44), textcoords="offset points", ha="center", va="top",
                        fontsize=10, fontweight="bold",
                        color=_INCOMPLETE if b["incomplete"] else _INK)
            if gi > 0:
                ax.axvline(first - 0.5, color="#bbbbbb", linewidth=0.8, zorder=1)

        ax.set_xlim(-0.6, len(bars) - 0.4)
        ax.set_title(title, fontsize=12, fontweight="bold", color=_INK, pad=10)
        ax.set_ylim(0, 106)
        ax.yaxis.grid(True, linestyle="-", color="#e3e3e3", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#999999")
        ax.tick_params(axis="y", colors=_MUTED)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("Score (%)", fontsize=11, color=_INK)

    handles = []
    if any_ci:
        handles.append(Line2D([0], [0], color=_INK, marker="|", markersize=10, linewidth=1.2,
                              label="95% CI"))
    if any_chance:
        handles.append(Line2D([0], [0], color=_INK, linewidth=1.3, linestyle=(0, (4, 3)),
                              label="Chance level (1/#options; 50% yes/no)"))
    if any_incomplete:
        handles.append(Patch(facecolor="#dddddd", edgecolor=_INCOMPLETE, hatch="///",
                             label="Incomplete run (not all items answered)"))

    fig.suptitle("Medical AI Benchmark Results", fontsize=15, fontweight="bold", color=_INK)
    fig.text(0.5, 0.93, f"Model: {model_name}", ha="center", fontsize=10, color=_MUTED)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9.5,
                   frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.07 if handles else 0.03, 1, 0.93])

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    root, ext = os.path.splitext(out_path)
    tmp_path = f"{root}.tmp{os.getpid()}{ext or '.png'}"
    try:
        fig.savefig(tmp_path, dpi=150)
        os.replace(tmp_path, out_path)
    finally:
        plt.close(fig)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    print(f"  Chart saved: {out_path}")

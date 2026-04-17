"""
Benchmark results bar chart generator.
"""
import os

# Each bar: (benchmark_key, metric_key, x-label, dataset_group)
# dataset_group drives the bar color and sub-separators within the VLM section
_BAR_SPEC = [
    ("medqa",           "accuracy_pct",             "MedQA\n(Accuracy)",              "Text MCQ"),
    ("rar",             "accuracy_pct",             "RaR\n(Accuracy)",                "Text MCQ"),
    ("radiorag",        "accuracy_pct",             "RadioRAG\n(Accuracy)",           "Text MCQ"),
    ("radbench",        "mcq_accuracy_pct",         "RadBench\nMCQ Acc.",             "RadBench"),
    ("radbench",        "yes_no_accuracy_pct",      "RadBench\nYes/No Acc.",          "RadBench"),
    ("radbench",        "open_wbss_pct",            "RadBench\nOpen WBSS",            "RadBench"),
    ("radbench",        "open_judge_accuracy_pct",  "RadBench\nOpen Judge",           "RadBench"),
    ("vqa_med_2019",    "open_wbss_pct",            "VQA-Med\nWBSS",                  "VQA-Med-2019"),
    ("vqa_med_2019",    "open_judge_accuracy_pct",  "VQA-Med\nLLM-Judge",             "VQA-Med-2019"),
    ("radimagenet_vqa", "mcq_accuracy_pct",         "RadImageNet\nMCQ Acc.",          "RadImageNet-VQA"),
    ("radimagenet_vqa", "yes_no_accuracy_pct",      "RadImageNet\nYes/No Acc.",       "RadImageNet-VQA"),
    ("radimagenet_vqa", "open_wbss_pct",            "RadImageNet\nOpen WBSS",         "RadImageNet-VQA"),
    ("radimagenet_vqa", "open_judge_accuracy_pct",  "RadImageNet\nOpen Judge",        "RadImageNet-VQA"),
    ("label_extraction","micro_f1_pct",             "Label Extraction\n(Micro F1)",   "NER"),
]

# Color per dataset group
_COLORS = {
    "Text MCQ":        "#4C72B0",   # blue
    "RadBench":        "#55A868",   # green
    "VQA-Med-2019":    "#C44E52",   # red
    "RadImageNet-VQA": "#8172B2",   # purple
    "NER":             "#CCB974",   # yellow
}

# Top-level section each dataset belongs to (for background shading)
_SECTION = {
    "Text MCQ":        "Text MCQ",
    "RadBench":        "VLM",
    "VQA-Med-2019":    "VLM",
    "RadImageNet-VQA": "VLM",
    "NER":             "NER",
}

_SECTION_BG = {
    "Text MCQ": "#EEF2FF",
    "VLM":      "#F0FFF4",
    "NER":      "#FFF8EE",
}

# Hardcoded random baselines per metric (None = no line drawn)
_BASELINES = {
    "accuracy_pct":            25.0,   # 4-choice MCQ
    "mcq_accuracy_pct":        25.0,
    "yes_no_accuracy_pct":     50.0,
    "open_wbss_pct":           None,
    "open_judge_accuracy_pct": None,
    "micro_f1_pct":            None,
}


def generate_results_chart(summary: list, model_name: str, out_path: str) -> None:
    """
    Generate a grouped bar chart from benchmark summary.

    summary: list of (benchmark_name, metrics_dict, error_or_None)
    model_name: string shown in chart subtitle
    out_path: file path for the saved PNG

    Random baseline lines are hardcoded per metric type:
      - MCQ Accuracy (4-choice): 25%
      - Yes/No Accuracy (binary): 50%
      - WBSS / LLM-Judge / F1: no baseline line (no obvious random baseline)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not installed — skipping chart generation.")
        return

    # Build lookup: benchmark_key → metrics_dict
    metrics_by_bench = {name.lower(): m for name, m, err in summary if not err and m}

    # Collect bars that have data
    bars = []
    for bench, metric_key, label, group in _BAR_SPEC:
        val = metrics_by_bench.get(bench, {}).get(metric_key)
        if val is not None:
            bars.append({
                "label":      label,
                "value":      float(val),
                "group":      group,
                "metric_key": metric_key,
                "color":      _COLORS[group],
                "section":    _SECTION[group],
                "baseline":   _BASELINES.get(metric_key),
            })

    if not bars:
        print("  No metrics to plot.")
        return

    n = len(bars)
    # Extra horizontal space: wider bars with gaps
    fig_width = max(14, n * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    positions = list(range(n))
    bar_width = 0.6

    # --- Background section shading ---
    section_spans = {}
    for i, b in enumerate(bars):
        sec = b["section"]
        if sec not in section_spans:
            section_spans[sec] = [i, i]
        else:
            section_spans[sec][1] = i

    for sec, (start, end) in section_spans.items():
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.12,
                   color=_SECTION_BG[sec], zorder=0)

    # --- Dataset sub-separators (thin dashed lines between dataset groups) ---
    prev_group = bars[0]["group"]
    for i, b in enumerate(bars[1:], start=1):
        if b["group"] != prev_group:
            ax.axvline(i - 0.5, color="#888888", linewidth=1.0,
                       linestyle="--", alpha=0.6, zorder=2)
        prev_group = b["group"]

    # --- Bars ---
    rects = ax.bar(
        positions,
        [b["value"] for b in bars],
        width=bar_width,
        color=[b["color"] for b in bars],
        alpha=0.88,
        zorder=3,
        edgecolor="white",
        linewidth=0.6,
    )

    # --- Value labels above bars ---
    for rect, b in zip(rects, bars):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.9,
            f"{b['value']:.1f}%",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color="#222222",
        )

    # --- Random baseline lines (per bar, only where defined) ---
    baseline_drawn = set()
    for i, b in enumerate(bars):
        bl = b["baseline"]
        if bl is not None:
            ax.hlines(bl, i - bar_width / 2, i + bar_width / 2,
                      colors="red", linewidths=1.4, linestyles="--", zorder=5)
            baseline_drawn.add(bl)

    # --- Section labels at top of plot ---
    for sec, (start, end) in section_spans.items():
        mid = (start + end) / 2
        ax.text(mid, 104, sec,
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color={"Text MCQ": "#4C72B0", "VLM": "#2a7a4a", "NER": "#9a7a10"}[sec])

    # --- Dataset labels inside shaded areas (below section label) ---
    group_spans = {}
    for i, b in enumerate(bars):
        g = b["group"]
        if g not in group_spans:
            group_spans[g] = [i, i]
        else:
            group_spans[g][1] = i

    for grp, (start, end) in group_spans.items():
        if _SECTION[grp] == "VLM":   # only label sub-groups within VLM
            mid = (start + end) / 2
            ax.text(mid, 99.5, grp,
                    ha="center", va="bottom",
                    fontsize=7.5, style="italic",
                    color=_COLORS[grp], alpha=0.9)

    # --- Axes ---
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [b["label"] for b in bars],
        fontsize=8.2,
        ha="center",
        multialignment="center",
    )
    ax.tick_params(axis="x", pad=6)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Title ---
    ax.set_title("Medical AI Benchmark Results", fontsize=14,
                 fontweight="bold", pad=28)
    ax.text(0.5, 1.025, f"Model: {model_name}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#555555")

    # --- Legend ---
    handles = [
        mpatches.Patch(color=col, alpha=0.88, label=grp)
        for grp, col in _COLORS.items()
    ]
    if baseline_drawn:
        handles.append(plt.Line2D([0], [0], color="red", linewidth=1.4,
                                  linestyle="--", label="Random baseline\n(25% MCQ / 50% Yes-No)"))
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.92, edgecolor="#cccccc", title="Dataset", title_fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")

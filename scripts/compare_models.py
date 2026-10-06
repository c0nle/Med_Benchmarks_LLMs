"""
Compare several benchmark runs (typically one per model).

    python scripts/compare_models.py results/run_A results/run_B [results/run_C ...]
        [--labels "Model A,Model B"] [--out-dir results/comparison_<ts>]

Reads per run dir: {benchmark}_report.jsonl (metric rows), {benchmark}_status.json
and the model name from run_info_*.json (fallback: fingerprint.json, dir name).

Writes to --out-dir:
  comparison.csv  benchmark, metric, model, value, ci_lo, ci_hi, n, complete
                  (all numeric report metrics; old key citation_match_pct is
                  reported as verbatim_citation_rate_pct)
  comparison.png  grouped bars of the headline metrics, one colour per model
                  (order of the run dirs), 95% CI where available, incomplete
                  benchmarks hatched
  mcnemar.csv     paired exact McNemar test (each run vs. the first run) on items
                  shared by both runs, for MCQ / yes-no subsets (is_correct) and
                  open questions (judge_correct), read from the report item rows
"""
import argparse
import csv
import datetime
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.plot import _PANELS, _DISPLAY, sample_size, _num   # noqa: E402
from core.summary import load_run_summary                     # noqa: E402

# Categorical colours in fixed order (one per model)
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

METRIC_RENAMES = {"citation_match_pct": "verbatim_citation_rate_pct"}
_SKIP_SUFFIXES = ("_ci_lo", "_ci_hi", "_ci_method", "_n")
_STATUS = {"n_items", "n_expected", "n_api_errors", "complete", "stop_reason"}

# Benchmarks with binary per-item correctness for the paired test
_PAIRED = {"medqa", "rar", "radiorag", "radbench", "vqa_med_2019", "radimagenet_vqa"}


def model_name_of(run_dir: str) -> str:
    infos = sorted(glob.glob(os.path.join(run_dir, "run_info_*.json")), key=os.path.getmtime)
    for path in reversed(infos):
        try:
            with open(path, encoding="utf-8") as f:
                name = json.load(f).get("config", {}).get("server", {}).get("model_name")
            if name:
                return str(name)
        except (OSError, ValueError):
            continue
    fp = os.path.join(run_dir, "fingerprint.json")
    if os.path.exists(fp):
        try:
            with open(fp, encoding="utf-8") as f:
                name = json.load(f).get("model_name")
            if name:
                return str(name)
        except (OSError, ValueError):
            pass
    return os.path.basename(os.path.normpath(run_dir))


def _renamed(metrics: dict) -> dict:
    out = dict(metrics)
    for old, new in METRIC_RENAMES.items():
        for suffix in ("",) + _SKIP_SUFFIXES:
            if old + suffix in out and new + suffix not in out:
                out[new + suffix] = out.pop(old + suffix)
    return out


def comparison_rows(runs: list) -> list:
    """runs: [(model_label, run_dir)] -> list of CSV row dicts."""
    rows = []
    for label, run_dir in runs:
        for bench, metrics, _ in load_run_summary(run_dir):
            metrics = _renamed(metrics)
            for key, value in metrics.items():
                if key in _STATUS or key.endswith(_SKIP_SUFFIXES):
                    continue
                v = _num(value)
                if v is None:
                    continue
                rows.append({
                    "benchmark": bench, "metric": key, "model": label, "value": v,
                    "ci_lo": _num(metrics.get(f"{key}_ci_lo")),
                    "ci_hi": _num(metrics.get(f"{key}_ci_hi")),
                    "n": sample_size(metrics, key),
                    "complete": metrics.get("complete"),
                })
    return rows


# ---------------------------------------------------------------------------
# Paired McNemar test
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def item_correctness(report_path: str) -> dict:
    """{(subset, id): 0/1} from the report item rows (is_correct, or judge_correct
    for open questions). Only id/subset/correctness are kept in memory."""
    out = {}
    if not os.path.exists(report_path):
        return out
    with open(report_path, encoding="utf-8") as f:
        for line in f:
            if '"type": "item"' not in line[:40]:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            subset = row.get("subset") or "all"
            val = row.get("judge_correct") if subset == "open" else row.get("is_correct")
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            out[(subset, str(row.get("id")))] = int(bool(val))
    return out


def paired_tests(runs: list) -> list:
    if len(runs) < 2:
        return []
    ref_label, ref_dir = runs[0]
    rows = []
    for bench in sorted(_PAIRED):
        ref = item_correctness(os.path.join(ref_dir, f"{bench}_report.jsonl"))
        if not ref:
            continue
        for label, run_dir in runs[1:]:
            other = item_correctness(os.path.join(run_dir, f"{bench}_report.jsonl"))
            for subset in sorted({s for s, _ in ref}):
                shared = [k for k in ref if k[0] == subset and k in other]
                if not shared:
                    continue
                a = sum(ref[k] for k in shared)
                b_ = sum(other[k] for k in shared)
                only_a = sum(1 for k in shared if ref[k] == 1 and other[k] == 0)
                only_b = sum(1 for k in shared if ref[k] == 0 and other[k] == 1)
                rows.append({
                    "benchmark": bench, "subset": subset, "model_a": ref_label, "model_b": label,
                    "n_shared": len(shared),
                    "acc_a_pct": round(100 * a / len(shared), 2),
                    "acc_b_pct": round(100 * b_ / len(shared), 2),
                    "only_a_correct": only_a, "only_b_correct": only_b,
                    "p_value": round(mcnemar_exact(only_a, only_b), 6),
                })
    return rows


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def comparison_chart(runs: list, out_path: str) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    labels = [label for label, _ in runs]
    summaries = [{b: _renamed(m) for b, m, _ in load_run_summary(d)} for _, d in runs]

    panels = []
    for title, _, spec in _PANELS:
        groups = []
        for bench, key, label, _chance in spec:
            vals = [s.get(bench, {}) for s in summaries]
            if any(_num(m.get(key)) is not None for m in vals):
                groups.append((bench, key, f"{_DISPLAY.get(bench, bench)}\n{label}", vals))
        if groups:
            panels.append((title, groups))
    if not panels:
        print("No headline metrics found in the given run dirs.")
        return False
    if len(labels) > len(MODEL_COLORS):
        print(f"Note: more than {len(MODEL_COLORS)} models – colours repeat; compare fewer runs at once.")

    n_models = len(labels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 4.2 * len(panels) + 1.2), squeeze=False)
    width = 0.8 / n_models
    any_ci = any_incomplete = False
    for ax, (title, groups) in zip(axes[:, 0], panels):
        for gi, (bench, key, glabel, vals) in enumerate(groups):
            for mi, m in enumerate(vals):
                v = _num(m.get(key))
                if v is None:
                    continue
                x = gi - 0.4 + width * (mi + 0.5)
                incomplete = m.get("complete") is False
                any_incomplete |= incomplete
                ax.bar(x, v, width=width * 0.92, color=MODEL_COLORS[mi % len(MODEL_COLORS)],
                       alpha=0.45 if incomplete else 0.9, hatch="///" if incomplete else None,
                       edgecolor="#c0392b" if incomplete else "white", linewidth=0.8, zorder=3)
                top = v
                lo, hi = _num(m.get(f"{key}_ci_lo")), _num(m.get(f"{key}_ci_hi"))
                if lo is not None and hi is not None:
                    ax.errorbar(x, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="none",
                                ecolor="#222222", elinewidth=1.0, capsize=2.5, zorder=4)
                    top = max(top, hi)
                    any_ci = True
                ax.text(x, top + 1.0, f"{v:.1f}", ha="center", va="bottom", rotation=90,
                        fontsize=7 if n_models > 2 else 8, color="#222222", zorder=5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[2] for g in groups], fontsize=8.5)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(0, 118)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Score (%)")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.yaxis.grid(True, color="#e3e3e3", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles = [Patch(facecolor=MODEL_COLORS[i % len(MODEL_COLORS)], label=lab) for i, lab in enumerate(labels)]
    if any_ci:
        handles.append(Line2D([0], [0], color="#222222", marker="|", markersize=9, linewidth=1.0, label="95% CI"))
    if any_incomplete:
        handles.append(Patch(facecolor="#dddddd", edgecolor="#c0392b", hatch="///", label="Incomplete run"))
    top = 1 - 0.75 / fig.get_figheight()          # space for title + legend (inches -> fraction)
    fig.suptitle("Model comparison", fontsize=14, fontweight="bold", y=1 - 0.12 / fig.get_figheight())
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.38 / fig.get_figheight()), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _write_csv(path: str, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark runs of several models.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories (results/run_<ts>), one per model.")
    parser.add_argument("--labels", help="Comma-separated model labels (default: model_name from run_info).")
    parser.add_argument("--out-dir", help="Output directory (default: results/comparison_<timestamp>).")
    args = parser.parse_args(argv)

    for d in args.run_dirs:
        if not os.path.isdir(d):
            parser.error(f"not a directory: {d}")
    labels = ([s.strip() for s in args.labels.split(",")] if args.labels
              else [model_name_of(d) for d in args.run_dirs])
    if len(labels) != len(args.run_dirs):
        parser.error("--labels must name every run dir")
    seen = {}
    for i, lab in enumerate(labels):      # keep labels unique (same model twice)
        if lab in seen:
            labels[i] = f"{lab} ({os.path.basename(os.path.normpath(args.run_dirs[i]))})"
        seen[lab] = True
    runs = list(zip(labels, args.run_dirs))

    out_dir = args.out_dir or os.path.join(
        "results", "comparison_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    rows = comparison_rows(runs)
    _write_csv(os.path.join(out_dir, "comparison.csv"), rows,
               ["benchmark", "metric", "model", "value", "ci_lo", "ci_hi", "n", "complete"])
    print(f"comparison.csv: {len(rows)} rows")

    tests = paired_tests(runs)
    if tests:
        _write_csv(os.path.join(out_dir, "mcnemar.csv"), tests, list(tests[0].keys()))
        print(f"mcnemar.csv: {len(tests)} paired tests (vs. {runs[0][0]})")

    if comparison_chart(runs, os.path.join(out_dir, "comparison.png")):
        print(f"Chart: {os.path.join(out_dir, 'comparison.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

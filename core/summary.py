"""
Flatten benchmark reports into summary dicts and read/write per-benchmark status.

Flat key scheme for report rows {"type": "metric", "metric": m, "value": v, ...}:
    <qualifiers joined by "_">_<metric>
where the qualifiers are the row's string-valued fields in the order
subset, field, region, lesion_view, then any other string field (sorted),
excluding type/metric/note/ci_method/unit. Characters outside [A-Za-z0-9] become "_".
Special cases (backward compatible with the dicts evaluate.py returns):
  * lesion_view == "type" (Mamma main lesion view) is not part of the key
  * metric "llm_judge_accuracy_pct" is flattened as "judge_accuracy_pct"
  * rows of type "region_metric" give "<region>_<key>" for every *_pct field
Examples:
  {"metric": "accuracy_pct"}                                 -> accuracy_pct
  {"subset": "mcq", "metric": "accuracy_pct"}                -> mcq_accuracy_pct
  {"subset": "open:category=plane", "metric": "accuracy_pct"}-> open_category_plane_accuracy_pct
  {"field": "birads_li", "metric": "accuracy_pct"}           -> birads_li_accuracy_pct
  {"field": "lesions_li", "lesion_view": "count", ...}       -> lesions_li_count_micro_f1_pct
CI bounds become <flatkey>_ci_lo / <flatkey>_ci_hi (and <flatkey>_ci_method),
a sample size ("n", "n_judged", "n_sides", "n_scored") becomes <flatkey>_n.
"""
import datetime
import json
import math
import os
import re

STATUS_KEYS = ("n_items", "n_expected", "n_api_errors", "complete")

# judge_model: informational; variant: the tag is already part of the metric name
_NON_QUALIFIERS = {"type", "metric", "note", "ci_method", "unit", "description",
                   "judge_model", "variant"}
_QUALIFIER_ORDER = ("subset", "field", "region", "lesion_view")
_METRIC_ALIASES = {"llm_judge_accuracy_pct": "judge_accuracy_pct"}
_N_FIELDS = ("n", "n_judged", "n_sides", "n_scored")


def _clean(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (
        isinstance(v, float) and math.isnan(v))


def flat_key(row: dict) -> str:
    """Flat summary key of one metric row (see module docstring)."""
    quals = []
    for q in _QUALIFIER_ORDER:
        v = row.get(q)
        if isinstance(v, str) and v and not (q == "lesion_view" and v == "type"):
            quals.append(_clean(v))
    for k in sorted(row):
        v = row[k]
        if k in _QUALIFIER_ORDER or k in _NON_QUALIFIERS or not isinstance(v, str) or not v:
            continue
        quals.append(_clean(v))
    metric = _METRIC_ALIASES.get(row.get("metric"), row.get("metric"))
    return "_".join([q for q in quals if q] + [_clean(metric)])


def flatten_report_rows(rows) -> dict:
    """Flatten an iterable of report rows (dicts) into {flatkey: value}."""
    flat = {}
    for row in rows:
        rtype = row.get("type")
        if rtype == "metric" and row.get("metric") is not None:
            key = flat_key(row)
            if key in flat:
                # never let a differently-qualified row overwrite an earlier one silently
                i = 2
                while f"{key}__{i}" in flat:
                    i += 1
                key = f"{key}__{i}"
            value = row.get("value")
            if _is_number(value):
                flat[key] = value
            for bound in ("ci_lo", "ci_hi"):
                if _is_number(row.get(bound)):
                    flat[f"{key}_{bound}"] = row[bound]
            if isinstance(row.get("ci_method"), str):
                flat[f"{key}_ci_method"] = row["ci_method"]
            for nf in _N_FIELDS:
                if _is_number(row.get(nf)):
                    flat[f"{key}_n"] = row[nf]
                    break
        elif rtype == "region_metric" and isinstance(row.get("region"), str):
            region = _clean(row["region"])
            for k, v in row.items():
                if k.endswith("_pct") and _is_number(v):
                    flat.setdefault(f"{region}_{k}", v)
    return flat


def read_report_rows(report_path: str, types=("metric", "region_metric")):
    """Yield report rows of the given types. Item rows are skipped without
    keeping them in memory (they can contain patient-derived text)."""
    if not (report_path and os.path.exists(report_path)):
        return
    with open(report_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # cheap pre-filter: item rows are by far the most frequent
            if '"type": "item"' in line[:40]:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("type") in types:
                yield row


def flatten_report(report_path: str) -> dict:
    return flatten_report_rows(read_report_rows(report_path))


def merge_metrics(returned: dict, flat: dict) -> dict:
    """Headline dict returned by evaluate.py, completed with CIs, sample sizes
    and all further flattened report metrics. Returned values win."""
    merged = {k: v for k, v in (returned or {}).items() if k != "path"}
    for k, v in flat.items():
        merged.setdefault(k, v)
    return merged


# ---------------------------------------------------------------------------
# Status files
# ---------------------------------------------------------------------------

def status_path(run_dir: str, benchmark: str) -> str:
    return os.path.join(run_dir, f"{benchmark}_status.json")


def build_status(n_items: int, n_expected: int, n_api_errors: int, stop_reason: str = None) -> dict:
    complete = stop_reason is None and n_items >= n_expected
    return {
        "n_items": int(n_items),
        "n_expected": int(n_expected),
        "n_api_errors": int(n_api_errors),
        "complete": bool(complete),
        "stop_reason": stop_reason if stop_reason else (
            None if complete else f"only {n_items}/{n_expected} items answered"),
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def write_status(run_dir: str, benchmark: str, status: dict) -> str:
    from core.fileio import atomic_write_json
    path = status_path(run_dir, benchmark)
    atomic_write_json(path, status)
    return path


def read_status(run_dir: str, benchmark: str):
    path = status_path(run_dir, benchmark)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_run_summary(run_dir: str, benchmarks=None) -> list:
    """Summary list [(benchmark, metrics, None)] for every benchmark of *run_dir*
    that has a report (optionally restricted to *benchmarks*). Metrics are the
    flattened report plus the status keys; without a status file the benchmark
    counts as complete=None (unknown)."""
    out = []
    names = sorted(f[: -len("_report.jsonl")] for f in os.listdir(run_dir)
                   if f.endswith("_report.jsonl"))
    for name in names:
        if benchmarks is not None and name not in benchmarks:
            continue
        metrics = flatten_report(os.path.join(run_dir, f"{name}_report.jsonl"))
        status = read_status(run_dir, name) or {}
        for k in STATUS_KEYS + ("stop_reason",):
            metrics[k] = status.get(k)
        out.append((name, metrics, None))
    return out


def format_metrics_line(metrics: dict, headline_keys=None) -> str:
    """Terminal summary: status flag plus headline metrics with CIs."""
    parts = []
    if metrics.get("complete") is False:
        parts.append(f"INCOMPLETE ({metrics.get('n_items')}/{metrics.get('n_expected')} items"
                     + (f"; {metrics['stop_reason']}" if metrics.get("stop_reason") else "") + ")")
    keys = headline_keys if headline_keys is not None else [
        k for k in metrics
        if not k.endswith(("_ci_lo", "_ci_hi", "_ci_method", "_n")) and k != "stop_reason"]
    for k in keys:
        v = metrics.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            s = f"{k}: {v:.2f}%" if k.endswith("_pct") else f"{k}: {v:.4g}"
            lo, hi = metrics.get(f"{k}_ci_lo"), metrics.get(f"{k}_ci_hi")
            if lo is not None and hi is not None:
                s += f" [{lo:.1f}–{hi:.1f}]"
        else:
            s = f"{k}: {v}"
        parts.append(s)
    return "  ".join(parts)

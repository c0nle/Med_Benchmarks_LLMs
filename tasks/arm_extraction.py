"""
Arm-Röntgen Label Extraction Task Runner.
(Kreutzer et al., Eur Radiol 2025)

Das Modell erhält einen (deutschsprachigen) Radiologiebericht und soll für jedes Label
angeben, ob es vorhanden ist, mit Zitatbeleg.

Output-Schema:
{
  "Label Name": {"finding": true|false, "citation": "<Zitat aus Bericht>"},
  ...
}

Results-CSV-Spalten:
    id, benchmark, region, phase, gt_labels_json, model_raw, model_labels_json, parse_error
"""
import csv
import json
import os
import re
import time

import pandas as pd

from tasks.mcq import _parse_benchmark_settings

_FIELDNAMES = [
    "id", "benchmark", "region", "phase",
    "gt_labels_json", "model_raw", "model_labels_json", "parse_error",
    "citation_check_json",
]


_MIN_CITATION_CHARS = 4


def _norm_for_match(text: str) -> str:
    text = text.lower().replace("_x000d_", " ")
    text = re.sub(r"[\"'“”‘’`„«»]", "", text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def check_citations(parsed: dict, report_text: str) -> dict:
    """
    For every label the model marks as present and backs with a citation:
    True if the citation occurs verbatim (case/whitespace/quote-insensitive) in the report.
    Fragments joined by "..." must each occur.
    """
    norm_report = _norm_for_match(report_text)
    result = {}
    for label, entry in parsed.items():
        if not isinstance(entry, dict) or not entry.get("finding"):
            continue
        citation = str(entry.get("citation") or "").strip()
        if not citation:
            continue
        fragments = [
            _norm_for_match(frag).strip(" .,;:")
            for frag in re.split(r"\.\.\.|…", citation)
        ]
        fragments = [frag for frag in fragments if frag]
        # Trivially short quotes ("a") would always match; count them as failed
        result[label] = (
            bool(fragments)
            and sum(len(frag) for frag in fragments) >= _MIN_CITATION_CHARS
            and all(frag in norm_report for frag in fragments)
        )
    return result


def _build_prompt(text: str, template_labels: list) -> str:
    entries = "\n".join(
        f'  "{label}": {{"finding": true/false, "citation": "..."}}'
        for label in template_labels
    )
    return (
        "You are a radiology report analysis assistant. "
        "Analyze the following radiology report and extract findings. "
        "The report may be written in German; the label names are in English.\n\n"
        f"Report:\n{text}\n\n"
        "For each label below, determine if the finding is present (true) or absent (false). "
        "Provide a short direct quote from the report that supports your decision, "
        "or an empty string if no relevant text exists.\n\n"
        "Return valid JSON with exactly this structure:\n"
        "{\n"
        f"{entries}\n"
        "}\n\n"
        "Rules:\n"
        '- Set "finding" to true only if the finding is explicitly stated or strongly implied\n'
        '- "citation" must be a verbatim excerpt from the report above (original language, not translated)\n'
        "- Return only valid JSON, no explanations."
    )


def _as_bool(value) -> bool:
    """Only true / 1 / "true" / "yes" / "1" count as a positive finding."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in ("true", "yes", "1", "ja")


def _parse_response(raw: str, template_labels: list) -> tuple[dict, bool]:
    """
    Parst JSON-Antwort des Modells.
    Normalisiert auf template_labels; gibt (parsed_dict, parse_error) zurück.
    """
    if not raw or raw.startswith("Error:"):
        return {}, True

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)

    def _normalize(parsed) -> tuple[dict, bool]:
        if not isinstance(parsed, dict):
            return {}, True
        # Unwrap {"findings": {...}}-style answers
        if len(parsed) == 1:
            inner = next(iter(parsed.values()))
            if isinstance(inner, dict) and not any(k in parsed for k in template_labels):
                parsed = inner
        by_lower = {str(k).strip().lower(): v for k, v in parsed.items()}
        result = {}
        n_found = 0
        for label in template_labels:
            entry = by_lower.get(label.lower())
            if entry is not None:
                n_found += 1
            if isinstance(entry, dict):
                result[label] = {
                    "finding":  _as_bool(entry.get("finding")),
                    "citation": str(entry.get("citation") or ""),
                }
            else:
                result[label] = {"finding": _as_bool(entry), "citation": ""}
        # Valid JSON that answers less than half of the labels is not a usable answer
        return result, n_found < len(template_labels) / 2

    try:
        return _normalize(json.loads(text))
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return _normalize(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass

    return {}, True


# ---------------------------------------------------------------------------
# Task Runner
# ---------------------------------------------------------------------------

def run(config: dict, client, data: list, results_path: str, logger=None) -> str:
    sleep_s, max_errors = _parse_benchmark_settings(config)

    completed_ids: set = set()
    if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
        try:
            existing = pd.read_csv(results_path, usecols=["id"])
            completed_ids = set(existing["id"].dropna().astype(str).tolist())
            if completed_ids:
                print(f"Resume: {len(completed_ids)} Reports bereits vorhanden.")
        except Exception:
            pass

    total     = len(data)
    remaining = sum(1 for it in data if str(it.get("id")) not in completed_ids)
    print(f"  {total} Reports  ({remaining} remaining)...")

    start         = time.time()
    processed_new = 0
    errors        = 0

    file_exists = os.path.exists(results_path) and os.path.getsize(results_path) > 0
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for idx, item in enumerate(data, start=1):
            item_id = str(item.get("id"))
            if item_id in completed_ids:
                continue

            template_labels = item.get("template_labels", [])
            prompt          = _build_prompt(item["text"], template_labels)
            model_answer    = client.ask_question(prompt)

            is_error = isinstance(model_answer, str) and model_answer.startswith("Error:")
            if is_error:
                errors += 1

            parsed, parse_error = _parse_response(model_answer, template_labels)

            if logger:
                status = "ERROR" if is_error else f"parse_err={parse_error}"
                logger.verbose(
                    f"[{idx:>{len(str(total))}}/{total}] {item_id}  →  {status}"
                )

            if is_error and max_errors is not None and errors >= max_errors:
                print(f"Abbruch: max_errors={max_errors} erreicht.")
                break

            writer.writerow({
                "id":              item_id,
                "benchmark":       item.get("benchmark", "LabelExtractionArm"),
                "region":          item.get("region", ""),
                "phase":           item.get("phase", ""),
                "gt_labels_json":  json.dumps(item.get("gt_labels", {}), ensure_ascii=False),
                "model_raw":       model_answer or "",
                "model_labels_json": json.dumps(parsed, ensure_ascii=False),
                "parse_error":     str(parse_error),
                "citation_check_json": json.dumps(
                    check_citations(parsed, item["text"]), ensure_ascii=False
                ),
            })
            f.flush()
            processed_new += 1

            if sleep_s > 0:
                time.sleep(sleep_s)

            if processed_new % 50 == 0:
                elapsed = time.time() - start
                rate    = processed_new / elapsed if elapsed > 0 else 0.0
                eta_s   = int((remaining - processed_new) / rate) if rate > 0 else -1
                eta     = f"{eta_s//60:02d}:{eta_s%60:02d}" if eta_s >= 0 else "?"
                pct     = int(processed_new / remaining * 100) if remaining > 0 else 100
                print(
                    f"  [{processed_new:>{len(str(remaining))}}/{remaining}]"
                    f" {pct:3d}%  {rate:.1f} q/s  ETA {eta}  errors: {errors}"
                )

    elapsed_total = time.time() - start
    print(f"  Done: {processed_new}/{remaining}  errors: {errors}  ({elapsed_total/60:.1f} min)")
    return results_path

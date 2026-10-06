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
    id, benchmark, region, phase, gt_labels_json, model_raw, model_labels_json, parse_error,
    citation_check_json, finish_reason, completion_tokens

model_labels_json: {label: {"finding": true|false|null, "citation": str}}; null = the label
is missing in the model answer (scored as missing, not as negative).
citation_check_json: {label: bool} for every label the model marked present with a
citation: True if the citation occurs verbatim in the report (checked at run time,
because the report text is not stored).
finish_reason / completion_tokens come from client.last_meta (empty if not provided).
"""
import json
import re

from tasks import _extraction_runner as _runner

_FIELDNAMES = [
    "id", "benchmark", "region", "phase",
    "gt_labels_json", "model_raw", "model_labels_json", "parse_error",
    "citation_check_json", "finish_reason", "completion_tokens",
]

# Explicit system prompt: the client's default one asks for concise *English* answers,
# which conflicts with verbatim citations from German reports. No language constraint
# here; the user prompt asks for citations in the original language.
_SYSTEM_PROMPT_ARM = (
    "You are a medical AI assistant specialised in the structured analysis of "
    "radiology reports. Reply only with valid JSON, without explanatory text."
)


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
    # Label names only: no authoritative label definitions / annotation guidelines exist
    # in the dataset folder (templates contain names only), so none are invented here.
    entries = ",\n".join(
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
            if entry is None:
                # Label not answered: keep it as missing (finding None), not as negative
                result[label] = {"finding": None, "citation": ""}
            elif isinstance(entry, dict):
                finding = entry.get("finding")
                result[label] = {
                    "finding":  None if finding is None else _as_bool(finding),
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

def _process_item(client, item: dict):
    """Runs in a worker thread: ask, parse, check citations against the report text."""
    item_id = str(item.get("id"))
    template_labels = item.get("template_labels", [])
    prompt = _build_prompt(item["text"], template_labels)
    model_answer, meta = _runner.call_model(client, prompt, _SYSTEM_PROMPT_ARM)

    is_error = isinstance(model_answer, str) and model_answer.startswith("Error:")
    parsed, parse_error = _parse_response(model_answer, template_labels)

    row = {
        "id":              item_id,
        "benchmark":       item.get("benchmark", "LabelExtractionArm"),
        "region":          item.get("region", ""),
        "phase":           item.get("phase", ""),
        "gt_labels_json":  json.dumps(item.get("gt_labels", {}), ensure_ascii=False),
        "model_raw":       model_answer or "",
        "model_labels_json": json.dumps(parsed, ensure_ascii=False),
        "parse_error":     str(parse_error),
        # Needs the report text, so it is computed here (the text is not stored in the CSV)
        "citation_check_json": json.dumps(
            check_citations(parsed, item["text"]), ensure_ascii=False
        ),
        **_runner.meta_columns(meta),
    }
    status = "ERROR" if is_error else f"parse_err={parse_error}"
    if meta.get("finish_reason") == "length":
        status += " (truncated)"
    return row, is_error, status


def run(config: dict, client, data: list, results_path: str, logger=None) -> str:
    """Concurrent + resumable; see tasks/_extraction_runner.py."""
    return _runner.run_items(
        config, data, results_path, _FIELDNAMES,
        lambda item: _process_item(client, item),
        logger=logger, unit="reports",
    )

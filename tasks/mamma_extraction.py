"""
Mamma-MRT label-extraction task runner.

The model receives a German breast MRI report and returns structured labels as JSON
(the prompt is German, see _build_prompt).

Output schema expected from the model:
{
  "menopause": "prä" | "post" | "peri" | null,
  "links": {
    "birads": 2–5 | null,
    "acr":    1–4 | null,
    "lesionen": ["Typ1", "Typ2", ...]
  },
  "rechts": { ... }
}

Results CSV columns:
    id, benchmark,
    gt_menopause, gt_birads_li, gt_birads_re, gt_acr_li, gt_acr_re,
    gt_lesions_li, gt_lesions_re,
    model_raw, model_menopause, model_birads_li, model_birads_re,
    model_acr_li, model_acr_re, model_lesions_li, model_lesions_re,
    parse_error, finish_reason, completion_tokens

finish_reason / completion_tokens come from client.last_meta (empty if the client does
not provide it); finish_reason == "length" means the JSON answer was truncated.
"""
import json
import re

from tasks import _extraction_runner as _runner

_SYSTEM_PROMPT_DE = (
    "Du bist ein medizinischer KI-Assistent, spezialisiert auf die strukturierte "
    "Auswertung von deutschen Mamma-MRT-Befunden. "
    "Antworte ausschließlich mit validem JSON ohne erklärenden Text."
)

_LESION_VOCABULARY = [
    "invasives Karzinom", "DCIS", "Fibroadenom", "Mastopathie", "Adenose",
    "Lymphknoten", "Papillom", "Zyste", "Hamartom", "Lipom",
    "Hämatom", "Ödem", "Narbe", "Atherom", "unspezifische Anreicherung",
    "hormonelle Stimulation", "Mamille", "sonstige Läsion",
]

_FIELDNAMES = [
    "id", "benchmark",
    "gt_menopause", "gt_birads_li", "gt_birads_re", "gt_acr_li", "gt_acr_re",
    "gt_lesions_li", "gt_lesions_re",
    "model_raw", "model_menopause", "model_birads_li", "model_birads_re",
    "model_acr_li", "model_acr_re", "model_lesions_li", "model_lesions_re",
    "parse_error", "finish_reason", "completion_tokens",
]


def _build_prompt(text: str) -> str:
    vocab = ", ".join(f'"{v}"' for v in _LESION_VOCABULARY)
    return (
        "Analysiere den folgenden deutschen Befundtext einer Mamma-MRT "
        "und extrahiere die strukturierten Informationen.\n\n"
        f"Befundtext:\n{text}\n\n"
        "Gib deine Antwort als valides JSON zurück:\n"
        "{\n"
        '  "menopause": "prä" | "post" | "peri" | null,\n'
        '  "links": {\n'
        '    "birads": 2 | 3 | 4 | 5 | null,\n'
        '    "acr": 1 | 2 | 3 | 4 | null,\n'
        '    "lesionen": [<Typen aus erlaubtem Vokabular>]\n'
        "  },\n"
        '  "rechts": {\n'
        '    "birads": 2 | 3 | 4 | 5 | null,\n'
        '    "acr": 1 | 2 | 3 | 4 | null,\n'
        '    "lesionen": [<Typen aus erlaubtem Vokabular>]\n'
        "  }\n"
        "}\n\n"
        f"Erlaubtes Vokabular für Läsionstypen: {vocab}\n\n"
        "Hinweise:\n"
        "- BIRADS: Kategorie nach dem Bildbefund der jeweiligen Seite: 2=sicher benigne, "
        "3=wahrsch. benigne, 4=suspekt, 5=hochgradig malignitätsverdächtig. Auch bei bereits "
        "histologisch gesichertem Karzinom die Kategorie nach dem Bildbefund angeben "
        "(höchstens 5, keine 6)\n"
        "- ACR/BPE (Hintergrundanreicherung): 1=minimal, 2=mild, 3=moderat, 4=ausgeprägt\n"
        "- Läsionen: pro Seite die im Befund beschriebenen Herdbefunde/Anreicherungen "
        "(Läsionen mit eigener Beurteilung) mit ihrem Typ aus dem Vokabular; "
        "Normalbefunde nicht auflisten; leere Liste, wenn keine Läsion beschrieben ist\n"
        "- Falls eine Seite nicht erwähnt oder nicht beurteilbar: null\n"
        "- Nur valides JSON, keine Erklärungen."
    )


def _parse_response(raw: str) -> tuple[dict, bool]:
    """
    Parse the model's JSON answer.
    Returns (parsed_dict, parse_error: bool).
    """
    if not raw or raw.startswith("Error:"):
        return {}, True

    text = raw.strip()
    # Strip a Markdown code fence
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)

    for candidate in (text, *re.findall(r"\{.*\}", text, re.DOTALL)[:1]):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, False
        return {}, True

    return {}, True


def _extract_side(parsed: dict, side_key: str):
    """Extract birads, acr and lesionen of one side from the parsed dict."""
    side = parsed.get(side_key)
    if not isinstance(side, dict):
        return None, None, []

    birads = side.get("birads")
    acr    = side.get("acr")
    lesions = side.get("lesionen") or side.get("lesions") or []

    try:
        birads_str = str(int(birads)) if birads is not None else None
    except (ValueError, TypeError):
        birads_str = str(birads).strip() if birads else None

    try:
        acr_str = str(int(acr)) if acr is not None else None
    except (ValueError, TypeError):
        acr_str = str(acr).strip() if acr else None

    if not isinstance(lesions, list):
        lesions = []
    clean_lesions = [str(l).strip() for l in lesions if l]

    return birads_str, acr_str, clean_lesions


# ---------------------------------------------------------------------------
# Task Runner
# ---------------------------------------------------------------------------

def _process_item(client, item: dict):
    """Runs in a worker thread: ask the model, parse, build the CSV row."""
    item_id = str(item.get("id"))
    model_answer, meta = _runner.call_model(client, _build_prompt(item["text"]), _SYSTEM_PROMPT_DE)

    is_error = isinstance(model_answer, str) and model_answer.startswith("Error:")
    parsed, parse_error = _parse_response(model_answer)

    model_meno = str(parsed.get("menopause") or "").strip() or ""
    birads_li, acr_li, lesions_li = _extract_side(parsed, "links")
    birads_re, acr_re, lesions_re = _extract_side(parsed, "rechts")

    gt = item.get("gt", {})
    row = {
        "id":             item_id,
        "benchmark":      item.get("benchmark", "LabelExtractionMamma"),
        "gt_menopause":   gt.get("menopause") or "",
        "gt_birads_li":   gt.get("birads_li") or "",
        "gt_birads_re":   gt.get("birads_re") or "",
        "gt_acr_li":      gt.get("acr_li") or "",
        "gt_acr_re":      gt.get("acr_re") or "",
        "gt_lesions_li":  json.dumps(gt.get("lesions_li") or [], ensure_ascii=False),
        "gt_lesions_re":  json.dumps(gt.get("lesions_re") or [], ensure_ascii=False),
        "model_raw":      model_answer or "",
        "model_menopause": model_meno,
        "model_birads_li": birads_li or "",
        "model_birads_re": birads_re or "",
        "model_acr_li":   acr_li or "",
        "model_acr_re":   acr_re or "",
        "model_lesions_li": json.dumps(lesions_li, ensure_ascii=False),
        "model_lesions_re": json.dumps(lesions_re, ensure_ascii=False),
        "parse_error":    str(parse_error),
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
        logger=logger, unit="exams",
    )

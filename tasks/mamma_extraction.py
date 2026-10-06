"""
Mamma-MRT Label Extraction Task Runner.

Das Modell erhält einen deutschen Befundtext und soll strukturierte Labels
als JSON zurückgeben.

Output-Schema des Modells:
{
  "menopause": "prä" | "post" | "peri" | null,
  "links": {
    "birads": 2–6 | null,
    "acr":    1–4 | null,
    "lesionen": ["Typ1", "Typ2", ...]
  },
  "rechts": { ... }
}

Results-CSV-Spalten:
    id, benchmark,
    gt_menopause, gt_birads_li, gt_birads_re, gt_acr_li, gt_acr_re,
    gt_lesions_li, gt_lesions_re,
    model_raw, model_menopause, model_birads_li, model_birads_re,
    model_acr_li, model_acr_re, model_lesions_li, model_lesions_re,
    parse_error
"""
import csv
import json
import os
import re
import time

import pandas as pd

from tasks.mcq import _parse_benchmark_settings

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
    "parse_error",
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
        '    "birads": 2 | 3 | 4 | 5 | 6 | null,\n'
        '    "acr": 1 | 2 | 3 | 4 | null,\n'
        '    "lesionen": [<Typen aus erlaubtem Vokabular>]\n'
        "  },\n"
        '  "rechts": {\n'
        '    "birads": 2 | 3 | 4 | 5 | 6 | null,\n'
        '    "acr": 1 | 2 | 3 | 4 | null,\n'
        '    "lesionen": [<Typen aus erlaubtem Vokabular>]\n'
        "  }\n"
        "}\n\n"
        f"Erlaubtes Vokabular für Läsionstypen: {vocab}\n\n"
        "Hinweise:\n"
        "- BIRADS: 2=sicher benigne, 3=wahrsch. benigne, 4=suspekt, "
        "5=hochgradig maligne, 6=gesicherte Malignität\n"
        "- ACR/BPE (Hintergrundanreicherung): 1=minimal, 2=mild, 3=moderat, 4=ausgeprägt\n"
        "- Läsionen: pro Seite die im Befund beschriebenen Herdbefunde/Anreicherungen "
        "(Läsionen mit eigener Beurteilung) mit ihrem Typ aus dem Vokabular; "
        "Normalbefunde nicht auflisten; leere Liste, wenn keine Läsion beschrieben ist\n"
        "- Falls eine Seite nicht erwähnt oder nicht beurteilbar: null\n"
        "- Nur valides JSON, keine Erklärungen."
    )


def _parse_response(raw: str) -> tuple[dict, bool]:
    """
    Parst JSON-Antwort des Modells.
    Gibt (parsed_dict, parse_error: bool) zurück.
    """
    if not raw or raw.startswith("Error:"):
        return {}, True

    text = raw.strip()
    # Markdown-Codeblock entfernen
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
    """Extrahiert birads, acr, lesionen für eine Seite aus dem geparsten Dict."""
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

def run(config: dict, client, data: list, results_path: str, logger=None) -> str:
    sleep_s, max_errors = _parse_benchmark_settings(config)

    completed_ids: set = set()
    if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
        try:
            existing = pd.read_csv(results_path, usecols=["id"])
            completed_ids = set(existing["id"].dropna().astype(str).tolist())
            if completed_ids:
                print(f"Resume: {len(completed_ids)} Untersuchungen bereits vorhanden.")
        except Exception:
            pass

    total     = len(data)
    remaining = sum(1 for it in data if str(it.get("id")) not in completed_ids)
    print(f"  {total} Untersuchungen  ({remaining} remaining)...")

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

            prompt       = _build_prompt(item["text"])
            model_answer = client.ask_question(prompt, system_prompt=_SYSTEM_PROMPT_DE)

            is_error = isinstance(model_answer, str) and model_answer.startswith("Error:")
            if is_error:
                errors += 1

            parsed, parse_error = _parse_response(model_answer)

            model_meno          = str(parsed.get("menopause") or "").strip() or ""
            birads_li, acr_li, lesions_li = _extract_side(parsed, "links")
            birads_re, acr_re, lesions_re = _extract_side(parsed, "rechts")

            if logger:
                status = "ERROR" if is_error else f"parse_err={parse_error}"
                logger.verbose(
                    f"[{idx:>{len(str(total))}}/{total}] {item_id}  →  {status}"
                )

            if is_error and max_errors is not None and errors >= max_errors:
                print(f"Abbruch: max_errors={max_errors} erreicht.")
                break

            gt = item.get("gt", {})
            writer.writerow({
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

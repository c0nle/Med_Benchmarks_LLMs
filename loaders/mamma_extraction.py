"""
Mamma-MRT label-extraction loader.

Reads label_extraction_gt.xlsx and hiwi_gt_ergaenzung.xlsx, joined on
'ID gekürzt' (ground truth) = 'AnforderungsNr' (report file). All ids are read as
strings (avoids float / exponent formatting).

Role of the files:
    label_extraction_gt.xlsx   – ground truth, one row per lesion: menopause, BI-RADS
                                 (maximum per side), MR-ACR (= BPE) per side, lesion types.
                                 Exam-level fields (menopause, BPE, risk, history, ...) are
                                 either all filled or all empty for an exam; empty means
                                 "not annotated".
    hiwi_gt_ergaenzung.xlsx    – provides the report text ('Befund') per exam. Its *_GT
                                 columns (menopause / BI-RADS / ACR) repeat the values derived
                                 from label_extraction_gt.xlsx (identical for all 302 exams)
                                 and are only used as a consistency check: a differing value
                                 would win and be recorded in meta.conflicts.

Returns one item per exam that has a report text (302 for the full data set).

Item schema:
    id            : AnforderungsNr as str
    benchmark     : "LabelExtractionMamma"
    text          : report text (str) – never write it to logs
    gt            : dict
        menopause : "prä" | "post" | None
        birads_li : "2"–"5" | None  (maximum over all left-side lesions)
        birads_re : "2"–"5" | None  (maximum over all right-side lesions)
        acr_li    : "1"–"4" | None
        acr_re    : "1"–"4" | None
        lesions_li: list[str]  ('Simone befund', else 'Rad Befund', left)
        lesions_re: list[str]  ('Simone befund', else 'Rad Befund', right)
    meta          : dict
        conflicts : list[str]  (report-file *_GT ≠ ground truth; empty for the current data)
        n_lesions_li, n_lesions_re: int
        menopause_in_report: bool  (the report text mentions the menopausal status, see
                                    menopause_in_report(); the annotation often takes the
                                    status from other sources)
"""
import logging
import re
from pathlib import Path

import pandas as pd

_GT_PATH   = Path("data/label_extraction/label_extraction_gt.xlsx")
_HIWI_PATH = Path("data/label_extraction/hiwi_gt_ergaenzung.xlsx")

_log = logging.getLogger(__name__)

# Wording that states (or lets one read off) the menopausal status in a German report:
# "postmenopausal", "Prämenopause", cycle day / week, last period, amenorrhoea.
_MENOPAUSE_MENTION = re.compile(
    r"(prä|prae|pre|post|peri)[\s-]*menopaus|menopaus|zyklus(tag|woche|mitte)?|\bZT\s*\d"
    r"|letzte\s+regel|menstruation|amenorrh",
    re.IGNORECASE,
)


def menopause_in_report(text: str) -> bool:
    """True if the report text mentions the menopausal status (keyword match)."""
    return bool(_MENOPAUSE_MENTION.search(text or ""))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_int(val):
    """String → int, None if not a number."""
    if val is None:
        return None
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None


def _max_birads(series: pd.Series):
    """Maximum BI-RADS value (as int string) over a series of string values."""
    vals = [_safe_int(v) for v in series.dropna()
            if str(v).strip() not in ("", "?", "nan", "None")]
    vals = [v for v in vals if v is not None]
    return str(max(vals)) if vals else None


def _first_valid(series: pd.Series):
    """First non-empty value that is not '?', as string."""
    for v in series.dropna():
        s = str(v).strip()
        if s and s not in ("?", "nan", "None"):
            return s
    return None


def _lesion_type(row):
    """'Simone befund' if present, else 'Rad Befund'."""
    s = str(row.get("Simone befund") or "").strip()
    if s and s not in ("nan", "None", "?"):
        return s
    r = str(row.get("Rad Befund") or "").strip()
    return r if r and r not in ("nan", "None", "?") else None


def _normalize_side(val):
    """Normalise 'Links'/'links'/'rechts'/'Rechts' → lowercase."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("links", "rechts"):
        return s
    return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_mamma_extraction(limit=None, config=None):
    """
    Load the Mamma-MRT label-extraction data.

    Requires:
        data/label_extraction/label_extraction_gt.xlsx
        data/label_extraction/hiwi_gt_ergaenzung.xlsx

    Returns a list of dicts, one per exam with a report text.
    """
    print("--- Loading Mamma-MRT label extraction ---")

    if not _GT_PATH.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {_GT_PATH}\n"
            "Expected: data/label_extraction/label_extraction_gt.xlsx"
        )
    if not _HIWI_PATH.exists():
        raise FileNotFoundError(
            f"Report file not found: {_HIWI_PATH}\n"
            "Expected: data/label_extraction/hiwi_gt_ergaenzung.xlsx"
        )

    gt   = pd.read_excel(_GT_PATH,   dtype=str)
    hiwi = pd.read_excel(_HIWI_PATH, dtype=str)

    # Normalise the join keys
    gt["_id"]   = gt["ID gekürzt"].fillna("").str.strip()
    hiwi["_id"] = hiwi["AnforderungsNr"].fillna("").str.strip()

    # Index the report file by id (one row per exam expected)
    hiwi_idx = hiwi.set_index("_id")

    gt_grouped = gt.groupby("_id", sort=False)

    items: list = []
    n_no_text = 0

    for exam_id, group in gt_grouped:
        if not exam_id:
            continue
        if exam_id not in hiwi_idx.index:
            n_no_text += 1
            continue

        hiwi_row = hiwi_idx.loc[exam_id]
        if isinstance(hiwi_row, pd.DataFrame):
            hiwi_row = hiwi_row.iloc[0]

        befund = str(hiwi_row.get("Befund") or "").strip()
        if not befund or befund in ("nan", "None"):
            n_no_text += 1
            continue

        # Normalise the side
        group = group.copy()
        group["_side"] = group["Seite.1"].map(_normalize_side)

        li_rows = group[group["_side"] == "links"]
        re_rows = group[group["_side"] == "rechts"]

        # Menopause (exam level)
        menopause = _first_valid(group["Menopause"])

        # BI-RADS: maximum over the lesions of each side
        birads_li = _max_birads(li_rows["MR BIRADS"]) if not li_rows.empty else None
        birads_re = _max_birads(re_rows["MR BIRADS"]) if not re_rows.empty else None

        # ACR / BPE (exam level, same value in all rows of an exam)
        acr_li = _first_valid(group["MR-ACR links"])
        acr_re = _first_valid(group["MR-ACR rechts"])

        # Lesions per side
        lesions_li = [t for _, r in li_rows.iterrows() if (t := _lesion_type(r)) is not None]
        lesions_re = [t for _, r in re_rows.iterrows() if (t := _lesion_type(r)) is not None]

        # Consistency check against the *_GT columns of the report file. They repeat the
        # values derived above (currently identical); a differing value would win and be
        # recorded as a conflict.
        conflicts: list = []

        def _apply_gt(hiwi_col: str, current, field: str):
            raw = str(hiwi_row.get(hiwi_col) or "").strip()
            if not raw or raw in ("nan", "None", "?"):
                return current
            if current and current != raw:
                conflicts.append(f"{field}: orig={current!r} → hiwi={raw!r}")
                _log.debug("ID %s – %s", exam_id, conflicts[-1])
            return raw

        menopause = _apply_gt("Menopause_GT", menopause, "menopause")
        birads_li = _apply_gt("BIRADS_li_GT", birads_li, "birads_li")
        birads_re = _apply_gt("BIRADS_re_GT", birads_re, "birads_re")
        acr_li    = _apply_gt("ACR_li_GT",    acr_li,    "acr_li")
        acr_re    = _apply_gt("ACR_re_GT",    acr_re,    "acr_re")

        items.append({
            "id":        exam_id,
            "benchmark": "LabelExtractionMamma",
            "text":      befund,
            "gt": {
                "menopause": menopause,
                "birads_li": birads_li,
                "birads_re": birads_re,
                "acr_li":    acr_li,
                "acr_re":    acr_re,
                "lesions_li": lesions_li,
                "lesions_re": lesions_re,
            },
            "meta": {
                "conflicts":    conflicts,
                "n_lesions_li": len(lesions_li),
                "n_lesions_re": len(lesions_re),
                "menopause_in_report": menopause_in_report(befund),
            },
        })

    n_conflicts_total = sum(len(it["meta"]["conflicts"]) for it in items)
    print(
        f"  {len(items)} exams loaded  "
        f"({n_no_text} without report text skipped, "
        f"{n_conflicts_total} ground-truth conflicts logged)"
    )

    if limit:
        items = items[:limit]

    return items

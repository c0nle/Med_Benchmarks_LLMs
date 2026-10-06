"""
Mamma-MRT Label-Extraction Loader.

Lädt label_extraction_gt.xlsx und hiwi_gt_ergaenzung.xlsx,
joined per 'ID gekürzt' (GT) = 'AnforderungsNr' (Hiwi).
Alle IDs werden als String geladen (vermeidet Float/Exponentialdarstellung).

Gibt eine Liste von Items zurück: ein Item pro Untersuchung mit Befundtext (302 bei vollem Datensatz).

Item-Schema:
    id            : AnforderungsNr als str
    benchmark     : "LabelExtractionMamma"
    text          : Befundtext (str)  – NICHT in Logs ausgeben
    gt            : dict
        menopause : "prä" | "post" | None
        birads_li : "2"–"5" | None  (Maximum über alle Läsionen links)
        birads_re : "2"–"5" | None  (Maximum über alle Läsionen rechts)
        acr_li    : "1"–"4" | None
        acr_re    : "1"–"4" | None
        lesions_li: list[str]  (Simone befund > Rad Befund, links)
        lesions_re: list[str]  (Simone befund > Rad Befund, rechts)
    meta          : dict
        conflicts : list[str]  (Hiwi-GT vs. Original-GT Konflikte)
        n_lesions_li, n_lesions_re: int
"""
import logging
from pathlib import Path

import pandas as pd

_GT_PATH   = Path("data/label_extraction/label_extraction_gt.xlsx")
_HIWI_PATH = Path("data/label_extraction/hiwi_gt_ergaenzung.xlsx")

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _safe_int(val):
    """String → int, None bei Fehler."""
    if val is None:
        return None
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None


def _max_birads(series: pd.Series):
    """Maximaler BIRADS-Wert (als int-String) über eine Series von String-Werten."""
    vals = [_safe_int(v) for v in series.dropna()
            if str(v).strip() not in ("", "?", "nan", "None")]
    vals = [v for v in vals if v is not None]
    return str(max(vals)) if vals else None


def _first_valid(series: pd.Series):
    """Erstes nicht-leeres, nicht-'?'-Element als String."""
    for v in series.dropna():
        s = str(v).strip()
        if s and s not in ("?", "nan", "None"):
            return s
    return None


def _lesion_type(row):
    """Simone befund wenn vorhanden, sonst Rad Befund."""
    s = str(row.get("Simone befund") or "").strip()
    if s and s not in ("nan", "None", "?"):
        return s
    r = str(row.get("Rad Befund") or "").strip()
    return r if r and r not in ("nan", "None", "?") else None


def _normalize_side(val):
    """Normalisiert 'Links'/'links'/'rechts'/'Rechts' → lowercase."""
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
    Lädt Mamma-MRT Label-Extraction-Daten.

    Benötigt:
        data/label_extraction/label_extraction_gt.xlsx
        data/label_extraction/hiwi_gt_ergaenzung.xlsx

    Gibt Liste von Dicts zurück (ein Dict pro Untersuchung mit Befundtext).
    """
    print("--- Lade Mamma-MRT Label Extraction ---")

    if not _GT_PATH.exists():
        raise FileNotFoundError(
            f"GT-Datei nicht gefunden: {_GT_PATH}\n"
            "Erwartet: data/label_extraction/label_extraction_gt.xlsx"
        )
    if not _HIWI_PATH.exists():
        raise FileNotFoundError(
            f"Hiwi-Datei nicht gefunden: {_HIWI_PATH}\n"
            "Erwartet: data/label_extraction/hiwi_gt_ergaenzung.xlsx"
        )

    gt   = pd.read_excel(_GT_PATH,   dtype=str)
    hiwi = pd.read_excel(_HIWI_PATH, dtype=str)

    # Join-Keys normalisieren
    gt["_id"]   = gt["ID gekürzt"].fillna("").str.strip()
    hiwi["_id"] = hiwi["AnforderungsNr"].fillna("").str.strip()

    # Hiwi nach ID indizieren (eine Zeile pro Untersuchung erwartet)
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

        # Seite normalisieren
        group = group.copy()
        group["_side"] = group["Seite.1"].map(_normalize_side)

        li_rows = group[group["_side"] == "links"]
        re_rows = group[group["_side"] == "rechts"]

        # Menopause (Untersuchungsebene)
        menopause = _first_valid(group["Menopause"])

        # BIRADS: Maximum über Läsionen pro Seite
        birads_li = _max_birads(li_rows["MR BIRADS"]) if not li_rows.empty else None
        birads_re = _max_birads(re_rows["MR BIRADS"]) if not re_rows.empty else None

        # ACR (Untersuchungsebene, gleiche Spalte in allen Zeilen)
        acr_li = _first_valid(group["MR-ACR links"])
        acr_re = _first_valid(group["MR-ACR rechts"])

        # Läsionen pro Seite
        lesions_li = [t for _, r in li_rows.iterrows() if (t := _lesion_type(r)) is not None]
        lesions_re = [t for _, r in re_rows.iterrows() if (t := _lesion_type(r)) is not None]

        # Hiwi-GT-Overrides (Priorität Hiwi > Original, Konflikte loggen)
        conflicts: list[str] = []

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
            },
        })

    n_conflicts_total = sum(len(it["meta"]["conflicts"]) for it in items)
    print(
        f"  {len(items)} Untersuchungen geladen  "
        f"({n_no_text} ohne Befundtext übersprungen, "
        f"{n_conflicts_total} GT-Konflikte geloggt)"
    )

    if limit:
        items = items[:limit]

    return items

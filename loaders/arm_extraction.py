"""
Arm-Röntgen Label-Extraction Loader.
(Kreutzer et al., Eur Radiol 2025, https://doi.org/10.1007/s00330-025-12102-1)

Ordnerstruktur:
    data/label_extraction/Label_Extraction_Kilian/Label_Extraction_Kilian/
        clavicle/
            clavicle_ids_test_lax.csv   – binäre Labels (0/1), Spalte Phase
            Reports_4o_1609/<id>.txt    – Befundtexte
            New_template_clavicle.json  – Template (ungültiges JSON, wird repariert)
        elbow/
            elbow_ids_test_lax.csv
            Reports_4o_1309/<id>.txt
            New_Template_elbow.json     – valides JSON
        thumb/
            thumb_ids_test_lax.csv
            Reports_4o_1309/<id>.txt
            New_template_thumb.json     – ungültiges JSON, wird repariert

ID-Extraktion aus Bildpfad:
    clavicle : .../ConvertedPNGs/<id>.png         → Dateiname ohne Extension
    elbow    : .../<id>/ap.png                    → vorletztes Segment
    thumb    : .../<id>/ap.png                    → vorletztes Segment

Item-Schema:
    id             : "<region>-<report_id>" (str)
    benchmark      : "LabelExtractionArm"
    text           : Befundtext (str)  – _x000D_ durch Zeilenumbruch ersetzt
    region         : "clavicle" | "elbow" | "thumb"
    phase          : "train" | "val" | "test"
    gt_labels      : dict {label_name: int(0|1)}  – nur CSV-Labels
    template_labels: list[str]  – Reihenfolge aus Template (soweit in CSV vorhanden)
    meta           : dict {report_id: str, region: str}
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

_ARM_BASE = Path(
    "data/label_extraction/Label_Extraction_Kilian/Label_Extraction_Kilian"
)

_REGION_CFG = {
    "clavicle": {
        "csv":         "clavicle_ids_test_lax.csv",
        "reports_dir": "Reports_4o_1609",
        "template":    "New_template_clavicle.json",
        "id_col":      "Image_Path",
        "id_mode":     "filename",
    },
    "elbow": {
        "csv":         "elbow_ids_test_lax.csv",
        "reports_dir": "Reports_4o_1309",
        "template":    "New_Template_elbow.json",
        "id_col":      "AP_Image_Path",
        "id_mode":     "parent_dir",
    },
    "thumb": {
        "csv":         "thumb_ids_test_lax.csv",
        "reports_dir": "Reports_4o_1309",
        "template":    "New_template_thumb.json",
        "id_col":      "AP_Image_Path",
        "id_mode":     "parent_dir",
    },
}

_NON_LABEL_COLS = frozenset(
    {"Image_Path", "AP_Image_Path", "LAT_Image_Path", "Phase", "_id"}
)


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------

def _extract_id(path_str: str, mode: str) -> str:
    """Extrahiert numerische ID aus Bildpfad."""
    p = str(path_str).strip()
    if mode == "filename":
        return os.path.splitext(os.path.basename(p))[0]
    # parent_dir: .../<id>/ap.png
    parts = p.rstrip("/").split("/")
    if p.endswith("ap.png") and len(parts) >= 2:
        return parts[-2]
    return os.path.splitext(os.path.basename(p))[0]


def _repair_json(text: str) -> str:
    """
    Repariert bekanntes JSON-Problem: fehlendes Komma vor "Ossicles"-Eintrag.
    Ändert die Originaldatei NICHT.
    """
    return re.sub(r"(\})\s*\n(\s*\"Ossicles\")", r"\1,\n\2", text)


def _load_template(json_path: Path) -> dict:
    """Lädt Template-JSON robust; repariert bekannte Syntaxfehler."""
    with open(json_path, encoding="utf-8") as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return json.loads(_repair_json(content))


def _load_report(reports_dir: Path, report_id: str) -> str:
    """Lädt Befundtext; ersetzt _x000D_ durch Zeilenumbruch."""
    path = reports_dir / f"{report_id}.txt"
    if not path.exists():
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text.replace("_x000D_", "\n").strip()


def _load_region(region: str, phase_filter: str = "test") -> list:
    """Lädt alle Items für eine Region."""
    cfg = _REGION_CFG[region]
    region_dir = _ARM_BASE / region

    csv_path = region_dir / cfg["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    df["_id"] = df[cfg["id_col"]].apply(lambda x: _extract_id(x, cfg["id_mode"]))

    if phase_filter and "Phase" in df.columns:
        df = df[df["Phase"].str.lower() == phase_filter.lower()].copy()

    label_cols = [c for c in df.columns if c not in _NON_LABEL_COLS]

    template_path = region_dir / cfg["template"]
    if template_path.exists():
        template_order = list(_load_template(template_path).keys())
        # Nur CSV-Labels verwenden (CSV ist autoritativ), aber in Template-Reihenfolge
        template_labels = [l for l in template_order if l in label_cols]
        if not template_labels:
            template_labels = label_cols
    else:
        template_labels = label_cols

    reports_dir = region_dir / cfg["reports_dir"]
    items = []

    for _, row in df.iterrows():
        report_id = str(row["_id"]).strip()
        text = _load_report(reports_dir, report_id)
        if not text:
            continue  # Reports ohne Text werden übersprungen

        gt_labels: dict[str, int] = {}
        for col in template_labels:
            try:
                gt_labels[col] = int(row.get(col, 0))
            except (ValueError, TypeError):
                gt_labels[col] = 0

        items.append({
            "id":             f"{region}-{report_id}",
            "benchmark":      "LabelExtractionArm",
            "text":           text,
            "region":         region,
            "phase":          str(row.get("Phase") or phase_filter or "").lower(),
            "gt_labels":      gt_labels,
            "template_labels": template_labels,
            "meta":           {"report_id": report_id, "region": region},
        })

    return items


# ---------------------------------------------------------------------------
# Öffentlicher Loader
# ---------------------------------------------------------------------------

def load_arm_extraction(limit=None, config=None):
    """
    Lädt Arm-Röntgen Label-Extraction-Daten.

    Config-Optionen (unter task_settings.label_extraction_arm):
        regions : "clavicle" | "elbow" | "thumb" | "all"  (Standard: "all")
        phase   : "train" | "val" | "test"                 (Standard: "test")

    Benötigt: data/label_extraction/Label_Extraction_Kilian/Label_Extraction_Kilian/
    """
    print("--- Lade Arm-Röntgen Label Extraction ---")

    if not _ARM_BASE.exists():
        raise FileNotFoundError(
            f"Arm-Datensatz nicht gefunden: {_ARM_BASE}\n"
            "Erwartet: data/label_extraction/Label_Extraction_Kilian/"
            "Label_Extraction_Kilian/"
        )

    task_cfg: dict = {}
    if config:
        task_cfg = config.get("task_settings", {}).get("label_extraction_arm", {})

    regions_cfg = task_cfg.get("regions", "all")
    phase       = str(task_cfg.get("phase", "test")).strip().lower()

    if regions_cfg == "all":
        regions = ["clavicle", "elbow", "thumb"]
    elif isinstance(regions_cfg, list):
        regions = [str(r).strip().lower() for r in regions_cfg]
    else:
        regions = [str(regions_cfg).strip().lower()]

    per_region: list = []
    for region in regions:
        region_items = _load_region(region, phase_filter=phase)
        print(f"  {region}: {len(region_items)} Reports geladen (phase={phase})")
        per_region.append(region_items)

    # Interleave regions so that `limit` keeps all regions represented
    from itertools import chain, zip_longest
    all_items: list = [it for it in chain.from_iterable(zip_longest(*per_region)) if it is not None]

    print(f"  Gesamt: {len(all_items)} Reports")

    if limit:
        all_items = all_items[:limit]

    return all_items

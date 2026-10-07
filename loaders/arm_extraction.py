"""
Arm X-ray label-extraction loader
(data from Kreutzer et al., Eur Radiol 2025, https://doi.org/10.1007/s00330-025-12102-1).

Folder layout:
    data/label_extraction/Label_Extraction_Kilian/Label_Extraction_Kilian/
        clavicle/
            clavicle_ids_test_lax.csv   – binary labels (0/1), column Phase
            Reports_4o_1609/<id>.txt    – report texts
            New_template_clavicle.json  – label template (invalid JSON, repaired on load)
        elbow/
            elbow_ids_test_lax.csv
            Reports_4o_1309/<id>.txt
            New_Template_elbow.json     – valid JSON
        thumb/
            thumb_ids_test_lax.csv
            Reports_4o_1309/<id>.txt
            New_template_thumb.json     – invalid JSON, repaired on load

Label definitions: the templates only contain label names with empty
{"finding": false, "citation": ""} entries; the data folder has no annotation guidelines
or label definitions, so the prompt lists the label names only. The templates have more
labels than the CSVs (clavicle 26/18, elbow 29/28, thumb 25/23); only CSV labels are scored.

The CSVs reference X-ray image paths; they are only used to derive the report id
(images are not loaded – this is a text-extraction task):
    clavicle : .../ConvertedPNGs/<id>.png         → file name without extension
    elbow    : .../<id>/ap.png                    → parent folder name
    thumb    : .../<id>/ap.png                    → parent folder name

Item schema:
    id             : "<region>-<report_id>" (str)
    benchmark      : "LabelExtractionArm"
    text           : report text (str); "_x000D_" replaced by a line break
    region         : "clavicle" | "elbow" | "thumb"
    phase          : "train" | "val" | "test"
    gt_labels      : dict {label_name: int(0|1)}  – CSV labels only
    template_labels: list[str]  – order as in the template (labels present in the CSV)
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
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_id(path_str: str, mode: str) -> str:
    """Derive the report id from an image path."""
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
    Repair the known JSON problem in the templates: a missing comma before the
    "Ossicles" entry. The original file is not changed.
    """
    return re.sub(r"(\})\s*\n(\s*\"Ossicles\")", r"\1,\n\2", text)


def _load_template(json_path: Path) -> dict:
    """Load a template JSON; repair the known syntax error if needed."""
    with open(json_path, encoding="utf-8") as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return json.loads(_repair_json(content))


def _load_report(reports_dir: Path, report_id: str) -> str:
    """Load a report text; replace "_x000D_" by a line break."""
    path = reports_dir / f"{report_id}.txt"
    if not path.exists():
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text.replace("_x000D_", "\n").strip()


def _load_region(region: str, phase_filter: str = "test") -> list:
    """Load all items of one region."""
    cfg = _REGION_CFG[region]
    region_dir = _ARM_BASE / region

    csv_path = region_dir / cfg["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    df["_id"] = df[cfg["id_col"]].apply(lambda x: _extract_id(x, cfg["id_mode"]))

    if phase_filter and "Phase" in df.columns:
        df = df[df["Phase"].str.lower() == phase_filter.lower()].copy()

    label_cols = [c for c in df.columns if c not in _NON_LABEL_COLS]

    template_path = region_dir / cfg["template"]
    if template_path.exists():
        template_order = list(_load_template(template_path).keys())
        # Only CSV labels are scored (the CSV is authoritative), in template order
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
            continue  # reports without text are skipped

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
# Public loader
# ---------------------------------------------------------------------------

def load_arm_extraction(limit=None, config=None):
    """
    Load the Arm X-ray label-extraction data.

    Config options (task_settings.label_extraction_arm):
        regions : "clavicle" | "elbow" | "thumb" | "all"  (default: "all")
        phase   : "train" | "val" | "test"                 (default: "test")

    Requires: data/label_extraction/Label_Extraction_Kilian/Label_Extraction_Kilian/
    """
    print("--- Loading Arm X-ray label extraction ---")

    if not _ARM_BASE.exists():
        raise FileNotFoundError(
            f"Arm X-ray data not found: {_ARM_BASE}\n"
            "Expected: data/label_extraction/Label_Extraction_Kilian/"
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
        print(f"  {region}: {len(region_items)} reports loaded (phase={phase})")
        per_region.append(region_items)

    # Interleave regions so that `limit` keeps all regions represented
    from itertools import chain, zip_longest
    all_items: list = [it for it in chain.from_iterable(zip_longest(*per_region)) if it is not None]

    print(f"  Total: {len(all_items)} reports")

    if limit:
        all_items = all_items[:limit]

    return all_items

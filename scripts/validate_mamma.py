"""
Validierungsskript: liest MammaMR_extraction_gpt-oss-120b.xlsx als Modell-Output
und schickt ihn durch die neue Mamma-Auswertungs-Pipeline.

Referenzwerte der alten Auswertung:
  Menopause:    Accuracy 0.65
  BIRADS li:    Accuracy 0.444
  BIRADS re:    Accuracy 0.515
  ACR li:       Accuracy 0.029
  ACR re:       Accuracy 0.029
  Laesionen F1 li: 0.590
  Laesionen F1 re: 0.605

Erwartete Abweichungen:
  ACR: Die alte Auswertung verglich Brustdichte-Text vs. BPE-Nummern
       → Accuracy ~0.03 ist korrekt fuer die alte (falsche) Auswertung.
       Mit korrektem BPE-Matching ist eine deutlich hoehere Accuracy zu erwarten,
       sofern das Modell BPE-Werte numerisch ausgegeben hat.
  BIRADS: Die alte Auswertung kannte BIRADS 6; das neue Mapping mappt 6→5 (konfigurierbar).

Aufruf:
  python scripts/validate_mamma.py [--out results/validate_mamma.csv]
"""
import argparse
import json
import os
import re
import sys
import tempfile

import pandas as pd

# Projektpfad hinzufuegen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.mamma_extraction import load_mamma_extraction


_EXTRACTION_PATH = "data/label_extraction/MammaMR_extraction_gpt-oss-120b.xlsx"

_REFERENCE = {
    "menopause_accuracy":  0.650,
    "birads_li_accuracy":  0.444,
    "birads_re_accuracy":  0.515,
    "acr_li_accuracy":     0.029,
    "acr_re_accuracy":     0.029,
    "lesions_li_micro_f1": 0.590,
    "lesions_re_micro_f1": 0.605,
}


def _get_lesions_for_side(group: pd.DataFrame, side: str) -> list:
    """Sammelt Laesionstypen aus Lesion_*_Typ_Art-Spalten fuer eine Seite."""
    lesions = []
    for _, row in group[group["Seite"].str.upper() == side.upper()].iterrows():
        for i in range(1, 11):
            val = str(row.get(f"Lesion_{i}_Typ_Art") or "").strip()
            if val and val not in ("nan", "None", "ERROR"):
                lesions.append(val)
    return lesions


def build_results_df(extraction_df: pd.DataFrame, gt_items: list) -> pd.DataFrame:
    """
    Konvertiert MammaMR_extraction_gpt-oss-120b.xlsx in das Standard-CSV-Format
    der Mamma-Task-Ausgabe (eine Zeile pro Untersuchung).
    """
    gt_by_id = {item["id"]: item for item in gt_items}
    rows = []

    for anfnr, group in extraction_df.groupby("Anforderungsnummer"):
        exam_id = str(anfnr).strip()
        if exam_id not in gt_by_id:
            continue

        gt = gt_by_id[exam_id]["gt"]

        li_rows = group[group["Seite"].str.upper() == "LINKS"]
        re_rows = group[group["Seite"].str.upper() == "RECHTS"]

        # Menopausenstatus (Untersuchungsebene, gleich in allen Zeilen)
        meno_vals = group["Menopausenstatus"].dropna()
        meno_val  = meno_vals.iloc[0] if not meno_vals.empty else ""
        meno_val  = "" if str(meno_val) in ("nan", "None", "ERROR") else str(meno_val).strip()

        # BIRADS pro Seite
        def _first_birads(side_rows: pd.DataFrame) -> str:
            # Highest BIRADS of the side, matching the GT definition (max over lesions)
            best, best_val = "", -1
            for v in side_rows["BIRADS_Einschaetzung"].dropna():
                s = str(v).strip()
                m = re.match(r"\s*([1-6])", s)
                if m and int(m.group(1)) > best_val:
                    best, best_val = s, int(m.group(1))
            return best

        birads_li = _first_birads(li_rows)
        birads_re = _first_birads(re_rows)

        # ACR/BPE: Hintergrundanreicherung ist Untersuchungsebene
        acr_vals = group["Hintergrundanreicherung"].dropna()
        acr_val  = ""
        for v in acr_vals:
            s = str(v).strip()
            if s and s not in ("nan", "None", "ERROR"):
                acr_val = s
                break

        # Laesionen pro Seite
        lesions_li = _get_lesions_for_side(group, "LINKS")
        lesions_re = _get_lesions_for_side(group, "RECHTS")

        rows.append({
            "id":              exam_id,
            "benchmark":       "LabelExtractionMamma",
            "gt_menopause":    gt.get("menopause") or "",
            "gt_birads_li":    gt.get("birads_li") or "",
            "gt_birads_re":    gt.get("birads_re") or "",
            "gt_acr_li":       gt.get("acr_li") or "",
            "gt_acr_re":       gt.get("acr_re") or "",
            "gt_lesions_li":   json.dumps(gt.get("lesions_li") or [], ensure_ascii=False),
            "gt_lesions_re":   json.dumps(gt.get("lesions_re") or [], ensure_ascii=False),
            "model_raw":       "",
            "model_menopause": meno_val,
            "model_birads_li": birads_li,
            "model_birads_re": birads_re,
            "model_acr_li":    acr_val,   # gleicher Wert fuer beide Seiten
            "model_acr_re":    acr_val,
            "model_lesions_li": json.dumps(lesions_li, ensure_ascii=False),
            "model_lesions_re": json.dumps(lesions_re, ensure_ascii=False),
            "parse_error":     "False",
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Validierung der Mamma-Auswertungs-Pipeline")
    parser.add_argument(
        "--out",
        default="results/validate_mamma_results.csv",
        help="Pfad fuer temporaere Results-CSV",
    )
    parser.add_argument(
        "--report",
        default="results/validate_mamma_report.jsonl",
        help="Pfad fuer JSONL-Report",
    )
    args = parser.parse_args()

    if not os.path.exists(_EXTRACTION_PATH):
        print(f"FEHLER: Extraktionsdatei nicht gefunden: {_EXTRACTION_PATH}")
        sys.exit(1)

    print("Lade GT-Daten...")
    gt_items = load_mamma_extraction()

    print(f"Lade Extraktionsdatei: {_EXTRACTION_PATH}")
    ext_df = pd.read_excel(_EXTRACTION_PATH, dtype=str)

    print(f"  Zeilen: {len(ext_df)}, unique IDs: {ext_df['Anforderungsnummer'].nunique()}")

    results_df = build_results_df(ext_df, gt_items)
    print(f"  Gematchte Untersuchungen: {len(results_df)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results_df.to_csv(args.out, index=False)
    print(f"  Results CSV geschrieben: {args.out}")

    from evaluate import write_mamma_extraction_report_jsonl

    config = {
        "task_settings": {
            "label_extraction_mamma": {
                "acr_range":           "min",
                "birads6_handling":    "map_to_5",
                "gt_empty_ext_present": "ignore",
            }
        }
    }

    report = write_mamma_extraction_report_jsonl(
        args.out, out_path=args.report, config=config
    )
    print(f"  Report JSONL geschrieben: {args.report}")

    print("\n" + "=" * 60)
    print("  ERGEBNISSE (neue Auswertung vs. Referenz alte Auswertung)")
    print("=" * 60)

    fields_map = {
        "menopause_accuracy_pct":  "menopause_accuracy",
        "birads_li_accuracy_pct":  "birads_li_accuracy",
        "birads_re_accuracy_pct":  "birads_re_accuracy",
        "acr_li_accuracy_pct":     "acr_li_accuracy",
        "acr_re_accuracy_pct":     "acr_re_accuracy",
        "lesions_li_micro_f1_pct": "lesions_li_micro_f1",
        "lesions_re_micro_f1_pct": "lesions_re_micro_f1",
    }

    for report_key, ref_key in fields_map.items():
        neue_val = report.get(report_key)
        ref_val  = _REFERENCE.get(ref_key)
        if neue_val is None:
            continue
        neue_frac = neue_val / 100
        diff      = neue_frac - ref_val
        symbol    = "+" if diff >= 0 else ""
        print(
            f"  {ref_key:<28}  "
            f"neu={neue_frac:.3f}  ref={ref_val:.3f}  diff={symbol}{diff:.3f}"
        )

    # The old evaluation kept BIRADS 6 as its own class: the birads6_keep sensitivity
    # analysis (always computed) is the like-for-like comparison.
    for side in ("birads_li", "birads_re"):
        keep_val = report.get(f"{side}_accuracy_birads6keep_pct")
        if keep_val is not None:
            ref_val = _REFERENCE[f"{side}_accuracy"]
            print(f"  {side + '_accuracy (6 kept)':<28}  neu={keep_val / 100:.3f}  "
                  f"ref={ref_val:.3f}  diff={keep_val / 100 - ref_val:+.3f}")

    print()
    print("Hinweise zu erwarteten Abweichungen:")
    print("  ACR: Die alte Auswertung verglich Brustdichte-Text (z.B. 'Fast ausschliesslich Fett')")
    print("       gegen numerische BPE-GT-Werte (1-4) → Accuracy ~0.03 war systematisch falsch.")
    print("       'Hintergrundanreicherung' in der Extraktionsdatei ist ebenfalls BPE-Text,")
    print("       aber das YAML-Mapping unterstuetzt diese Textvarianten.")
    print("  BIRADS 6: Die neue Pipeline mappt BIRADS 6 → 5 (konfigurierbar).")
    print("            In der alten Auswertung war BIRADS 6 eine eigene Klasse.")
    print("            Damit kann die neue Accuracy fuer BIRADS hoeher liegen;")
    print("            '(6 kept)' = Sensitivitaetsanalyse birads6_handling=keep (wie alt).")
    print("  Laesionen: Die Normalisierung via mamma_normalization.yaml kann Synonyme")
    print("             zusammenfuehren (z.B. 'Fibroadenom ' → 'fibroadenom').")


if __name__ == "__main__":
    main()

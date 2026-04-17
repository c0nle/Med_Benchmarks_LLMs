"""
Label Extraction Task Runner.

The model receives a radiology report text (Befundtext) and must extract
all medical entities (findings, diagnoses, anatomical structures, pathologies)
as a comma-separated list.

Results CSV columns: id, benchmark, text, reference_entities, model_entities
Evaluation: Micro F1 (TP/FP/FN aggregated globally, RadGraph protocol), handled by evaluate.py.
"""
import os
import csv
import time

import pandas as pd

from tasks.mcq import _parse_benchmark_settings


def run(config: dict, client, data: list, results_path: str, logger=None) -> str:
    """
    Run the label extraction benchmark and write incremental results.

    Returns the path to the written CSV.
    """
    sleep_s, max_errors = _parse_benchmark_settings(config)

    fieldnames = ["id", "benchmark", "text", "reference_entities", "model_entities"]

    completed_ids: set = set()
    if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
        try:
            existing = pd.read_csv(results_path, usecols=["id"])
            completed_ids = set(existing["id"].dropna().astype(str).tolist())
            if completed_ids:
                print(f"Resume: {len(completed_ids)} Texte bereits vorhanden, überspringe.")
        except Exception:
            completed_ids = set()

    total = len(data)
    remaining = sum(1 for item in data if str(item.get("id")) not in completed_ids)
    print(f"  {total} texts  ({remaining} remaining)...")

    start = time.time()
    processed_new = 0
    errors = 0

    file_exists = os.path.exists(results_path) and os.path.getsize(results_path) > 0
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for idx, item in enumerate(data, start=1):
            item_id = str(item.get("id"))
            if item_id in completed_ids:
                continue

            prompt = (
                f"Radiologischer Befundtext:\n{item['text']}\n\n"
                "Extrahiere alle medizinischen Entitäten aus dem Text "
                "(Befunde, Diagnosen, Anatomie, Pathologien, Modalitäten). "
                "Gib ausschließlich eine kommaseparierte Liste der Entitäten zurück. "
                "Keine Erklärungen, keine Nummerierung."
            )

            model_answer = client.ask_question(prompt)

            is_error = isinstance(model_answer, str) and model_answer.startswith("Error:")
            if is_error:
                errors += 1

            if logger:
                status = f"ERROR: {model_answer}" if is_error else model_answer.strip()[:60]
                logger.verbose(f"[{idx:>{len(str(total))}}/{total}] {item_id}  →  {status}")

            if is_error and max_errors is not None and errors >= max_errors:
                print(f"Abbruch: max_errors={max_errors} erreicht.")
                writer.writerow({
                    "id": item_id,
                    "benchmark": item.get("benchmark", ""),
                    "text": item.get("text", ""),
                    "reference_entities": item.get("entities", ""),
                    "model_entities": model_answer,
                })
                f.flush()
                break

            writer.writerow({
                "id": item_id,
                "benchmark": item.get("benchmark", ""),
                "text": item.get("text", ""),
                "reference_entities": item.get("entities", ""),
                "model_entities": model_answer,
            })
            f.flush()
            processed_new += 1

            if sleep_s > 0:
                time.sleep(sleep_s)

            if processed_new % 50 == 0:
                elapsed = time.time() - start
                rate = processed_new / elapsed if elapsed > 0 else 0.0
                eta_s = int((remaining - processed_new) / rate) if rate > 0 else -1
                eta = f"{eta_s//60:02d}:{eta_s%60:02d}" if eta_s >= 0 else "?"
                pct = int(processed_new / remaining * 100) if remaining > 0 else 100
                print(f"  [{processed_new:>{len(str(remaining))}}/{remaining}] {pct:3d}%  {rate:.1f} q/s  ETA {eta}  errors: {errors}")

    elapsed_total = time.time() - start
    print(f"  Done: {processed_new}/{remaining}  errors: {errors}  ({elapsed_total/60:.1f} min)")
    return results_path

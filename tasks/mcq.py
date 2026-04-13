"""
MCQ Task Runner – used for MedQA, RaR, and RadBench.

All three benchmarks share the same multiple-choice schema:
  item = { id, benchmark, question, options: [{key, value}], correct_answer, meta }

Results CSV columns: id, benchmark, question, correct_answer, model_answer
Evaluation: letter extraction → accuracy (handled by evaluate.py)
"""
import os
import csv
import time

import pandas as pd


def run(config: dict, client, data: list, results_path: str) -> str:
    """
    Run an MCQ benchmark and write incremental results to *results_path*.

    Returns the path to the written CSV.
    """
    sleep_s = float(config.get("benchmark_settings", {}).get("sleep_s", 0) or 0)
    max_errors = config.get("benchmark_settings", {}).get("max_errors", None)
    max_errors = int(max_errors) if max_errors is not None else None

    fieldnames = ["id", "benchmark", "question", "correct_answer", "model_answer"]

    # Resume: skip already processed IDs
    completed_ids: set = set()
    if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
        try:
            existing = pd.read_csv(results_path, usecols=["id"])
            completed_ids = set(existing["id"].dropna().astype(str).tolist())
            if completed_ids:
                print(f"Resume: {len(completed_ids)} Fragen bereits vorhanden, überspringe.")
        except Exception:
            completed_ids = set()

    total = len(data)
    remaining = sum(1 for item in data if str(item.get("id")) not in completed_ids)
    print(f"MCQ Benchmark: {total} Fragen ({remaining} neu zu bearbeiten)...")

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

            options_str = ", ".join(
                f"{opt['key']}: {opt['value']}" for opt in item.get("options", [])
            )
            prompt = (
                f"Frage: {item['question']}\n"
                f"Optionen: {options_str}\n"
                "Antworte nur mit dem korrekten Buchstaben (A/B/C/D)."
            )

            print(f"[{idx}/{total}] ID: {item_id}...")
            model_answer = client.ask_question(prompt)

            if isinstance(model_answer, str) and model_answer.startswith("Error:"):
                errors += 1
                if max_errors is not None and errors >= max_errors:
                    print(f"Abbruch: max_errors={max_errors} erreicht.")
                    writer.writerow({
                        "id": item_id,
                        "benchmark": item.get("benchmark", ""),
                        "question": item["question"],
                        "correct_answer": item.get("correct_answer", ""),
                        "model_answer": model_answer,
                    })
                    f.flush()
                    break

            writer.writerow({
                "id": item_id,
                "benchmark": item.get("benchmark", ""),
                "question": item["question"],
                "correct_answer": item.get("correct_answer", ""),
                "model_answer": model_answer,
            })
            f.flush()
            processed_new += 1

            if sleep_s > 0:
                time.sleep(sleep_s)

            if processed_new % 50 == 0:
                elapsed = time.time() - start
                rate = processed_new / elapsed if elapsed > 0 else 0.0
                eta_s = int((remaining - processed_new) / rate) if rate > 0 else -1
                eta = f"{eta_s//3600:02d}:{(eta_s%3600)//60:02d}:{eta_s%60:02d}" if eta_s >= 0 else "?"
                print(f"Progress: {processed_new}/{remaining}, Errors: {errors}, Rate: {rate:.2f} q/s, ETA: {eta}")

    print(f"MCQ Benchmark abgeschlossen. Ergebnisse in {results_path}")
    return results_path

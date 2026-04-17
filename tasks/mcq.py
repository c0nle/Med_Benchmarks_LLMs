"""
MCQ Task Runner – used for MedQA and RaR.

Both benchmarks share the same multiple-choice schema:
  item = { id, benchmark, question, options: [{key, value}], correct_answer, meta }

Results CSV columns: id, benchmark, question, correct_answer, model_answer
Evaluation: letter extraction → accuracy (handled by evaluate.py)

Note: RadBench is a VLM benchmark (X-ray images) and goes through tasks/vqa.py.
"""
import os
import csv
import time

import pandas as pd


def _parse_benchmark_settings(config: dict):
    s = config.get("benchmark_settings", {})
    sleep_s = float(s.get("sleep_s", 0) or 0)
    max_errors = s.get("max_errors", None)
    return sleep_s, int(max_errors) if max_errors is not None else None


def run(config: dict, client, data: list, results_path: str, logger=None) -> str:
    """
    Run an MCQ benchmark and write incremental results to *results_path*.

    Returns the path to the written CSV.
    """
    sleep_s, max_errors = _parse_benchmark_settings(config)

    fieldnames = ["id", "benchmark", "question", "correct_answer", "model_answer"]

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
    print(f"  {total} questions  ({remaining} remaining)...")

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

            opts = item.get("options", [])
            options_str = ", ".join(f"{opt['key']}: {opt['value']}" for opt in opts)
            keys = "/".join(opt["key"] for opt in opts) if opts else "A/B/C/D"
            prompt = (
                f"Question: {item['question']}\n"
                f"Options: {options_str}\n"
                f"Reply with only the correct letter ({keys})."
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
                eta = f"{eta_s//60:02d}:{eta_s%60:02d}" if eta_s >= 0 else "?"
                pct = int(processed_new / remaining * 100) if remaining > 0 else 100
                print(f"  [{processed_new:>{len(str(remaining))}}/{remaining}] {pct:3d}%  {rate:.1f} q/s  ETA {eta}  errors: {errors}")

    elapsed_total = time.time() - start
    print(f"  Done: {processed_new}/{remaining}  errors: {errors}  ({elapsed_total/60:.1f} min)")
    return results_path

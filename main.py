import yaml
import pandas as pd
import os
import csv
import time
from loaders.text_benchmarks import load_medqa
from core.client import MedicalLLMClient

def main():
    # Config
    config_path = "config.yaml" if os.path.exists("config.yaml") else "config.default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config_path != "config.yaml":
        print("Hinweis: config.yaml nicht gefunden, nutze config.default.yaml. "
              "Lege eine eigene config.yaml an (wird via .gitignore ignoriert).")

    # Data
    limit = config.get("benchmark_settings", {}).get("limit_samples", None)
    data = load_medqa(limit=limit)

    sleep_s = float(config.get("benchmark_settings", {}).get("sleep_s", 0) or 0)
    max_errors = config.get("benchmark_settings", {}).get("max_errors", None)
    max_errors = int(max_errors) if max_errors is not None else None
    
    # 3 Client
    client = MedicalLLMClient(config)
    
    os.makedirs("results", exist_ok=True)
    results_path = "results/benchmark_results.csv"
    fieldnames = ["id", "question", "correct_answer", "model_answer"]

    # Resume if file exists
    completed_ids = set()
    if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
        try:
            existing = pd.read_csv(results_path, usecols=["id"])
            completed_ids = set(existing["id"].dropna().astype(str).tolist())
            if completed_ids:
                print(f"Resume: {len(completed_ids)} Fragen bereits in {results_path} vorhanden, überspringe diese.")
        except Exception:
            # If the file is corrupted/partial, we just append and don't skip.
            completed_ids = set()

    total = len(data)
    remaining = sum(1 for item in data if str(item.get("id")) not in completed_ids)
    print(f"Starte Benchmarking für {total} Fragen (neu zu bearbeiten: {remaining})...")

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
                f"Frage: {item['question']}\n"
                f"Optionen: {item['options']}\n"
                "Antworte nur mit dem korrekten Buchstaben (A/B/C/D)."
            )

            print(f"[{idx}/{total}] Bearbeite Frage ID: {item_id}...")
            model_answer = client.ask_question(prompt)
            if isinstance(model_answer, str) and model_answer.startswith("Error:"):
                errors += 1
                if max_errors is not None and errors >= max_errors:
                    print(f"Abbruch: max_errors={max_errors} erreicht.")
                    writer.writerow(
                        {
                            "id": item_id,
                            "question": item["question"],
                            "correct_answer": item["correct_answer"],
                            "model_answer": model_answer,
                        }
                    )
                    f.flush()
                    break

            writer.writerow(
                {
                    "id": item_id,
                    "question": item["question"],
                    "correct_answer": item["correct_answer"],
                    "model_answer": model_answer,
                }
            )
            f.flush()
            processed_new += 1

            if sleep_s > 0:
                time.sleep(sleep_s)

            if processed_new % 50 == 0:
                elapsed = time.time() - start
                rate = processed_new / elapsed if elapsed > 0 else 0.0
                eta_s = int((remaining - processed_new) / rate) if rate > 0 else -1
                eta = f"{eta_s//3600:02d}:{(eta_s%3600)//60:02d}:{eta_s%60:02d}" if eta_s >= 0 else "?"
                print(f"Progress: {processed_new}/{remaining} neu, Errors: {errors}, Rate: {rate:.2f} q/s, ETA: {eta}")

    print(f"Benchmark abgeschlossen. Ergebnisse in {results_path}")

    # Evaluate / Score
    try:
        from evaluate import print_terminal_report, write_report_jsonl

        report_path = "results/benchmark_report.jsonl"
        report = write_report_jsonl(results_path, out_path=report_path)
        print_terminal_report(results_path)
        print(f"Auswertung gespeichert in: {report['path']}")
    except Exception as e:
        print(f"Warnung: Auswertung fehlgeschlagen: {e}")

if __name__ == "__main__":
    main()

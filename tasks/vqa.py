"""
VQA Task Runner – used for VQA-Med-2019 and RadImageNet-VQA.

RadImageNet-VQA has two question types (see Zheng et al., 2023 / paper):
  - MCQ  : option letter extracted with rule-based parser → accuracy
  - Open : free-text answer → evaluated with LLM-as-a-Judge

VQA-Med-2019 is open-ended only.

Item schema (from vision_benchmarks.py):
  { id, benchmark, question, answer, image (PIL|None), image_format, meta }

Results CSV columns:
  id, benchmark, question_type, question, reference_answer, model_answer
  question_type: "mcq" | "open"
"""
import os
import csv
import time
import base64
import io

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_b64(image, fmt: str = "jpeg") -> str:
    """Convert a PIL Image to a base64 string. Returns '' if image is None."""
    if image is None:
        return ""
    try:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format=fmt.upper() if fmt.lower() != "jpg" else "JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _detect_question_type(item: dict) -> str:
    """
    Returns "mcq", "yes_no", or "open".

    RadImageNet-VQA (Butsanets et al., 2025) has three task types:
      - mcq:    multiple-choice → rule-based letter extraction → accuracy
      - yes_no: closed binary question → exact-match accuracy
      - open:   free-text pathology → LLM-as-a-Judge (binary correct/incorrect)

    VQA-Med-2019 is open-ended only → BLEU + exact-match.
    """
    q_type = str(item.get("meta", {}).get("question_type") or "").lower()
    if q_type == "mcq" or item.get("options"):
        return "mcq"
    if q_type == "yes_no":
        return "yes_no"
    return "open"


def _build_mcq_prompt(item: dict) -> str:
    options_str = ", ".join(
        f"{opt['key']}: {opt['value']}" for opt in item.get("options", [])
    )
    return (
        f"Frage zum medizinischen Bild: {item['question']}\n"
        f"Optionen: {options_str}\n"
        "Antworte nur mit dem korrekten Buchstaben (A/B/C/D)."
    )


def _build_yes_no_prompt(item: dict) -> str:
    return (
        f"Frage zum medizinischen Bild: {item['question']}\n"
        "Antworte nur mit 'Yes' oder 'No'."
    )


def _build_open_prompt(item: dict) -> str:
    return (
        f"Frage zum medizinischen Bild: {item['question']}\n"
        "Antworte mit einer kurzen, präzisen medizinischen Antwort."
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(config: dict, client, data: list, results_path: str) -> str:
    """
    Run a VQA benchmark (image + text) and write incremental results.

    For open-ended questions the model_answer column stores the raw model
    response; LLM-as-a-Judge scoring is done in evaluate.py.

    Returns the path to the written CSV.
    """
    sleep_s = float(config.get("benchmark_settings", {}).get("sleep_s", 0) or 0)
    max_errors = config.get("benchmark_settings", {}).get("max_errors", None)
    max_errors = int(max_errors) if max_errors is not None else None

    fieldnames = ["id", "benchmark", "question_type", "question", "reference_answer", "model_answer"]

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
    print(f"VQA Benchmark: {total} Fragen ({remaining} neu zu bearbeiten)...")

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

            q_type = _detect_question_type(item)
            if q_type == "mcq":
                prompt = _build_mcq_prompt(item)
            elif q_type == "yes_no":
                prompt = _build_yes_no_prompt(item)
            else:
                prompt = _build_open_prompt(item)

            # Encode image if present
            image_b64 = _pil_to_b64(item.get("image"), fmt=item.get("image_format", "jpeg"))

            print(f"[{idx}/{total}] ID: {item_id} ({q_type}, {'bild' if image_b64 else 'kein bild'})...")

            if image_b64:
                model_answer = client.ask_with_image(
                    prompt, image_b64, item.get("image_format", "jpeg")
                )
            else:
                model_answer = client.ask_question(prompt)

            if isinstance(model_answer, str) and model_answer.startswith("Error:"):
                errors += 1
                if max_errors is not None and errors >= max_errors:
                    print(f"Abbruch: max_errors={max_errors} erreicht.")
                    writer.writerow({
                        "id": item_id,
                        "benchmark": item.get("benchmark", ""),
                        "question_type": q_type,
                        "question": item["question"],
                        "reference_answer": item.get("answer", ""),
                        "model_answer": model_answer,
                    })
                    f.flush()
                    break

            writer.writerow({
                "id": item_id,
                "benchmark": item.get("benchmark", ""),
                "question_type": q_type,
                "question": item["question"],
                "reference_answer": item.get("answer", ""),
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

    print(f"VQA Benchmark abgeschlossen. Ergebnisse in {results_path}")
    return results_path

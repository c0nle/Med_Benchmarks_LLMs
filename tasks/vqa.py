"""
VQA Task Runner – used for RadBench, VQA-Med-2019, and RadImageNet-VQA.

RadImageNet-VQA and RadBench have multiple question types:
  - MCQ     : option letter extracted with rule-based parser → accuracy
  - Yes/No  : exact-match accuracy
  - Open    : free-text answer → evaluated with LLM-as-a-Judge

VQA-Med-2019 is open-ended only (BLEU primary metric).

Item schema (from vision_benchmarks.py):
  { id, benchmark, question, answer, image (PIL|None), image_format, meta }

Results CSV columns:
  id, benchmark, question_type, question, reference_answer, model_answer
  question_type: "mcq" | "open"
"""
import os
import csv
import json
import time

import pandas as pd

from loaders.vision_benchmarks import _pil_to_b64


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
    if q_type == "yes_no":
        return "yes_no"
    if q_type == "mcq" or item.get("options"):
        return "mcq"
    return "open"


def _build_mcq_prompt(item: dict) -> str:
    options_str = ", ".join(
        f"{opt['key']}: {opt['value']}" for opt in item.get("options", [])
    )
    return (
        f"Question about the medical image: {item['question']}\n"
        f"Options: {options_str}\n"
        "Reply with only the correct letter (A/B/C/D)."
    )


def _build_yes_no_prompt(item: dict) -> str:
    return (
        f"Question about the medical image: {item['question']}\n"
        "Reply with only 'Yes' or 'No'."
    )


def _build_open_prompt(item: dict) -> str:
    return (
        f"Question: {item['question']}\n"
        "Answer using key medical terms only. No full sentences, no explanations — just the essential term(s) in English."
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(config: dict, client, data: list, results_path: str, logger=None) -> str:
    """
    Run a VQA benchmark (image + text) and write incremental results.

    For open-ended questions the model_answer column stores the raw model
    response; LLM-as-a-Judge scoring is done in evaluate.py.

    Returns the path to the written CSV.
    """
    from tasks.mcq import _parse_benchmark_settings
    sleep_s, max_errors = _parse_benchmark_settings(config)

    fieldnames = ["id", "benchmark", "question_type", "question", "reference_answer", "model_answer", "options_json"]

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

            q_type = _detect_question_type(item)
            if q_type == "mcq":
                prompt = _build_mcq_prompt(item)
            elif q_type == "yes_no":
                prompt = _build_yes_no_prompt(item)
            else:
                prompt = _build_open_prompt(item)

            # Encode image if present
            image_b64 = _pil_to_b64(item.get("image"), fmt=item.get("image_format", "jpeg"))


            if image_b64:
                model_answer = client.ask_with_image(
                    prompt, image_b64, item.get("image_format", "jpeg")
                )
            else:
                model_answer = client.ask_question(prompt)

            is_error = isinstance(model_answer, str) and model_answer.startswith("Error:")

            if logger:
                status = f"ERROR: {model_answer}" if is_error else model_answer.strip()[:60]
                logger.verbose(f"[{idx:>{len(str(total))}}/{total}] {item_id} ({q_type})  →  {status}")

            options_json = json.dumps(item.get("options") or [])

            if is_error:
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
                        "options_json": options_json,
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
                "options_json": options_json,
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

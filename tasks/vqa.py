"""
VQA Task Runner – used for RadBench, VQA-Med-2019, and RadImageNet-VQA.

RadImageNet-VQA and RadBench have multiple question types:
  - MCQ     : option letter extracted with rule-based parser → accuracy
  - Yes/No  : first-word accuracy
  - Open    : free-text answer → exact match, WBSS, LLM-as-a-Judge

VQA-Med-2019 is open-ended only.

Item schema (from vision_benchmarks.py):
  { id, benchmark, question, answer, answers (optional list of alternatives),
    options, image (PIL|None), image_format, meta: {question_type, category, cluster_id, ...} }

Results CSV columns:
  id, benchmark, question_type, category, cluster_id, question, reference_answer,
  reference_answers_json, model_answer, options_json, finish_reason, completion_tokens
  question_type: "mcq" | "yes_no" | "open"

Prompts (user message; the client adds its system prompt):
  MCQ    : "Question about the medical image: …\nOptions: A: …, B: …\nReply with only the correct letter (A/B/…)."
  Yes/No : "Question about the medical image: …\nReply with only 'Yes' or 'No'."
  Open   : "Question: …\nAnswer the question with a single word or a short phrase."
           (standard short-answer VQA instruction; until 2026-10 the open prompt asked for
           "key medical terms only", which is not comparable with published VQA results)
"""
import json

from loaders.vision_benchmarks import _pil_to_b64
from tasks.mcq import META_FIELDS, run_items


FIELDNAMES = [
    "id", "benchmark", "question_type", "category", "cluster_id", "question",
    "reference_answer", "reference_answers_json", "model_answer", "options_json",
] + META_FIELDS


def _detect_question_type(item: dict) -> str:
    """
    Returns "mcq", "yes_no", or "open".

    RadImageNet-VQA (Butsanets et al., 2025) has three task types:
      - mcq:    multiple-choice → rule-based letter extraction → accuracy
      - yes_no: closed binary question → first-word accuracy
      - open:   free-text → LLM-as-a-Judge (binary correct/incorrect)

    VQA-Med-2019 is open-ended only → exact match (official accuracy) + LLM-Judge.
    """
    q_type = str(item.get("meta", {}).get("question_type") or "").lower()
    if q_type == "yes_no":
        return "yes_no"
    if q_type == "mcq" or item.get("options"):
        return "mcq"
    return "open"


def _build_mcq_prompt(item: dict) -> str:
    opts = item.get("options", [])
    options_str = ", ".join(f"{opt['key']}: {opt['value']}" for opt in opts)
    keys = "/".join(opt["key"] for opt in opts) if opts else "A/B/C/D"
    return (
        f"Question about the medical image: {item['question']}\n"
        f"Options: {options_str}\n"
        f"Reply with only the correct letter ({keys})."
    )


def _build_yes_no_prompt(item: dict) -> str:
    return (
        f"Question about the medical image: {item['question']}\n"
        "Reply with only 'Yes' or 'No'."
    )


OPEN_INSTRUCTION = "Answer the question with a single word or a short phrase."


def _build_open_prompt(item: dict) -> str:
    return (
        f"Question: {item['question']}\n"
        f"{OPEN_INSTRUCTION}"
    )


def build_prompt(item: dict) -> str:
    q_type = _detect_question_type(item)
    if q_type == "mcq":
        return _build_mcq_prompt(item)
    if q_type == "yes_no":
        return _build_yes_no_prompt(item)
    return _build_open_prompt(item)


def _images_b64(item: dict) -> list:
    fmt = item.get("image_format", "png")
    images = item.get("meta", {}).get("all_images") or (
        [item["image"]] if item.get("image") is not None else []
    )
    encoded = [_pil_to_b64(img, fmt=fmt) for img in images]
    if any(not b for b in encoded):
        raise ValueError(f"{item.get('id')}: could not encode {sum(not b for b in encoded)} of "
                         f"{len(encoded)} images; refusing to send the question with fewer images")
    return encoded


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(config: dict, client, data: list, results_path: str, logger=None) -> str:
    """
    Run a VQA benchmark (image + text) and write incremental results.

    For open-ended questions the model_answer column stores the raw model
    response; scoring (incl. LLM-as-a-Judge) is done in evaluate.py.
    Concurrency, resume and abort handling: tasks.mcq.run_items.

    Returns the path to the written CSV.
    """
    def _ask(item):
        prompt = build_prompt(item)
        images_b64 = _images_b64(item)
        if images_b64:
            return client.ask_with_images(prompt, images_b64, item.get("image_format", "png"))
        return client.ask_question(prompt)

    def _base_row(item):
        meta = item.get("meta", {}) or {}
        answers = item.get("answers") or [item.get("answer", "")]
        return {
            "benchmark": item.get("benchmark", ""),
            "question_type": _detect_question_type(item),
            "category": meta.get("category", ""),
            "cluster_id": meta.get("cluster_id", ""),
            "question": item["question"],
            "reference_answer": item.get("answer", ""),
            "reference_answers_json": json.dumps([str(a) for a in answers], ensure_ascii=False),
            "options_json": json.dumps(item.get("options") or [], ensure_ascii=False),
        }

    return run_items(config, client, data, results_path, FIELDNAMES,
                     ask=_ask, base_row=_base_row, logger=logger,
                     describe=_detect_question_type)

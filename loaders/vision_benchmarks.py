"""
Vision benchmark loaders: VQA-Med-2019 and RadImageNet-VQA.

Each item follows this schema:
{
    "id":           str,
    "benchmark":    str,
    "question":     str,
    "answer":       str,       # reference / ground-truth answer
    "image":        PIL.Image or None,
    "image_format": str,       # "jpeg" | "png"
    "meta":         dict,
}
"""
import os
from pathlib import Path

from loaders.text_benchmarks import _ensure_hf_cache_dir, _load_local_parquet

_ensure_hf_cache_dir()

from datasets import load_dataset


def _pil_to_b64(image) -> str:
    """Convert a PIL Image to a base64 JPEG string (used by the client)."""
    import io, base64
    if image is None:
        return ""
    buf = io.BytesIO()
    rgb = image.convert("RGB")
    rgb.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# VQA-Med-2019
# ---------------------------------------------------------------------------

def _format_vqa_med_item(item: dict, idx: int) -> dict:
    """Normalise a VQA-Med-2019 row."""
    # HF dataset fields vary by source; try common field names
    question = (
        item.get("question")
        or item.get("Question")
        or item.get("q")
        or ""
    )
    answer = (
        item.get("answer")
        or item.get("Answer")
        or item.get("a")
        or item.get("gt")
        or ""
    )
    image = item.get("image") or item.get("img") or None

    return {
        "id": str(item.get("id") or item.get("qid") or item.get("image_name") or f"vqamed-{idx}"),
        "benchmark": "VQA-Med-2019",
        "question": str(question),
        "answer": str(answer),
        "image": image,
        "image_format": "jpeg",
        "meta": {
            "category": item.get("category") or item.get("Category") or "",
            "source": "VQA-Med-2019",
        },
    }


def load_vqa_med_2019(limit=None):
    """
    Loads the VQA-Med-2019 benchmark (ImageCLEF 2019 Medical VQA).

    Priority:
    1) Local parquet via env var VQA_MED_2019_PARQUET_PATH
    2) Auto-detect data/vqa_med*.parquet
    3) Hugging Face dataset (flaviagiammarino/vqa-rad or similar)

    Note: VQA-Med-2019 is image-based. The model receives the question AND the
    medical image (base64-encoded JPEG) and must produce a free-text answer.
    A vision-capable LLM (VLM) is required.
    """
    print("--- Lade VQA-Med-2019 ---")

    local_path = os.getenv("VQA_MED_2019_PARQUET_PATH")
    if not local_path:
        for candidate in (
            "data/vqa_med_2019.parquet",
            "data/vqa-med-2019.parquet",
            "data/vqa_med.parquet",
        ):
            if Path(candidate).exists():
                local_path = candidate
                break

    if local_path:
        dataset = _load_local_parquet(local_path)
    else:
        # Official community mirrors of VQA-Med-2019 (Ben Abacha et al., ImageCLEF 2019)
        # Original: https://github.com/abachaa/VQA-Med-2019
        hf_candidates = [
            ("simwit/vqa-med-2019", "test"),
            ("simwit/vqa-med-2019", "train"),
            ("claudioreeves/imageclef-vqa-med-2019", "test"),
            ("claudioreeves/imageclef-vqa-med-2019", "train"),
        ]
        dataset = None
        last_err = None
        for hf_id, split in hf_candidates:
            try:
                dataset = load_dataset(hf_id, split=split)
                print(f"  Geladen von HuggingFace: {hf_id} (split={split})")
                break
            except Exception as e:
                last_err = e
        if dataset is None:
            raise RuntimeError(
                "VQA-Med-2019 konnte nicht geladen werden.\n\n"
                "Fix:\n"
                "1) Parquet-Datei bereitstellen und Pfad setzen:\n"
                "   export VQA_MED_2019_PARQUET_PATH=/pfad/zur/vqa_med_2019.parquet\n"
                "   (Format: Spalten 'question', 'answer', 'image')\n"
                f"Originalfehler: {type(last_err).__name__}: {last_err}"
            ) from last_err

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [_format_vqa_med_item(item, idx) for idx, item in enumerate(dataset)]


# ---------------------------------------------------------------------------
# RadImageNet-VQA
# ---------------------------------------------------------------------------

def _detect_radimagenet_qtype(item: dict) -> str:
    """
    Detect whether a RadImageNet-VQA item is MCQ, yes/no (closed), or open-ended.

    Per Butsanets et al. 2025 (RadImageNet-VQA paper):
    - Multiple-choice (MC): option letter extracted with rule-based parser → accuracy
    - Closed-ended (yes/no): exact-match accuracy
    - Open-ended (pathology description): LLM-as-a-Judge (binary correct/incorrect)

    The dataset stores this in a 'question_type' or 'task' field.
    """
    explicit = (
        str(item.get("question_type") or item.get("type") or item.get("task") or "")
    ).lower()

    if "mcq" in explicit or "multiple" in explicit or "choice" in explicit:
        return "mcq"
    if "yes" in explicit or "no" in explicit or "closed" in explicit or "binary" in explicit:
        return "yes_no"
    if "open" in explicit or "free" in explicit or "pathology" in explicit:
        return "open"

    # Infer from answer value and presence of choices
    if item.get("choices") or item.get("options"):
        return "mcq"
    answer = str(item.get("answer") or item.get("gt") or "").strip().lower()
    if answer in {"yes", "no"}:
        return "yes_no"
    return "open"


def _format_radimagenet_item(item: dict, idx: int) -> dict:
    """Normalise a RadImageNet-VQA row."""
    question = (
        item.get("question")
        or item.get("Question")
        or item.get("query")
        or ""
    )
    answer = (
        item.get("answer")
        or item.get("Answer")
        or item.get("label")
        or item.get("gt")
        or ""
    )
    image = item.get("image") or item.get("img") or None
    modality = item.get("modality") or item.get("Modality") or ""
    q_type = _detect_radimagenet_qtype(item)

    # For MCQ items, build options list if present
    choices_raw = item.get("choices") or item.get("options") or {}
    if isinstance(choices_raw, dict):
        options = [{"key": k, "value": v} for k, v in choices_raw.items()]
    elif isinstance(choices_raw, list) and choices_raw:
        if isinstance(choices_raw[0], str):
            keys = ["A", "B", "C", "D", "E"]
            options = [{"key": keys[i], "value": v} for i, v in enumerate(choices_raw)]
        else:
            options = choices_raw
    else:
        options = []

    return {
        "id": str(item.get("id") or item.get("image_id") or f"radimagenet-{idx}"),
        "benchmark": "RadImageNet-VQA",
        "question": str(question),
        "answer": str(answer),
        "options": options,          # non-empty only for MCQ items
        "image": image,
        "image_format": "jpeg",
        "meta": {
            "modality": modality,    # CT | MRI | X-ray
            "question_type": q_type, # "mcq" | "open"  – used by tasks/vqa.py
            "source": "RadImageNet-VQA",
        },
    }


def load_radimagenet_vqa(limit=None):
    """
    Loads the RadImageNet-VQA benchmark (CT, MRI, X-ray image understanding).

    Priority:
    1) Local parquet via env var RADIMAGENET_VQA_PARQUET_PATH
    2) Auto-detect data/radimagenet*.parquet
    3) Hugging Face dataset (UCSC-VLAA/RadImageNet-VQA or similar)

    Note: Requires a vision-capable LLM (VLM).
    """
    print("--- Lade RadImageNet-VQA (CT/MRT/Röntgen Bildverständnis) ---")

    local_path = os.getenv("RADIMAGENET_VQA_PARQUET_PATH")
    if not local_path:
        for candidate in (
            "data/radimagenet_vqa.parquet",
            "data/radimagenet-vqa.parquet",
            "data/radimagenet.parquet",
        ):
            if Path(candidate).exists():
                local_path = candidate
                break

    if local_path:
        dataset = _load_local_parquet(local_path)
    else:
        # Official HF dataset: raidium/RadImageNet-VQA (Butsanets et al., 2025)
        hf_candidates = [
            ("raidium/RadImageNet-VQA", "test"),
            ("raidium/RadImageNet-VQA", "train"),
            ("raidium/RadImageNet-VQA", "validation"),
        ]
        dataset = None
        last_err = None
        for hf_id, split in hf_candidates:
            try:
                dataset = load_dataset(hf_id, split=split)
                print(f"  Geladen von HuggingFace: {hf_id} (split={split})")
                break
            except Exception as e:
                last_err = e
        if dataset is None:
            raise RuntimeError(
                "RadImageNet-VQA konnte nicht geladen werden.\n\n"
                "Fix:\n"
                "1) Parquet-Datei bereitstellen und Pfad setzen:\n"
                "   export RADIMAGENET_VQA_PARQUET_PATH=/pfad/zur/radimagenet_vqa.parquet\n"
                "   (Format: Spalten 'question', 'answer', 'image', optional 'modality')\n"
                f"Originalfehler: {type(last_err).__name__}: {last_err}"
            ) from last_err

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [_format_radimagenet_item(item, idx) for idx, item in enumerate(dataset)]

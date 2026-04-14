"""
Vision benchmark loaders: RadBench, VQA-Med-2019, RadImageNet-VQA.

Each item follows this schema:
{
    "id":           str,
    "benchmark":    str,
    "question":     str,
    "answer":       str,       # reference / ground-truth answer
    "options":      list,      # [{"key": "A", "value": "..."}, ...] — MCQ only
    "image":        PIL.Image or None,
    "image_format": str,       # "jpeg" | "png"
    "meta": {
        "question_type": str,  # "mcq" | "yes_no" | "open"
        ...
    }
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
    # HF dataset sometimes returns answer as a list or as a list-repr string
    # e.g. ['cta - ct angiography'] → 'cta - ct angiography'
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    else:
        answer = str(answer).strip()
        if answer.startswith("['") and answer.endswith("']"):
            answer = answer[2:-2]
        elif answer.startswith('["') and answer.endswith('"]'):
            answer = answer[2:-2]

    image = item.get("image") or item.get("img") or None

    return {
        "id": str(item.get("id") or item.get("qid") or item.get("image_name") or f"vqamed-{idx}"),
        "benchmark": "VQA-Med-2019",
        "question": str(question),
        "answer": answer,
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
        # Config name is "alignment"; available split is "train".
        hf_candidates = [
            ("raidium/RadImageNet-VQA", "alignment", "train"),
            ("raidium/RadImageNet-VQA", "alignment", "test"),
            ("raidium/RadImageNet-VQA", None, "test"),
            ("raidium/RadImageNet-VQA", None, "train"),
        ]
        dataset = None
        last_err = None
        for hf_id, config_name, split in hf_candidates:
            try:
                kwargs = {"split": split}
                if config_name:
                    kwargs["name"] = config_name
                dataset = load_dataset(hf_id, **kwargs)
                label = f"{hf_id}" + (f" (config={config_name})" if config_name else "") + f" (split={split})"
                print(f"  Geladen von HuggingFace: {label}")
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


# ---------------------------------------------------------------------------
# RadBench (harrison.ai) – VLM benchmark with X-ray images
# ---------------------------------------------------------------------------

def _detect_radbench_qtype(item: dict) -> str:
    """
    RadBench has two question types stored in the A_TYPE field:
      - "closed-ended" → MCQ with answer options (377 questions)
      - "open-ended"   → free-text answer, evaluated with BLEU + LLM-judge (120 questions)
    """
    a_type = str(item.get("A_TYPE") or item.get("a_type") or item.get("type") or "").lower()
    if "open" in a_type:
        return "open"
    # Check for presence of options as fallback
    has_options = bool(item.get("OPTIONS") or item.get("options") or item.get("choices"))
    if has_options:
        return "mcq"
    return "open"


def _format_radbench_item(item: dict, idx: int) -> dict:
    """
    Normalise a RadBench (harrison.ai) row into the shared VQA schema.

    RadBench dataset fields (from https://github.com/harrison-ai/radbench):
      imageSource, CASE_ID, imageIDs, modality, IMAGE_ORGAN, PRIMARY_DX,
      QUESTION, Q_TYPE, ANSWER, A_TYPE, OPTIONS
    """
    q_type = _detect_radbench_qtype(item)

    # Build options list for MCQ items
    opts_raw = item.get("OPTIONS") or item.get("options") or item.get("choices") or {}
    if isinstance(opts_raw, dict):
        options = [{"key": k, "value": v} for k, v in opts_raw.items()]
    elif isinstance(opts_raw, list) and opts_raw:
        if isinstance(opts_raw[0], str):
            keys = ["A", "B", "C", "D", "E"]
            options = [{"key": keys[i], "value": v} for i, v in enumerate(opts_raw)]
        else:
            options = opts_raw
    else:
        options = []

    # If closed-ended but options look like yes/no, treat as yes_no
    if q_type == "mcq" and len(options) == 2:
        vals = {o["value"].strip().lower() for o in options}
        if vals <= {"yes", "no"}:
            q_type = "yes_no"

    answer = str(item.get("ANSWER") or item.get("answer") or item.get("gt") or "").strip()

    image = item.get("image") or item.get("img") or None

    return {
        "id": str(item.get("CASE_ID") or item.get("id") or item.get("qid") or f"radbench-{idx}"),
        "benchmark": "RadBench",
        "question": str(item.get("QUESTION") or item.get("question") or ""),
        "answer": answer,
        "options": options,
        "image": image,
        "image_format": "jpeg",
        "meta": {
            "question_type": q_type,       # "mcq" | "yes_no" | "open"
            "q_type_category": str(item.get("Q_TYPE") or ""),   # Pathology, Clinical, …
            "modality": str(item.get("modality") or "XR"),
            "organ": str(item.get("IMAGE_ORGAN") or ""),
            "source": str(item.get("imageSource") or "RadBench"),
        },
    }


def load_radbench(limit=None):
    """
    Loads the RadBench benchmark (harrison.ai).

    RadBench is a **VLM benchmark** using plain X-ray images (XR) from
    MedPix and Radiopaedia cases. It is NOT text-only.
      - 89 unique cases (40 MedPix, 49 Radiopaedia)
      - 497 questions: 377 closed-ended MCQ/Yes-No + 120 open-ended
      - Modality: X-ray (plain film), sometimes multi-image per case

    Evaluation:
      - Closed-ended MCQ   → letter-accuracy (rule-based)
      - Closed-ended Yes/No → exact-match accuracy
      - Open-ended          → BLEU + LLM-as-a-Judge

    RadBench is NOT on HuggingFace. Download from:
      https://github.com/harrison-ai/radbench
      https://harrison-ai.github.io/radbench/

    Priority:
    1) Local parquet via env var RADBENCH_PARQUET_PATH
    2) Auto-detect data/radbench*.parquet
    """
    print("--- Lade RadBench (harrison.ai – VLM Röntgen-Benchmark) ---")

    local_path = os.getenv("RADBENCH_PARQUET_PATH")
    if not local_path:
        for candidate in (
            "data/radbench.parquet",
            "data/radbench-test.parquet",
            "data/radbench_test.parquet",
        ):
            if Path(candidate).exists():
                local_path = candidate
                break

    if local_path is None:
        raise RuntimeError(
            "RadBench konnte nicht geladen werden.\n\n"
            "Das Dataset ist NICHT auf HuggingFace. Bitte:\n"
            "1) Daten von https://github.com/harrison-ai/radbench herunterladen.\n"
            "2) Als Parquet ablegen und Pfad setzen:\n"
            "   export RADBENCH_PARQUET_PATH=/pfad/zur/radbench.parquet\n"
            "   (Erwartete Spalten: QUESTION, ANSWER, A_TYPE, OPTIONS,\n"
            "    IMAGE_ORGAN, modality, CASE_ID, optional: image)\n"
            "Hinweis: Bilder (X-rays) sind separat in /images/ im Repo.\n"
            "         Ohne Bild läuft das Modell text-only (gültig als Baseline).\n"
        )

    dataset = _load_local_parquet(local_path)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [_format_radbench_item(item, idx) for idx, item in enumerate(dataset)]

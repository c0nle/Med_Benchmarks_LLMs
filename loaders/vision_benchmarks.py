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
from pathlib import Path

from loaders.text_benchmarks import _load_local_parquet, _load_local_file

# ---------------------------------------------------------------------------
# Expected data file locations (place files here before running)
# ---------------------------------------------------------------------------

_VQA_MED_PATH                = Path("data/vqa_med_2019.parquet")
_RADBENCH_PATH               = Path("data/radbench.csv")
_RADIMAGENET_BENCHMARK_PATH  = Path("data/radimagenet_vqa_benchmark.parquet")


def _pil_to_b64(image, fmt: str = "jpeg") -> str:
    """Convert a PIL Image to a base64 string. Returns '' if image is None."""
    import io, base64
    if image is None:
        return ""
    try:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format=fmt.upper() if fmt.lower() != "jpg" else "JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _build_options_list(raw) -> list:
    """Normalise a raw choices/options field to [{'key': ..., 'value': ...}]."""
    if isinstance(raw, dict):
        return [{"key": k, "value": v} for k, v in raw.items()]
    if isinstance(raw, list) and raw:
        if isinstance(raw[0], str):
            return [{"key": k, "value": v} for k, v in zip("ABCDE", raw)]
        return raw
    return []


# ---------------------------------------------------------------------------
# VQA-Med-2019  →  data/vqa_med_2019.parquet
# ---------------------------------------------------------------------------

def _format_vqa_med_item(item: dict, idx: int) -> dict:
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
    Loads VQA-Med-2019 from data/vqa_med_2019.parquet.
    Download (VQA-Med-2019): https://huggingface.co/datasets/simwit/vqa-med-2019
    Download (VQA-RAD, smaller): https://huggingface.co/datasets/flaviagiammarino/vqa-rad
    Either dataset works; VQA-Med-2019 is the ImageCLEF 2019 benchmark (Ben Abacha et al.).
    """
    print("--- Lade VQA-Med-2019 ---")

    if not _VQA_MED_PATH.exists():
        raise FileNotFoundError(
            f"VQA-Med-2019 nicht gefunden: {_VQA_MED_PATH}\n"
            "Download: https://huggingface.co/datasets/flaviagiammarino/vqa-rad\n"
            "Datei ablegen als: data/vqa_med_2019.parquet"
        )

    items = _load_local_parquet(str(_VQA_MED_PATH))
    if limit:
        items = items[:limit]
    return [_format_vqa_med_item(item, idx) for idx, item in enumerate(items)]


# ---------------------------------------------------------------------------
# RadImageNet-VQA
# ---------------------------------------------------------------------------

def _format_radimagenet_benchmark_item(item: dict, idx: int) -> dict:
    """
    Normalise a RadImageNet-VQA benchmark split row.

    Schema (raidium/RadImageNet-VQA, config=benchmark, split=test, 9K items):
      image         – PIL image
      question      – string
      choices       – list of strings for MCQ, None otherwise
      answer        – letter (A/B/C/D) for MCQ, "yes"/"no" for closed, text for open
      question_type – "multiple_choice" | "closed" | "open"
      metadata      – {content_type, correct_text, is_abnormal, location, modality, pathology, question_id}
    """
    meta = item.get("metadata") or {}
    raw_qt = str(item.get("question_type") or "").lower()

    if raw_qt == "multiple_choice":
        q_type = "mcq"
        raw_choices = item.get("choices") or []
        options = [{"key": k, "value": str(v)} for k, v in zip("ABCD", raw_choices)]
    elif raw_qt == "closed":
        q_type = "yes_no"
        options = []
    else:
        q_type = "open"
        options = []

    return {
        "id": str(meta.get("question_id") or f"radimagenet-{idx}"),
        "benchmark": "RadImageNet-VQA",
        "question": str(item.get("question") or ""),
        "answer": str(item.get("answer") or ""),
        "options": options,
        "image": item.get("image"),
        "image_format": "jpeg",
        "meta": {
            "question_type": q_type,
            "modality": str(meta.get("modality") or "").upper(),
            "pathology": str(meta.get("pathology") or ""),
            "location": str(meta.get("location") or ""),
            "source": "RadImageNet-VQA",
        },
    }


def load_radimagenet_vqa(limit=None):
    """
    Loads the RadImageNet-VQA benchmark test split (9K items, CT/MRI/X-ray).

    Uses raidium/RadImageNet-VQA, config=benchmark, split=test:
      - 2000 multiple_choice (MCQ, A/B/C/D) → Accuracy
      - 5000 closed (yes/no)                → Exact-match Accuracy
      - 2000 open (free-text pathology)     → WBSS + LLM-as-a-Judge

    Download (once):
      HF_TOKEN=hf_... python3 -c "
      from datasets import load_dataset; import os
      ds = load_dataset('raidium/RadImageNet-VQA', name='benchmark', split='test',
                        token=os.environ['HF_TOKEN'])
      ds.to_parquet('data/radimagenet_vqa_benchmark.parquet')"

    Note: Requires a vision-capable LLM (VLM).
    """
    print("--- Lade RadImageNet-VQA (CT/MRT/Röntgen Benchmark) ---")

    if not _RADIMAGENET_BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"RadImageNet-VQA benchmark split nicht gefunden: {_RADIMAGENET_BENCHMARK_PATH}\n"
            "Download:\n"
            "  HF_TOKEN=hf_... python3 -c \"\n"
            "  from datasets import load_dataset; import os\n"
            "  ds = load_dataset('raidium/RadImageNet-VQA', name='benchmark', split='test',\n"
            "                    token=os.environ['HF_TOKEN'])\n"
            "  ds.to_parquet('data/radimagenet_vqa_benchmark.parquet')\""
        )

    items = _load_local_parquet(str(_RADIMAGENET_BENCHMARK_PATH))
    if limit:
        items = items[:limit]

    print(f"  {len(items)} Items geladen.")
    return [_format_radimagenet_benchmark_item(item, idx) for idx, item in enumerate(items)]


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


_RADBENCH_IMAGE_DIR = Path("data/radbench_images")


def _load_radbench_images(image_ids_str: str) -> list:
    """
    Load PIL images for a RadBench row from the local image cache.

    image_ids_str is a comma-separated list of UUIDs (MedPix) or URLs (Radiopaedia).
    Returns a list of PIL Images for each ID that has a file in data/radbench_images/.
    """
    import io
    try:
        from PIL import Image as _PILImage
    except ImportError:
        return []

    images = []
    if not image_ids_str or str(image_ids_str).lower() in ("nan", "none", ""):
        return images

    for img_id in str(image_ids_str).split(","):
        img_id = img_id.strip()
        # Derive the same filename used by download_radbench_images.py
        fname = img_id.split("/")[-1].split("?")[0]
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fname += ".jpg"
        path = _RADBENCH_IMAGE_DIR / fname
        if path.exists():
            try:
                images.append(_PILImage.open(path).copy())
            except Exception:
                pass
    return images


def _format_radbench_item(item: dict, idx: int) -> dict:
    """
    Normalise a RadBench (harrison.ai) row into the shared VQA schema.

    RadBench dataset fields (from https://github.com/harrison-ai/radbench):
      imageSource, CASE_ID, imageIDs, modality, IMAGE_ORGAN, PRIMARY_DX,
      QUESTION, Q_TYPE, ANSWER, A_TYPE, OPTIONS
    """
    q_type = _detect_radbench_qtype(item)

    raw_opts = item.get("OPTIONS") or item.get("options") or item.get("choices") or {}
    # RadBench stores options as a comma-separated string e.g. "yes,no" or "frontal,oblique,lateral"
    if isinstance(raw_opts, str) and raw_opts.strip() and raw_opts.strip().lower() not in ("nan", "none"):
        raw_opts = [v.strip() for v in raw_opts.split(",") if v.strip()]
    options = _build_options_list(raw_opts)

    # If closed-ended but options look like yes/no, treat as yes_no
    if q_type == "mcq" and len(options) == 2:
        vals = {o["value"].strip().lower() for o in options}
        if vals <= {"yes", "no"}:
            q_type = "yes_no"

    answer = str(item.get("ANSWER") or item.get("answer") or item.get("gt") or "").strip()

    # Use embedded image if present (parquet), otherwise load from local image cache
    image = item.get("image") or item.get("img") or None
    images = []
    if image is not None:
        images = [image]
    else:
        images = _load_radbench_images(item.get("imageIDs") or "")

    # Primary image for single-image tasks; all images passed in meta for multi-image
    primary_image = images[0] if images else None

    return {
        "id": str(item.get("CASE_ID") or item.get("id") or item.get("qid") or f"radbench-{idx}"),
        "benchmark": "RadBench",
        "question": str(item.get("QUESTION") or item.get("question") or ""),
        "answer": answer,
        "options": options,
        "image": primary_image,
        "image_format": "jpeg",
        "meta": {
            "question_type": q_type,       # "mcq" | "yes_no" | "open"
            "q_type_category": str(item.get("Q_TYPE") or ""),   # Pathology, Clinical, …
            "modality": str(item.get("modality") or "XR"),
            "organ": str(item.get("IMAGE_ORGAN") or ""),
            "source": str(item.get("imageSource") or "RadBench"),
            "all_images": images,          # full list for multi-image questions
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
      - Open-ended          → WBSS + LLM-as-a-Judge

    Download: https://github.com/harrison-ai/radbench
    Datei ablegen als: data/radbench.csv
    X-ray Bilder herunterladen: python3 scripts/download_radbench_images.py
    """
    print("--- Lade RadBench (harrison.ai – VLM Röntgen-Benchmark) ---")

    if not _RADBENCH_PATH.exists():
        raise FileNotFoundError(
            f"RadBench nicht gefunden: {_RADBENCH_PATH}\n"
            "Download: git clone https://github.com/harrison-ai/radbench data/radbench_repo\n"
            "Dann: cp data/radbench_repo/data/radbench/radbench.csv data/radbench.csv\n"
            "Bilder: python3 scripts/download_radbench_images.py"
        )

    items = _load_local_file(str(_RADBENCH_PATH))

    # Filter out MedPix cases that have no local image (MedPix API is no longer available)
    before = len(items)
    items = [it for it in items
             if str(it.get("imageSource") or "").strip().lower() != "medpix"
             or any(
                 (_RADBENCH_IMAGE_DIR / (img_id.strip().split("/")[-1].split("?")[0] + (
                     "" if img_id.strip().split("/")[-1].split("?")[0].lower().endswith((".jpg",".jpeg",".png")) else ".jpg"
                 ))).exists()
                 for img_id in str(it.get("imageIDs") or "").split(",") if img_id.strip()
             )]
    n_filtered = before - len(items)
    if n_filtered:
        print(f"  {n_filtered} MedPix-Fragen ohne Bild herausgefiltert ({before} → {len(items)})")

    if limit:
        items = items[:limit]

    formatted = [_format_radbench_item(item, idx) for idx, item in enumerate(items)]
    n_with_img = sum(1 for it in formatted if it["image"] is not None)
    if _RADBENCH_IMAGE_DIR.exists():
        print(f"  {n_with_img}/{len(formatted)} Fragen mit Bild geladen aus {_RADBENCH_IMAGE_DIR}/")
    else:
        print(f"  Keine Bilder gefunden. Für Vision-Evaluation: python3 scripts/download_radbench_images.py")
    return formatted

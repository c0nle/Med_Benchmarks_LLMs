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

_VQA_MED_PATH    = Path("data/vqa_med_2019.parquet")
_RADBENCH_PATH   = Path("data/radbench.csv")


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

def _format_radimagenet_item(item: dict, idx: int) -> dict:
    """
    Normalise a RadImageNet-VQA row.

    The HuggingFace dataset (raidium/RadImageNet-VQA, alignment config) uses a
    conversations + metadata schema:
      conversations: [
        {"from": "human",    "value": "<image>\\nDescribe the clinical findings..."},
        {"from": "template", "value": "CT examination of the abdomen ..."},
      ]
      metadata: {"content_type": "description", "modality": "ct", "pathology": "...", ...}

    All items in the alignment config are open-ended image descriptions (no MCQ/Yes-No).
    """
    convs = item.get("conversations") or []
    meta  = item.get("metadata") or {}

    # Extract question from human turn, strip <image> tag
    question = ""
    for turn in convs:
        if turn.get("from") == "human":
            question = str(turn.get("value") or "").replace("<image>", "").strip()
            break

    # Extract answer from template/assistant turn
    answer = ""
    for turn in convs:
        if turn.get("from") in ("template", "gpt", "assistant"):
            answer = str(turn.get("value") or "").strip()
            break

    # Fallback to flat fields (for custom local parquet with different schema)
    if not question:
        question = str(item.get("question") or item.get("Question") or item.get("query") or "")
    if not answer:
        answer = str(item.get("answer") or item.get("Answer") or item.get("label") or item.get("gt") or "")

    modality = str(meta.get("modality") or item.get("modality") or "").upper()

    return {
        "id": str(item.get("id") or item.get("image_id") or f"radimagenet-{idx}"),
        "benchmark": "RadImageNet-VQA",
        "question": question,
        "answer": answer,
        "options": [],
        "image": item.get("image") or item.get("img") or None,
        "image_format": "png",
        "meta": {
            "modality": modality,
            "question_type": "open",  # alignment config is description-only
            "pathology": str(meta.get("pathology") or ""),
            "source": "RadImageNet-VQA",
        },
    }


def _load_radimagenet_local(limit: int = None) -> list:
    """
    Load RadImageNet-VQA from data/radimagenet_vqa_*.parquet chunks via pyarrow.
    Decodes image bytes → PIL on the fly. Stops at `limit` items.
    """
    import glob as _glob
    import io
    import pyarrow.parquet as pq
    from PIL import Image as _PILImage

    files = sorted(_glob.glob("data/radimagenet_vqa_*.parquet"))

    if not files:
        return None

    print(f"  {len(files)} lokale Chunk(s) gefunden, lade via pyarrow...")

    def _decode_image(val):
        """Convert parquet image value (bytes dict or raw bytes) → PIL Image."""
        try:
            if isinstance(val, dict):
                raw = val.get("bytes") or val.get("data")
            elif isinstance(val, (bytes, bytearray)):
                raw = val
            else:
                return None
            return _PILImage.open(io.BytesIO(raw)) if raw else None
        except Exception:
            return None

    def _decode_cell(key, val):
        """Unwrap pyarrow scalars and decode images."""
        import pyarrow as pa
        if isinstance(val, pa.lib.BaseArrowObject):
            val = val.as_py()
        if key == "image":
            return _decode_image(val)
        # conversations / metadata come back as dicts/lists already via as_py()
        return val

    items = []
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for batch in pf.iter_batches(batch_size=256):
            columns = {col: batch.column(col).to_pylist() for col in batch.schema.names}
            n = batch.num_rows
            for i in range(n):
                row = {}
                for col, vals in columns.items():
                    v = vals[i]
                    if col == "image":
                        v = _decode_image(v)
                    row[col] = v
                items.append(row)
                if limit is not None and len(items) >= limit:
                    return items
    return items


def load_radimagenet_vqa(limit=None):
    """
    Loads the RadImageNet-VQA dataset (CT, MRI, X-ray image descriptions).

    Uses the raidium/RadImageNet-VQA alignment training set, which contains
    750K open-ended image description items (no MCQ/Yes-No in this config).
    All questions ask the model to describe clinical findings for a given image.

    Priority:
    1) Local parquet via env var RADIMAGENET_VQA_PARQUET_PATH (single file)
    2) Auto-detect data/radimagenet_vqa_*.parquet (numbered chunks)
    3) Auto-detect data/radimagenet*.parquet (single file)

    HuggingFace is NOT used as fallback — dataset must be available locally.
    Download: see README for instructions.

    Note: Requires a vision-capable LLM (VLM).
    """
    print("--- Lade RadImageNet-VQA (CT/MRT/Röntgen Bildverständnis) ---")

    items = _load_radimagenet_local(limit=limit)

    if items is None:
        raise FileNotFoundError(
            "RadImageNet-VQA nicht gefunden: data/radimagenet_vqa_000.parquet\n"
            "Download (einmalig, ~750K Items):\n"
            "  python3 -c \"\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('raidium/RadImageNet-VQA', name='alignment', split='train')\n"
            "  for i in range(0, len(ds), 50000):\n"
            "      ds.select(range(i, min(i+50000, len(ds)))).to_parquet(f'data/radimagenet_vqa_{i//50000:03d}.parquet')\n"
            "  \"\n"
            "Dateien ablegen als: data/radimagenet_vqa_000.parquet, data/radimagenet_vqa_001.parquet, ..."
        )

    print(f"  {len(items)} Items geladen.")
    return [_format_radimagenet_item(item, idx) for idx, item in enumerate(items)]


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
      - Open-ended          → BLEU + LLM-as-a-Judge

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

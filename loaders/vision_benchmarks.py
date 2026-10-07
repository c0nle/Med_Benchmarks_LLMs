"""
Vision benchmark loaders: RadBench, VQA-Med-2019, RadImageNet-VQA.

Each item follows this schema:
{
    "id":           str,
    "benchmark":    str,
    "question":     str,
    "answer":       str,       # reference / ground-truth answer
    "answers":      list,      # all accepted alternatives (VQA-Med-2019 only; optional)
    "options":      list,      # [{"key": "A", "value": "..."}, ...] — MCQ only
    "image":        PIL.Image or None,
    "image_format": str,       # "png" (lossless, default) | "jpeg"
    "meta": {
        "question_type": str,  # "mcq" | "yes_no" | "open"
        "category":      str,  # per-category reporting (VQA-Med question_categories,
                               # RadImageNet content_type, RadBench Q_TYPE)
        "cluster_id":    str,  # image / case id for the cluster bootstrap CI
        ...
    }
}
"""
import hashlib
import os
import re
import string
from pathlib import Path
from urllib.parse import urlparse

from loaders.text_benchmarks import _load_local_parquet, _load_local_file

# ---------------------------------------------------------------------------
# Expected data file locations (place files here before running)
# ---------------------------------------------------------------------------

_VQA_MED_PATH                = Path("data/vqa_med_2019.parquet")
_RADBENCH_PATH               = Path("data/radbench.csv")
_RADIMAGENET_BENCHMARK_PATH  = Path("data/radimagenet_vqa_benchmark.parquet")


def _pil_to_b64(image, fmt: str = "png") -> str:
    """Convert a PIL Image to a base64 string. Returns '' if image is None.

    Images are sent as PNG (lossless): greyscale ("L") and RGB keep their pixels exactly;
    other modes (e.g. the fully opaque RGBA files in RadBench) are converted to RGB.
    JPEG is only used if explicitly requested (it re-compresses lossily).
    """
    import io, base64
    if image is None:
        return ""
    try:
        buf = io.BytesIO()
        if fmt.lower() in ("jpg", "jpeg"):
            image.convert("RGB").save(buf, format="JPEG", quality=95)
        else:
            img = image if image.mode in ("L", "RGB") else image.convert("RGB")
            img.save(buf, format=fmt.upper())
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _number_image_markers(question: str) -> str:
    """RadBench marks image positions with "<i>" (e.g. "Compare the first study <i> <i> to
    the second study <i>"). All images are sent in reference order before the text, so each
    marker is replaced by its number: "[Image 1] [Image 2] … [Image 3]"."""
    count = [0]

    def _repl(_m):
        count[0] += 1
        return f"[Image {count[0]}]"
    return re.sub(r"<i>", _repl, question)


def _build_options_list(raw) -> list:
    """Normalise a raw choices/options field to [{'key': ..., 'value': ...}]."""
    if isinstance(raw, dict):
        return [{"key": k, "value": v} for k, v in raw.items()]
    if isinstance(raw, list) and raw:
        if isinstance(raw[0], str):
            return [{"key": k, "value": v} for k, v in zip(string.ascii_uppercase, raw)]
        return raw
    return []


# ---------------------------------------------------------------------------
# VQA-Med-2019  →  data/vqa_med_2019.parquet
# ---------------------------------------------------------------------------

def _answer_alternatives(raw) -> list:
    """
    All accepted answers of a VQA-Med-2019 row. The HF dataset stores a list
    (e.g. ['ct w/contrast', 'ct w/contrast iv']); some exports store a list-repr string.
    """
    if isinstance(raw, (list, tuple)):
        vals = [str(a).strip() for a in raw]
    else:
        text = str(raw if raw is not None else "").strip()
        if (text.startswith("['") and text.endswith("']")) or (text.startswith('["') and text.endswith('"]')):
            import ast
            try:
                vals = [str(a).strip() for a in ast.literal_eval(text)]
            except Exception:
                vals = [text[2:-2]]
        else:
            vals = [text]
    return [v for v in vals if v]


def _format_vqa_med_item(item: dict, idx: int) -> dict:
    question = (
        item.get("question")
        or item.get("Question")
        or item.get("q")
        or ""
    )
    raw_answer = item.get("answer")
    if raw_answer is None or (isinstance(raw_answer, str) and not raw_answer):
        raw_answer = item.get("Answer") or item.get("a") or item.get("gt") or ""
    answers = _answer_alternatives(raw_answer)

    image = item.get("image") or item.get("img") or None
    item_id = str(item.get("id") or item.get("qid") or item.get("image_name") or f"vqamed-{idx}")

    return {
        "id": item_id,
        "benchmark": "VQA-Med-2019",
        "question": str(question),
        "answer": answers[0] if answers else "",
        "answers": answers,            # all accepted alternatives (29+3 questions have >1)
        "image": image,
        "image_format": "png",
        "meta": {
            # HF column is "question_categories": modality | plane | organ | abnormality
            "category": str(item.get("question_categories") or item.get("category")
                            or item.get("Category") or ""),
            # one image per question in the 500-item test set; image file name as cluster id
            "cluster_id": str(item.get("_image_path") or item_id),
            "source": "VQA-Med-2019",
        },
    }


def load_vqa_med_2019(limit=None):
    """
    Loads VQA-Med-2019 from data/vqa_med_2019.parquet: the 500-question test set of the
    ImageCLEF 2019 VQA-Med task (Ben Abacha et al.), e.g. the test split of
    https://huggingface.co/datasets/simwit/vqa-med-2019 saved as parquet.
    """
    print("--- Loading VQA-Med-2019 ---")

    if not _VQA_MED_PATH.exists():
        raise FileNotFoundError(
            f"VQA-Med-2019 not found: {_VQA_MED_PATH}\n"
            "Download the test split of https://huggingface.co/datasets/simwit/vqa-med-2019\n"
            "and save it as data/vqa_med_2019.parquet (see README)."
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
        # metadata.question_id names the question template (9 values), not the item
        "id": f"{meta.get('question_id') or 'radimagenet'}-{idx}",
        "benchmark": "RadImageNet-VQA",
        "question": str(item.get("question") or ""),
        "answer": str(item.get("answer") or ""),
        "options": options,
        "image": item.get("image"),
        "image_format": "png",
        "meta": {
            "question_type": q_type,
            # anatomy | pathology | pathology_specific; question_template = one of 9 templates
            "category": str(meta.get("content_type") or ""),
            "question_template": str(meta.get("question_id") or ""),
            # 1000 images with 9 questions each: image file name is the cluster id
            "cluster_id": str(item.get("_image_path") or ""),
            "modality": str(meta.get("modality") or "").upper(),
            "pathology": str(meta.get("pathology") or ""),
            "location": str(meta.get("location") or ""),
            "source": "RadImageNet-VQA",
        },
    }


def load_radimagenet_vqa(limit=None):
    """
    Loads the RadImageNet-VQA benchmark test split (9K items, CT/MRI).

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
    print("--- Loading RadImageNet-VQA (CT/MRI benchmark split) ---")

    if not _RADIMAGENET_BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"RadImageNet-VQA benchmark split not found: {_RADIMAGENET_BENCHMARK_PATH}\n"
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

    print(f"  {len(items)} items loaded.")
    return [_format_radimagenet_benchmark_item(item, idx) for idx, item in enumerate(items)]


# ---------------------------------------------------------------------------
# RadBench (harrison.ai) – VLM benchmark with X-ray images
# ---------------------------------------------------------------------------

def _detect_radbench_qtype(item: dict) -> str:
    """
    RadBench stores the answer type in A_TYPE:
      - "CLOSED" → MCQ / yes-no with answer options (377 questions)
      - "OPEN"   → free-text answer, evaluated with exact match, WBSS and LLM-judge (120 questions)
    """
    a_type = str(item.get("A_TYPE") or item.get("a_type") or item.get("type") or "").lower()
    if "open" in a_type:
        return "open"
    # Check for presence of options as fallback
    has_options = bool(item.get("OPTIONS") or item.get("options") or item.get("choices"))
    if has_options:
        return "mcq"
    return "open"


# Image cache, filled by scripts/download_radbench_images.py. File name = sha1(reference)[:16]
# + extension, because the last URL segment is not unique (4 Radiopaedia URLs end in
# "0._jumbo.jpeg").
_RADBENCH_IMAGE_DIR = Path("data/radbench_images_v2")
_MEDPIX_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def radbench_image_refs(image_ids) -> list:
    """Split the imageIDs field (comma-separated URLs / MedPix UUIDs) into references."""
    if image_ids is None:
        return []
    text = str(image_ids).strip()
    if text.lower() in ("", "nan", "none"):
        return []
    return [r.strip() for r in text.split(",") if r.strip()]


def radbench_image_kind(ref: str) -> str:
    """'url' (downloadable), 'medpix' (UUID; MedPix is currently offline) or 'unresolvable'."""
    ref = ref.strip()
    if ref.lower().startswith(("http://", "https://")):
        return "url"
    if _MEDPIX_UUID_RE.match(ref):
        return "medpix"
    return "unresolvable"     # e.g. bare Radiopaedia image id "52662257" (case 77654)


def radbench_image_filename(ref: str) -> str:
    """Deterministic, collision-free cache file name: sha1(reference)[:16] + original extension."""
    ref = ref.strip()
    path = urlparse(ref).path if radbench_image_kind(ref) == "url" else ref
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    return hashlib.sha1(ref.encode("utf-8")).hexdigest()[:16] + ext


def radbench_image_path(ref: str) -> Path:
    return _RADBENCH_IMAGE_DIR / radbench_image_filename(ref)


def _load_radbench_images(refs: list) -> list:
    """
    Load PIL images for all references of a RadBench row from the local cache, in order.
    Raises FileNotFoundError if one is missing, so a question is never sent with only
    some of its images (load_radbench checks and reports missing files beforehand).
    """
    from PIL import Image as _PILImage

    images = []
    for ref in refs:
        path = radbench_image_path(ref)
        if not path.exists():
            raise FileNotFoundError(f"RadBench image missing: {path} (reference {ref!r}); "
                                    "run: python scripts/download_radbench_images.py")
        with _PILImage.open(path) as im:
            images.append(im.copy())
    return images


def _radbench_case_id(item: dict) -> str:
    raw = item.get("CASE_ID")
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    return str(raw).strip()


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
    # Some closed questions store the answer as a ranked list ("C2,C3,C1,...");
    # the first entry is the correct option.
    if q_type in ("mcq", "yes_no") and "," in answer:
        option_values = {o["value"].strip().lower() for o in options}
        first = answer.split(",")[0].strip()
        if answer.lower() not in option_values and first.lower() in option_values:
            answer = first

    # Use embedded image if present (parquet), otherwise load all images of the row
    image = item.get("image") or item.get("img") or None
    refs = radbench_image_refs(item.get("imageIDs"))
    images = [image] if image is not None else _load_radbench_images(refs)

    # Primary image for single-image tasks; all images passed in meta for multi-image
    primary_image = images[0] if images else None

    # CASE_ID is shared by all questions of a case; append the CSV row number.
    # (The id keeps the float formatting "77654.0-q220" so that ids stay stable across runs.)
    case_id = str(item.get("CASE_ID") or item.get("id") or item.get("qid") or "radbench")
    cluster = _radbench_case_id(item) or ("medpix:" + ",".join(sorted(refs)) if refs else case_id)
    return {
        "id": f"{case_id}-q{item.get('_row', idx)}",
        "benchmark": "RadBench",
        "question": _number_image_markers(str(item.get("QUESTION") or item.get("question") or "")),
        "answer": answer,
        "options": options,
        "image": primary_image,
        "image_format": "png",
        "meta": {
            "question_type": q_type,       # "mcq" | "yes_no" | "open"
            "category": str(item.get("Q_TYPE") or "").strip(),  # pathology, anatomy, view, …
            "q_type_category": str(item.get("Q_TYPE") or ""),
            "cluster_id": cluster,         # Radiopaedia case (questions of a case share images)
            "modality": str(item.get("modality") or "XR"),
            "organ": str(item.get("IMAGE_ORGAN") or ""),
            "source": str(item.get("imageSource") or "RadBench"),
            "image_refs": refs,
            "all_images": images,          # full list for multi-image questions
        },
    }


def _radbench_missing_policy(config) -> str:
    """task_settings.radbench.missing_images: "error" (default) | "skip_question"."""
    ts = ((config or {}).get("task_settings") or {}).get("radbench") or {}
    policy = str(ts.get("missing_images") or "error").strip().lower()
    if policy not in ("error", "skip_question"):
        raise ValueError(f"task_settings.radbench.missing_images must be 'error' or 'skip_question', got {policy!r}")
    return policy


def load_radbench(limit=None, config=None):
    """
    Loads the RadBench benchmark (harrison.ai).

    RadBench is a **VLM benchmark** using plain X-ray images (XR) from
    MedPix and Radiopaedia cases. It is NOT text-only.
      - 497 questions in data/radbench.csv: 212 MedPix, 285 Radiopaedia
      - Modality: X-ray (plain film), often several images per question

    Which questions are run (all counts are printed):
      1. MedPix questions are dropped unless their images are in the cache – MedPix is
         currently offline (212 questions).
      2. Questions with an image reference that is neither a URL nor a MedPix id are
         dropped unless that image was placed in the cache by hand: case 77654 lists the
         bare Radiopaedia image id "52662257" (2 questions, both "compare first/second study").
      3. Every other question needs all its images in data/radbench_images_v2/. A missing
         file raises an error (task_settings.radbench.missing_images: skip_question skips
         those questions with a warning instead). Questions are never sent with fewer images.

    Evaluation:
      - Closed-ended MCQ   → letter-accuracy (rule-based)
      - Closed-ended Yes/No → first-word accuracy
      - Open-ended          → exact match, WBSS, LLM-as-a-Judge

    Download: https://github.com/harrison-ai/radbench
    Place the file at: data/radbench.csv
    Download the X-ray images: python scripts/download_radbench_images.py
    """
    print("--- Loading RadBench (harrison.ai, X-ray VQA) ---")

    if not _RADBENCH_PATH.exists():
        raise FileNotFoundError(
            f"RadBench not found: {_RADBENCH_PATH}\n"
            "Download: git clone https://github.com/harrison-ai/radbench data/radbench_repo\n"
            "Then: cp data/radbench_repo/data/radbench/radbench.csv data/radbench.csv\n"
            "Images: python scripts/download_radbench_images.py"
        )
    policy = _radbench_missing_policy(config)

    items = _load_local_file(str(_RADBENCH_PATH))
    for row_no, it in enumerate(items):
        it["_row"] = row_no
    n_rows = len(items)

    def _qid(it):
        return f"{it.get('CASE_ID') or 'radbench'}-q{it['_row']}"

    def _cached(ref):
        return radbench_image_path(ref).exists()

    # 1. MedPix: images are not obtainable any more (kept only if cached locally)
    medpix = [it for it in items if str(it.get("imageSource") or "").strip().lower() == "medpix"]
    medpix_dropped = [it for it in medpix
                      if not any(_cached(r) for r in radbench_image_refs(it.get("imageIDs")))]
    dropped_ids = {id(it) for it in medpix_dropped}
    items = [it for it in items if id(it) not in dropped_ids]
    if medpix_dropped:
        print(f"  {len(medpix_dropped)} MedPix questions without images skipped "
              f"(MedPix is currently offline)")

    # 2. References that cannot be downloaded (not a URL, not MedPix) and were not supplied by hand
    unresolvable = [it for it in items
                    if any(radbench_image_kind(r) == "unresolvable" and not _cached(r)
                           for r in radbench_image_refs(it.get("imageIDs")))]
    if unresolvable:
        refs = sorted({r for it in unresolvable for r in radbench_image_refs(it.get("imageIDs"))
                       if radbench_image_kind(r) == "unresolvable"})
        print(f"  WARNING: {len(unresolvable)} questions dropped – image reference is not a URL "
              f"({', '.join(refs)}): {', '.join(_qid(it) for it in unresolvable)}. "
              f"Place the image at {_RADBENCH_IMAGE_DIR}/<sha1(ref)[:16]>.jpg to include them.")
        dropped_ids = {id(it) for it in unresolvable}
        items = [it for it in items if id(it) not in dropped_ids]

    # 3. All remaining images must be in the cache
    missing = {}
    for it in items:
        miss = [r for r in radbench_image_refs(it.get("imageIDs")) if not _cached(r)]
        if miss:
            missing[_qid(it)] = miss
    if missing:
        n_files = len({r for refs in missing.values() for r in refs})
        msg = (f"RadBench: {n_files} image(s) missing in {_RADBENCH_IMAGE_DIR}/ for "
               f"{len(missing)} question(s), e.g. {next(iter(missing.items()))}. "
               "Run: python scripts/download_radbench_images.py")
        if policy == "error":
            raise FileNotFoundError(msg + "  (or set task_settings.radbench.missing_images: skip_question)")
        print(f"  WARNING: {msg} – skipping these questions (missing_images: skip_question)")
        items = [it for it in items if _qid(it) not in missing]

    print(f"  {n_rows} rows → {len(items)} questions with all images")

    if limit:
        items = items[:limit]

    formatted = [_format_radbench_item(item, idx) for idx, item in enumerate(items)]
    n_imgs = sum(len(it["meta"]["all_images"]) for it in formatted)
    print(f"  {len(formatted)} questions, {n_imgs} images loaded from {_RADBENCH_IMAGE_DIR}/")
    return formatted

import io
from pathlib import Path


# ---------------------------------------------------------------------------
# Expected data file locations (place files here before running)
# ---------------------------------------------------------------------------

_MEDQA_PATH       = Path("data/medqa-test.parquet")
_RAR_PATH         = Path("data/RaR_dataset_WithAnswer.csv")
_RADIORAG_PATH    = Path("data/RadioRAG_WithOptions_WithAnswer.csv")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_local_parquet(path: str) -> list:
    """Load a local parquet file → list of row dicts via pyarrow (no HF caching)."""
    import pyarrow.parquet as pq

    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        from PIL import Image as _PILImage
        _has_pil = True
    except ImportError:
        _has_pil = False

    def _maybe_decode_image(val):
        if not _has_pil or val is None:
            return val
        try:
            if isinstance(val, dict):
                raw = val.get("bytes") or val.get("data")
            elif isinstance(val, (bytes, bytearray)):
                raw = val
            else:
                return val
            return _PILImage.open(io.BytesIO(raw)) if raw else None
        except Exception:
            return None

    items = []
    pf = pq.ParquetFile(str(parquet_path))
    for batch in pf.iter_batches(batch_size=1000):
        cols = {col: batch.column(col).to_pylist() for col in batch.schema.names}
        image_cols = {col for col in cols if col in ("image", "img")}
        for i in range(batch.num_rows):
            row = {}
            for col, vals in cols.items():
                v = vals[i]
                if col in image_cols:
                    # keep the original file name (HF Image feature: {"bytes", "path"}),
                    # used as image/cluster id by the vision loaders
                    if isinstance(v, dict) and v.get("path"):
                        row["_image_path"] = str(v["path"])
                    v = _maybe_decode_image(v)
                row[col] = v
            items.append(row)
    return items


def _load_local_csv(path: str) -> list:
    """Load a local CSV file → list of row dicts via pandas."""
    import pandas as pd
    df = pd.read_csv(path)
    return df.where(df.notna(), None).to_dict(orient="records")


def _load_local_file(path: str) -> list:
    """Dispatch to CSV or parquet loader based on file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() == ".csv":
        return _load_local_csv(path)
    return _load_local_parquet(path)


# ---------------------------------------------------------------------------
# MedQA (USMLE)  →  data/medqa-test.parquet
# ---------------------------------------------------------------------------

def _format_medqa_openlifescienceai(item: dict) -> dict:
    data = item.get("data") or {}
    options = data.get("Options") or {}
    if isinstance(options, dict):
        options = [{"key": key, "value": value} for key, value in options.items()]

    return {
        "id": item.get("id"),
        "benchmark": "MedQA",
        "question": data.get("Question"),
        "options": options,
        "correct_answer": data.get("Correct Option"),
        "meta": {
            "complexity": "high",
            "correct_answer_text": data.get("Correct Answer"),
            "subject_name": item.get("subject_name"),
            "source": "openlifescienceai/medqa",
        },
    }


def load_medqa(limit=None):
    """
    Loads MedQA (USMLE) from data/medqa-test.parquet.
    Download: https://huggingface.co/datasets/openlifescienceai/medqa
    """
    print("--- Loading MedQA (USMLE) ---")

    if not _MEDQA_PATH.exists():
        raise FileNotFoundError(
            f"MedQA not found: {_MEDQA_PATH}\n"
            "Download: https://huggingface.co/datasets/openlifescienceai/medqa\n"
            "Place the file at: data/medqa-test.parquet"
        )

    items = _load_local_parquet(str(_MEDQA_PATH))
    if limit:
        items = items[:limit]
    return [_format_medqa_openlifescienceai(item) for item in items]


# ---------------------------------------------------------------------------
# RaR (radiology Retrieval and Reasoning; board-exam questions)  →  data/rar-test.parquet
# ---------------------------------------------------------------------------

def _format_rar_item(item: dict, idx: int) -> dict:
    # Columns: question_number, question, option_A..option_E, solution_index
    options = []
    for key in ["A", "B", "C", "D", "E"]:
        val = item.get(f"option_{key}")
        if val is not None and str(val).strip():
            options.append({"key": key, "value": str(val).strip()})

    answer = str(item.get("solution_index") or "").strip().upper()

    return {
        "id": str(item.get("question_number") or f"rar-{idx}"),
        "benchmark": "RaR",
        "question": str(item.get("question") or ""),
        "options": options,
        "correct_answer": answer,
        "meta": {
            "source": "RaR",
        },
    }


def load_rar(limit=None):
    """
    Loads RaR (radiology Retrieval and Reasoning; board-exam questions) from data/RaR_dataset_WithAnswer.csv.
    Columns: question_number, question, option_A..option_E, solution_index
    Dataset not public — contact authors: https://www.nature.com/articles/s41746-025-02250-5
    """
    print("--- Loading RaR (board-exam questions from Wind et al. 2025) ---")

    if not _RAR_PATH.exists():
        raise FileNotFoundError(
            f"RaR not found: {_RAR_PATH}\n"
            "The 65 questions are in Supplementary Note 5 of Wind et al. 2025:\n"
            "https://doi.org/10.1038/s41746-025-02250-5\n"
            "Place the file at: data/RaR_dataset_WithAnswer.csv"
        )

    items = _load_local_csv(str(_RAR_PATH))
    if limit:
        items = items[:limit]
    return [_format_rar_item(item, idx) for idx, item in enumerate(items)]


# ---------------------------------------------------------------------------
# RadioRAG (MCQ)  →  data/RadioRAG_WithOptions_WithAnswer.csv
# ---------------------------------------------------------------------------

def _format_radiorag_item(item: dict, idx: int) -> dict:
    # Columns: q number, question, option 1..4, answer index (1-based)
    options = []
    for i, key in enumerate(["A", "B", "C", "D"], start=1):
        val = item.get(f"option {i}")
        if val is not None and str(val).strip():
            options.append({"key": key, "value": str(val).strip()})

    answer_idx = item.get("answer index")
    try:
        correct_answer = ["A", "B", "C", "D"][int(answer_idx) - 1]
    except (TypeError, ValueError, IndexError):
        correct_answer = ""

    return {
        "id": str(item.get("q number") or f"radiorag-{idx}"),
        "benchmark": "RadioRAG",
        "question": str(item.get("question") or ""),
        "options": options,
        "correct_answer": correct_answer,
        "meta": {
            "source": "RadioRAG",
        },
    }


def load_radiorag(limit=None):
    """
    Loads RadioRAG from data/RadioRAG_WithOptions_WithAnswer.csv.
    Columns: q number, question, option 1..4, answer index (1-based)
    Dataset not public — contact authors: https://github.com/tayebiarasteh/RadioRAG
    """
    print("--- Loading RadioRAG (4-option version) ---")

    if not _RADIORAG_PATH.exists():
        raise FileNotFoundError(
            f"RadioRAG not found: {_RADIORAG_PATH}\n"
            "The 4-option version (Wind et al. 2025) is available from the authors;\n"
            "the open-ended questions are in the appendix of https://doi.org/10.1148/ryai.240476\n"
            "Place the file at: data/RadioRAG_WithOptions_WithAnswer.csv"
        )

    items = _load_local_csv(str(_RADIORAG_PATH))
    if limit:
        items = items[:limit]
    return [_format_radiorag_item(item, idx) for idx, item in enumerate(items)]

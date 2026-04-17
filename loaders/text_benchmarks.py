import io
from pathlib import Path


# ---------------------------------------------------------------------------
# Expected data file locations (place files here before running)
# ---------------------------------------------------------------------------

_MEDQA_PATH       = Path("data/medqa-test.parquet")
_RAR_PATH         = Path("data/RaR_dataset_WithAnswer.csv")
_EXTRACTION_PATH  = Path("data/extraction.parquet")
_RADIORAG_PATH    = Path("data/RadioRAG_WithOptions_WithAnswer.csv")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_local_parquet(path: str) -> list:
    """Load a local parquet file → list of row dicts via pyarrow (no HF caching)."""
    import pyarrow.parquet as pq

    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

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
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
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
    print("--- Lade MedQA (USMLE) ---")

    if not _MEDQA_PATH.exists():
        raise FileNotFoundError(
            f"MedQA nicht gefunden: {_MEDQA_PATH}\n"
            "Download: https://huggingface.co/datasets/openlifescienceai/medqa\n"
            "Datei ablegen als: data/medqa-test.parquet"
        )

    items = _load_local_parquet(str(_MEDQA_PATH))
    if limit:
        items = items[:limit]
    return [_format_medqa_openlifescienceai(item) for item in items]


# ---------------------------------------------------------------------------
# RaR (Reasoning and Radiology)  →  data/rar-test.parquet
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
    Loads RaR (Reasoning and Radiology) from data/RaR_dataset_WithAnswer.csv.
    Columns: question_number, question, option_A..option_E, solution_index
    Dataset not public — contact authors: https://www.nature.com/articles/s41746-025-02250-5
    """
    print("--- Lade RaR (Reasoning and Radiology) ---")

    if not _RAR_PATH.exists():
        raise FileNotFoundError(
            f"RaR nicht gefunden: {_RAR_PATH}\n"
            "Dataset nicht öffentlich — Autoren kontaktieren:\n"
            "https://www.nature.com/articles/s41746-025-02250-5\n"
            "Datei ablegen als: data/RaR_dataset_WithAnswer.csv"
        )

    items = _load_local_csv(str(_RAR_PATH))
    if limit:
        items = items[:limit]
    return [_format_rar_item(item, idx) for idx, item in enumerate(items)]


# ---------------------------------------------------------------------------
# Label Extraction (NER)  →  data/extraction.parquet
# ---------------------------------------------------------------------------

def _format_extraction_item(item: dict, idx: int) -> dict:
    entities_raw = item.get("entities") or item.get("labels") or item.get("ner_tags") or []
    if isinstance(entities_raw, list):
        entities_str = ", ".join(str(e) for e in entities_raw if e and str(e) not in {"O", "0"})
    else:
        entities_str = str(entities_raw)

    text = (
        item.get("text")
        or item.get("report")
        or item.get("findings")
        or item.get("impression")
        or item.get("sentence")
        or ""
    )

    return {
        "id": str(item.get("id") or f"extraction-{idx}"),
        "benchmark": "LabelExtraction",
        "text": text,
        "entities": entities_str,
        "meta": {
            "source": item.get("source", ""),
            "category": item.get("category") or item.get("label_type") or "",
        },
    }


def load_label_extraction(limit=None):
    """
    Loads radiology NER data from data/extraction.parquet.
    Also accepts data/extraction.csv.

    Required columns: "text" (report text), "entities" (comma-separated reference entities)
    Recommended source: RadGraph (https://physionet.org/content/radgraph/)
    """
    print("--- Lade Label Extraction (Medizinische Entitäten aus Befundtexten) ---")

    path = _EXTRACTION_PATH
    if not path.exists():
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            path = csv_path
        else:
            raise FileNotFoundError(
                f"Label Extraction Dataset nicht gefunden: {_EXTRACTION_PATH}\n"
                "Empfohlene Quelle: RadGraph (https://physionet.org/content/radgraph/)\n"
                "Datei ablegen als: data/extraction.parquet  (oder .csv)\n"
                "Benötigte Spalten: 'text' (Befundtext), 'entities' (kommaseparierte Entitäten)"
            )

    raw_items = _load_local_file(str(path))

    text_items = [
        (idx, item) for idx, item in enumerate(raw_items)
        if str(item.get("text") or item.get("report") or item.get("findings")
           or item.get("impression") or item.get("sentence") or "").strip()
    ]

    if limit:
        text_items = text_items[:limit]

    return [_format_extraction_item(item, idx) for idx, item in text_items]


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
    print("--- Lade RadioRAG (Radiology MCQ) ---")

    if not _RADIORAG_PATH.exists():
        raise FileNotFoundError(
            f"RadioRAG nicht gefunden: {_RADIORAG_PATH}\n"
            "Dataset nicht öffentlich — Autoren kontaktieren:\n"
            "https://github.com/tayebiarasteh/RadioRAG\n"
            "Datei ablegen als: data/RadioRAG_WithOptions_WithAnswer.csv"
        )

    items = _load_local_csv(str(_RADIORAG_PATH))
    if limit:
        items = items[:limit]
    return [_format_radiorag_item(item, idx) for idx, item in enumerate(items)]

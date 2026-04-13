import os
import uuid
from pathlib import Path

def _ensure_hf_cache_dir() -> None:
    """
    Some environments have a read-only $HOME. If no Hugging Face cache env vars are
    set, default to a local, writable cache inside the repo.
    """
    if os.getenv("HF_HOME") or os.getenv("HF_DATASETS_CACHE") or os.getenv("HF_HUB_CACHE"):
        return
    cache_dir = Path(".cache/huggingface").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)

# Must run before importing `datasets` so it picks up the env var.
_ensure_hf_cache_dir()

from datasets import Dataset, load_dataset

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


def _load_local_parquet(path: str) -> Dataset:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Local MedQA parquet not found: {path}")
    return Dataset.from_parquet(str(parquet_path))


def load_medqa(limit=None):
    """
    Loads a MedQA test split in a stable way (no dataset scripts).

    Priority:
    1) Local parquet via env var MEDQA_PARQUET_PATH
    2) Hugging Face dataset openlifescienceai/medqa (parquet)
    """
    print("--- Lade MedQA (USMLE) ---")

    local_parquet_path = os.getenv("MEDQA_PARQUET_PATH")
    if not local_parquet_path:
        for candidate in (
            "data/medqa-test.parquet",
            "data/medqa_test.parquet",
            "data/test-00000-of-00001.parquet",
        ):
            if Path(candidate).exists():
                local_parquet_path = candidate
                break

    if local_parquet_path:
        dataset = _load_local_parquet(local_parquet_path)
        formatter = _format_medqa_openlifescienceai
    else:
        try:
            dataset = load_dataset("openlifescienceai/medqa", split="test")
            formatter = _format_medqa_openlifescienceai
        except Exception as e:
            raise RuntimeError(
                "MedQA konnte nicht geladen werden. Gründe sind meist (a) kein Internet/DNS in der Umgebung oder "
                "(b) ein Dataset, das ein Loading-Script benötigt (was in datasets>=4 nicht mehr unterstützt wird).\n\n"
                "Fix:\n"
                "1) MedQA Parquet-Datei(en) auf einem Rechner mit Internet herunterladen (z.B. von "
                "'openlifescienceai/medqa').\n"
                "2) Pfad setzen: export MEDQA_PARQUET_PATH=/pfad/zur/medqa-test.parquet\n"
                "3) Dann erneut: python main.py\n\n"
                f"Originalfehler: {type(e).__name__}: {e}"
            ) from e

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    formatted_data = []
    for item in dataset:
        formatted_data.append(formatter(item))

    return formatted_data


# ---------------------------------------------------------------------------
# RaR – Reasoning and Radiology
# ---------------------------------------------------------------------------

def _format_rar_item(item: dict, idx: int) -> dict:
    """Normalise a RaR row into the shared MCQ schema."""
    raw_options = item.get("options") or item.get("choices") or {}
    if isinstance(raw_options, dict):
        options = [{"key": k, "value": v} for k, v in raw_options.items()]
    elif isinstance(raw_options, list):
        # Already a list – could be strings ["opt A", ...] or dicts
        if raw_options and isinstance(raw_options[0], str):
            keys = ["A", "B", "C", "D", "E"]
            options = [{"key": keys[i], "value": v} for i, v in enumerate(raw_options)]
        else:
            options = raw_options
    else:
        options = []

    answer = str(item.get("answer") or item.get("correct_answer") or item.get("label") or "")
    if len(answer) > 1:
        answer = answer.strip()[0].upper()

    return {
        "id": str(item.get("id") or item.get("qid") or f"rar-{idx}"),
        "benchmark": "RaR",
        "question": item.get("question") or item.get("Question") or "",
        "options": options,
        "correct_answer": answer.upper(),
        "meta": {
            "source": item.get("source", "RaR"),
            "category": item.get("category") or item.get("type") or "",
        },
    }


def load_rar(limit=None):
    """
    Loads the RaR (Reasoning and Radiology) benchmark.

    Priority:
    1) Local parquet via env var RAR_PARQUET_PATH
    2) Auto-detect data/rar*.parquet
    3) Hugging Face dataset (MedRAG/RaR or similar)
    """
    print("--- Lade RaR (Reasoning and Radiology) ---")

    local_path = os.getenv("RAR_PARQUET_PATH")
    if not local_path:
        for candidate in (
            "data/rar-test.parquet",
            "data/rar_test.parquet",
            "data/rar.parquet",
        ):
            if Path(candidate).exists():
                local_path = candidate
                break

    if local_path:
        dataset = _load_local_parquet(local_path)
    else:
        try:
            dataset = load_dataset("MedRAG/RaR", split="test")
        except Exception:
            try:
                dataset = load_dataset("RaR-Medical/RaR", split="test")
            except Exception as e:
                raise RuntimeError(
                    "RaR konnte nicht geladen werden.\n\n"
                    "Fix:\n"
                    "1) Parquet-Datei bereitstellen und Pfad setzen:\n"
                    "   export RAR_PARQUET_PATH=/pfad/zur/rar-test.parquet\n"
                    f"Originalfehler: {type(e).__name__}: {e}"
                ) from e

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [_format_rar_item(item, idx) for idx, item in enumerate(dataset)]


# ---------------------------------------------------------------------------
# RadBench – Clinical Decision Making in Radiology
# ---------------------------------------------------------------------------

def _format_radbench_item(item: dict, idx: int) -> dict:
    """Normalise a RadBench row into the shared MCQ schema."""
    # RadBench (UCSC-VLAA) typically has fields: question, choices, gt, task_name
    choices_raw = item.get("choices") or item.get("options") or {}
    if isinstance(choices_raw, dict):
        options = [{"key": k, "value": v} for k, v in choices_raw.items()]
    elif isinstance(choices_raw, list):
        if choices_raw and isinstance(choices_raw[0], str):
            keys = ["A", "B", "C", "D", "E"]
            options = [{"key": keys[i], "value": v} for i, v in enumerate(choices_raw)]
        else:
            options = choices_raw
    else:
        options = []

    answer = str(
        item.get("gt") or item.get("answer") or item.get("correct_answer") or item.get("label") or ""
    ).strip()
    if len(answer) > 1:
        answer = answer[0].upper()

    return {
        "id": str(item.get("id") or item.get("qid") or f"radbench-{idx}"),
        "benchmark": "RadBench",
        "question": item.get("question") or item.get("Question") or "",
        "options": options,
        "correct_answer": answer.upper(),
        "meta": {
            "task_name": item.get("task_name") or item.get("task") or "",
            "source": "RadBench",
        },
    }


def load_radbench(limit=None):
    """
    Loads the RadBench benchmark (MCQ subset, closed-ended questions).

    RadBench (harrison.ai, https://harrison-ai.github.io/radbench/) contains
    497 questions total: 377 closed-ended MCQ + 120 open-ended.
    We load only the closed-ended MCQ subset for letter-accuracy evaluation.

    Priority:
    1) Local parquet via env var RADBENCH_PARQUET_PATH
    2) Auto-detect data/radbench*.parquet
    3) Hugging Face harrison-ai/radbench  (or github.com/harrison-ai/radbench)

    Note: harrison-ai RadBench is NOT the same as UCSC-VLAA/RadBench.
    """
    print("--- Lade RadBench (harrison.ai – Klinische Entscheidungsfindung Radiologie) ---")

    local_path = os.getenv("RADBENCH_PARQUET_PATH")
    if not local_path:
        for candidate in (
            "data/radbench-test.parquet",
            "data/radbench_test.parquet",
            "data/radbench.parquet",
        ):
            if Path(candidate).exists():
                local_path = candidate
                break

    if local_path:
        dataset = _load_local_parquet(local_path)
    else:
        hf_candidates = [
            ("harrison-ai/radbench", "test"),
            ("harrison-ai/radbench", "train"),
            ("harrison-ai/radbench", None),         # no split
        ]
        dataset = None
        last_err = None
        for hf_id, split in hf_candidates:
            try:
                if split:
                    dataset = load_dataset(hf_id, split=split)
                else:
                    ds = load_dataset(hf_id)
                    dataset = ds[list(ds.keys())[0]]
                print(f"  Geladen von HuggingFace: {hf_id}")
                break
            except Exception as e:
                last_err = e
        if dataset is None:
            raise RuntimeError(
                "RadBench (harrison.ai) konnte nicht geladen werden.\n\n"
                "Fix:\n"
                "1) Parquet-Datei von https://github.com/harrison-ai/radbench herunterladen.\n"
                "2) Pfad setzen:\n"
                "   export RADBENCH_PARQUET_PATH=/pfad/zur/radbench.parquet\n"
                "   (Format: Spalten 'question', 'choices'/'options', 'gt'/'answer')\n"
                f"Originalfehler: {type(last_err).__name__}: {last_err}"
            ) from last_err

    # Keep only closed-ended / MCQ items
    mcq_items = []
    for idx, item in enumerate(dataset):
        has_choices = bool(item.get("choices") or item.get("options"))
        has_answer_key = bool(item.get("gt") or item.get("answer") or item.get("correct_answer"))
        if has_choices and has_answer_key:
            mcq_items.append((idx, item))

    if not mcq_items:
        mcq_items = list(enumerate(dataset))

    if limit:
        mcq_items = mcq_items[:limit]

    return [_format_radbench_item(item, idx) for idx, item in mcq_items]


# ---------------------------------------------------------------------------
# Label Extraction – NER aus Radiologiebefunden
# ---------------------------------------------------------------------------

def _format_extraction_item(item: dict, idx: int) -> dict:
    """Normalise a radiology report item into the extraction schema."""
    entities_raw = item.get("entities") or item.get("labels") or item.get("ner_tags") or []
    if isinstance(entities_raw, list):
        # Could be token-level BIO tags; extract unique entity strings if available
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
    Loads a radiology NER dataset for the Label Extraction task.

    Priority:
    1) Local parquet via env var EXTRACTION_PARQUET_PATH
    2) Auto-detect data/extraction*.parquet or data/ner*.parquet
    3) Hugging Face n2c2_2018_track2 (radiology NER) or fallback
    """
    print("--- Lade Label Extraction (Medizinische Entitäten aus Befundtexten) ---")

    local_path = os.getenv("EXTRACTION_PARQUET_PATH")
    if not local_path:
        for candidate in (
            "data/extraction-test.parquet",
            "data/extraction_test.parquet",
            "data/ner-test.parquet",
            "data/ner_test.parquet",
        ):
            if Path(candidate).exists():
                local_path = candidate
                break

    if local_path:
        dataset = _load_local_parquet(local_path)
    else:
        try:
            # radiology_report_section_header is a radiology text classification dataset
            # with structured report sections – useful for extraction
            dataset = load_dataset(
                "StanfordAIMI/radiology_report_section_header", split="test"
            )
        except Exception:
            try:
                # Fallback: use RadBench report texts and derive entity extraction
                dataset = load_dataset("UCSC-VLAA/RadBench", split="test")
            except Exception as e:
                raise RuntimeError(
                    "Label-Extraction Datensatz konnte nicht geladen werden.\n\n"
                    "Fix:\n"
                    "1) Parquet-Datei bereitstellen und Pfad setzen:\n"
                    "   export EXTRACTION_PARQUET_PATH=/pfad/zur/extraction.parquet\n"
                    "   (Format: Spalten 'text'/'report'/'findings' + 'entities'/'labels')\n"
                    f"Originalfehler: {type(e).__name__}: {e}"
                ) from e

    # Filter for items that have actual report text
    text_items = []
    for idx, item in enumerate(dataset):
        text = (
            item.get("text")
            or item.get("report")
            or item.get("findings")
            or item.get("impression")
            or item.get("sentence")
            or ""
        )
        if text.strip():
            text_items.append((idx, item))

    if limit:
        text_items = text_items[:limit]

    return [_format_extraction_item(item, idx) for idx, item in text_items]


# ---------------------------------------------------------------------------
# RadioRAG – Open-ended Radiology QA
# ---------------------------------------------------------------------------

def _format_radiorag_item(item: dict, idx: int) -> dict:
    """Normalise a RadioRAG row into the open-ended QA schema."""
    question = (
        item.get("question")
        or item.get("Question")
        or item.get("prompt")
        or ""
    )
    reference = (
        item.get("answer")
        or item.get("reference_answer")
        or item.get("gt")
        or item.get("Answer")
        or ""
    )
    return {
        "id": str(item.get("id") or item.get("question_id") or f"radiorag-{idx}"),
        "benchmark": "RadioRAG",
        "question": str(question),
        "reference_answer": str(reference),
        "meta": {
            "subspecialty": item.get("subspecialty") or item.get("category") or "",
            "source": item.get("source") or "RadioRAG",
        },
    }


def load_radiorag(limit=None):
    """
    Loads the RadioRAG benchmark (open-ended radiology QA).

    RadioRAG (Tayebi Arasteh et al., 2024/2025) contains 104 open-ended
    radiology questions across 18 subspecialties:
      - 80 questions from RSNA Case Collection (RSNA-RadioQA)
      - 24 expert-curated questions (ExtendedQA)

    Evaluation: Binary LLM-as-a-Judge (correct / incorrect).
    Human expert baseline: ~63% accuracy.

    Paper: https://arxiv.org/abs/2407.15621
    GitHub: https://github.com/tayebiarasteh/RadioRAG

    Priority:
    1) Local parquet/JSON via env var RADIORAG_PARQUET_PATH
    2) Auto-detect data/radiorag*.parquet  or  data/radiorag*.json
    3) Dataset is NOT on HuggingFace — a local file is required.
    """
    print("--- Lade RadioRAG (Open-ended Radiology QA) ---")

    local_path = os.getenv("RADIORAG_PARQUET_PATH")
    if not local_path:
        for candidate in (
            "data/radiorag.parquet",
            "data/radiorag-test.parquet",
            "data/radiorag_test.parquet",
            "data/radiorag.json",
            "data/RadioRAG.json",
        ):
            if Path(candidate).exists():
                local_path = candidate
                break

    if local_path is None:
        raise RuntimeError(
            "RadioRAG konnte nicht geladen werden.\n\n"
            "Das Dataset ist nicht auf HuggingFace. Bitte:\n"
            "1) Daten von https://github.com/tayebiarasteh/RadioRAG herunterladen.\n"
            "2) Als Parquet oder JSON ablegen und Pfad setzen:\n"
            "   export RADIORAG_PARQUET_PATH=/pfad/zur/radiorag.parquet\n"
            "   (Format: Spalten 'question', 'answer'/'reference_answer',\n"
            "    optional: 'id', 'subspecialty')\n"
        )

    path = Path(local_path)
    if path.suffix.lower() == ".json":
        import json as _json
        with open(path, encoding="utf-8") as fh:
            raw = _json.load(fh)
        # Accept list of dicts or dict-of-dicts
        if isinstance(raw, dict):
            rows = list(raw.values())
        else:
            rows = raw
        from datasets import Dataset as _Dataset
        dataset = _Dataset.from_list(rows)
    else:
        dataset = _load_local_parquet(local_path)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [_format_radiorag_item(item, idx) for idx, item in enumerate(dataset)]

import os
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

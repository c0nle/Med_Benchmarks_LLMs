"""
Label Extraction Task Runner (benchmark `label_extraction`, needs data/extraction.parquet).

The model receives a radiology report text (Befundtext) and must extract
all medical entities (findings, diagnoses, anatomical structures, pathologies)
as a comma-separated list.

Results CSV columns: id, benchmark, text, reference_entities, model_entities,
finish_reason, completion_tokens
Evaluation: entity-string micro-F1 – TP/FP/FN of normalised entity strings summed over
all texts (evaluate.py). This is not the RadGraph entity/relation protocol.
"""
from tasks.mcq import META_FIELDS, run_items

FIELDNAMES = ["id", "benchmark", "text", "reference_entities", "model_entities"] + META_FIELDS


def build_prompt(item: dict) -> str:
    return (
        f"Radiologischer Befundtext:\n{item['text']}\n\n"
        "Extrahiere alle medizinischen Entitäten aus dem Text "
        "(Befunde, Diagnosen, Anatomie, Pathologien, Modalitäten). "
        "Gib ausschließlich eine kommaseparierte Liste der Entitäten zurück. "
        "Keine Erklärungen, keine Nummerierung."
    )


def run(config: dict, client, data: list, results_path: str, logger=None) -> str:
    """
    Run the label extraction benchmark and write incremental results
    (concurrency/resume: tasks.mcq.run_items). Returns the path to the written CSV.
    """
    def _base_row(item):
        return {
            "benchmark": item.get("benchmark", ""),
            "text": item.get("text", ""),
            "reference_entities": item.get("entities", ""),
        }

    # run_items stores the answer as model_answer; this benchmark calls it model_entities
    def _ask(item):
        return client.ask_question(build_prompt(item))

    return run_items(config, client, data, results_path, FIELDNAMES,
                     ask=_ask, base_row=_base_row, logger=logger, label="texts",
                     answer_field="model_entities")

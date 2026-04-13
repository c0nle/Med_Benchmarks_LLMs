"""
Medical Benchmark Runner

Supports running one or multiple benchmarks in a single call.

config.yaml options:
  benchmark: medqa                        # single benchmark
  benchmark: [medqa, rar, radbench]       # list
  benchmark: all                          # run every registered benchmark
"""
import yaml
import os
import importlib

from core.client import MedicalLLMClient

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------
# Maps benchmark name → (loader_callable, task_module_path, eval_type)
# eval_type: "mcq" | "vqa" | "extraction"

def _registry():
    from loaders.text_benchmarks import (
        load_medqa,
        load_rar,
        load_radbench,
        load_label_extraction,
        load_radiorag,
    )
    from loaders.vision_benchmarks import (
        load_vqa_med_2019,
        load_radimagenet_vqa,
    )
    return {
        "medqa":            (load_medqa,            "tasks.mcq",        "mcq"),
        "rar":              (load_rar,               "tasks.mcq",        "mcq"),
        "radbench":         (load_radbench,          "tasks.mcq",        "mcq"),
        "vqa_med_2019":     (load_vqa_med_2019,      "tasks.vqa",        "vqa"),
        "radimagenet_vqa":  (load_radimagenet_vqa,   "tasks.vqa",        "vqa"),
        "label_extraction": (load_label_extraction,  "tasks.extraction", "extraction"),
        "radiorag":         (load_radiorag,           "tasks.open_qa",    "open_qa"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_benchmarks(config: dict, registry: dict) -> list:
    """
    Parse the 'benchmark' config key.

    Accepted forms:
      benchmark: medqa                          → ["medqa"]
      benchmark: [medqa, rar, radbench]         → ["medqa", "rar", "radbench"]
      benchmark: all                            → all registered benchmarks (sorted)
    """
    raw = config.get("benchmark", "medqa")

    if isinstance(raw, list):
        names = [str(b).strip().lower() for b in raw]
    elif str(raw).strip().lower() == "all":
        names = sorted(registry.keys())
    else:
        names = [str(raw).strip().lower()]

    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(
            f"Unbekannte Benchmark(s): {unknown}. "
            f"Verfügbar: {', '.join(sorted(registry.keys()))}"
        )
    return names


def _run_one(benchmark: str, registry: dict, config: dict, client) -> None:
    """Load data, run the task, and evaluate for a single benchmark."""
    loader, task_module_path, eval_type = registry[benchmark]

    print(f"\n{'='*60}")
    print(f"  Benchmark: {benchmark.upper()}")
    print(f"{'='*60}")

    limit = config.get("benchmark_settings", {}).get("limit_samples", None)
    data = loader(limit=limit)

    os.makedirs("results", exist_ok=True)
    results_path = f"results/{benchmark}_results.csv"
    report_path  = f"results/{benchmark}_report.jsonl"

    task = importlib.import_module(task_module_path)
    task.run(config, client, data, results_path)

    _evaluate(benchmark, eval_type, results_path, report_path, config)


def _evaluate(benchmark: str, eval_type: str, results_path: str, report_path: str, config: dict) -> None:
    """Run the appropriate evaluation and print a terminal report."""
    try:
        if eval_type == "mcq":
            from evaluate import write_report_jsonl, print_terminal_report
            report = write_report_jsonl(results_path, out_path=report_path)
            print_terminal_report(results_path)
            print(f"Auswertung gespeichert: {report['path']}")

        elif eval_type == "vqa":
            from evaluate import write_vqa_report_jsonl, print_vqa_terminal_report
            report = write_vqa_report_jsonl(results_path, out_path=report_path)
            print_vqa_terminal_report(results_path)
            print(f"Auswertung gespeichert: {report_path}")
            print("Tipp: LLM-as-a-Judge für open-ended Fragen nachträglich:")
            print(f"  python evaluate.py {results_path} --type vqa --judge")

        elif eval_type == "extraction":
            from evaluate import write_extraction_report_jsonl, print_extraction_terminal_report
            report = write_extraction_report_jsonl(results_path, out_path=report_path)
            print_extraction_terminal_report(results_path)
            print(f"Auswertung gespeichert: {report_path}")

        elif eval_type == "open_qa":
            from evaluate import write_open_qa_report_jsonl, print_open_qa_terminal_report
            # Automatic metrics only; LLM-as-a-Judge requires --judge flag
            report = write_open_qa_report_jsonl(results_path, out_path=report_path)
            print_open_qa_terminal_report(results_path)
            print(f"Auswertung gespeichert: {report_path}")
            print("Tipp: LLM-as-a-Judge (primäre Metrik laut Paper) nachträglich:")
            print(f"  python evaluate.py {results_path} --type open_qa --judge")

    except Exception as e:
        print(f"Warnung: Auswertung für '{benchmark}' fehlgeschlagen: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config_path = "config.yaml" if os.path.exists("config.yaml") else "config.default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config_path != "config.yaml":
        print("Hinweis: config.yaml nicht gefunden, nutze config.default.yaml. "
              "Lege eine eigene config.yaml an (wird via .gitignore ignoriert).")

    registry = _registry()
    benchmarks = _resolve_benchmarks(config, registry)

    if len(benchmarks) > 1:
        print(f"Multi-Benchmark Modus: {benchmarks}")

    client = MedicalLLMClient(config)

    summary = []
    for benchmark in benchmarks:
        try:
            _run_one(benchmark, registry, config, client)
            summary.append((benchmark, "OK"))
        except Exception as e:
            print(f"\nFehler bei Benchmark '{benchmark}': {e}")
            summary.append((benchmark, f"FEHLER: {e}"))

    if len(benchmarks) > 1:
        print(f"\n{'='*60}")
        print("  Zusammenfassung")
        print(f"{'='*60}")
        for name, status in summary:
            print(f"  {name:<20} {status}")


if __name__ == "__main__":
    main()

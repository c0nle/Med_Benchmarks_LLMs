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
import datetime

from core.client import MedicalLLMClient
from core.logger import RunLogger

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------
# Maps benchmark name → (loader_callable, task_module_path, eval_type)
# eval_type: "mcq" | "vqa" | "extraction"

def _registry():
    from loaders.text_benchmarks import (
        load_medqa,
        load_rar,
        load_label_extraction,
        load_radiorag,
    )
    from loaders.vision_benchmarks import (
        load_radbench,
        load_vqa_med_2019,
        load_radimagenet_vqa,
    )
    return {
        # Text-only MCQ
        "medqa":            (load_medqa,            "tasks.mcq",        "mcq"),
        "rar":              (load_rar,               "tasks.mcq",        "mcq"),
        # VLM (image + text)
        "radbench":         (load_radbench,          "tasks.vqa",        "vqa"),
        "vqa_med_2019":     (load_vqa_med_2019,      "tasks.vqa",        "vqa"),
        "radimagenet_vqa":  (load_radimagenet_vqa,   "tasks.vqa",        "vqa"),
        # Text-only specialised
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


def _build_judge_client(config: dict):
    """Return a MedicalLLMClient for the judge model, or None if not configured."""
    judge_cfg = config.get("judge")
    if not judge_cfg:
        return None
    # Build a minimal config dict for the judge, reusing top-level defaults where missing
    merged = {
        "server": judge_cfg,
        "benchmark_settings": config.get("benchmark_settings", {}),
    }
    try:
        return MedicalLLMClient(merged)
    except Exception as e:
        print(f"  Warning: could not create judge client: {e}")
        return None


def _run_one(benchmark: str, registry: dict, config: dict, client, judge_client, logger=None) -> dict:
    """Load data, run the task, and evaluate for a single benchmark. Returns metrics dict."""
    loader, task_module_path, eval_type = registry[benchmark]

    print(f"\n--- {benchmark.upper()} ---")

    limit = config.get("benchmark_settings", {}).get("limit_samples", None)
    data = loader(limit=limit)

    os.makedirs("results", exist_ok=True)
    results_path = f"results/{benchmark}_results.csv"
    report_path  = f"results/{benchmark}_report.jsonl"

    task = importlib.import_module(task_module_path)
    task.run(config, client, data, results_path, logger=logger)

    return _evaluate(benchmark, eval_type, results_path, report_path, judge_client, logger=logger)


def _evaluate(benchmark: str, eval_type: str, results_path: str, report_path: str, judge_client, logger=None) -> dict:
    """Run the appropriate evaluation and return a metrics dict."""
    run_judge = judge_client is not None
    try:
        if eval_type == "mcq":
            from evaluate import write_report_jsonl, print_terminal_report
            report = write_report_jsonl(results_path, out_path=report_path, logger=logger)
            print_terminal_report(results_path)
            return {"accuracy_pct": report.get("accuracy_pct")}

        elif eval_type == "vqa":
            from evaluate import write_vqa_report_jsonl, print_vqa_terminal_report
            report = write_vqa_report_jsonl(results_path, out_path=report_path, client=judge_client, run_judge=run_judge, logger=logger)
            print_vqa_terminal_report(results_path, report=report)
            return {k: v for k, v in report.items() if k != "path"}

        elif eval_type == "extraction":
            from evaluate import write_extraction_report_jsonl, print_extraction_terminal_report
            report = write_extraction_report_jsonl(results_path, out_path=report_path, logger=logger)
            print_extraction_terminal_report(results_path)
            return {"micro_f1_pct": report.get("micro_f1_pct")}

        elif eval_type == "open_qa":
            from evaluate import write_open_qa_report_jsonl, print_open_qa_terminal_report
            report = write_open_qa_report_jsonl(results_path, out_path=report_path, client=judge_client, run_judge=run_judge, logger=logger)
            print_open_qa_terminal_report(results_path, report=report)
            return {k: v for k, v in report.items() if k != "path"}

    except Exception as e:
        print(f"  Warning: evaluation failed: {e}")
    return {}


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

    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/run_{ts}.log"

    with RunLogger(log_path) as logger:
        print(f"Log: {log_path}")

        client = MedicalLLMClient(config)
        judge_client = _build_judge_client(config)
        if judge_client:
            print(f"Judge model: {judge_client.model}")
        else:
            print("No judge model configured — LLM-as-a-Judge will be skipped.")
            print("To enable: add a 'judge:' section to config.yaml")

        summary = []
        for benchmark in benchmarks:
            try:
                metrics = _run_one(benchmark, registry, config, client, judge_client, logger=logger)
                summary.append((benchmark, metrics, None))
            except Exception as e:
                print(f"\nError in benchmark '{benchmark}': {e}")
                summary.append((benchmark, {}, str(e)))

        print(f"\n{'='*60}")
        print("  RESULTS")
        print(f"{'='*60}")
        for name, metrics, err in summary:
            if err:
                print(f"  {name:<22} ERROR: {err}")
            else:
                metric_str = "  ".join(f"{k}: {v:.2f}%" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
                print(f"  {name:<22} {metric_str if metric_str else 'no metrics'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

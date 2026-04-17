"""
Medical Benchmark Runner

Supports running one or multiple benchmarks in a single call.

config.yaml options:
  benchmark: medqa                        # single benchmark
  benchmark: [medqa, rar, radbench]       # list
  benchmark: all                          # run every registered benchmark

Output layout:
  results/
    run_<timestamp>/          ← per-run subfolder: CSVs + JSONL reports
    run_<timestamp>.log       ← full run log
    benchmark_results_<model>.png  ← bar chart (overwritten per model)
"""
import yaml
import os
import re
import importlib
import datetime

from core.client import MedicalLLMClient
from core.logger import RunLogger

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

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
        "medqa":            (load_medqa,            "tasks.mcq",        "mcq"),
        "rar":              (load_rar,               "tasks.mcq",        "mcq"),
        "radbench":         (load_radbench,          "tasks.vqa",        "vqa"),
        "vqa_med_2019":     (load_vqa_med_2019,      "tasks.vqa",        "vqa"),
        "radimagenet_vqa":  (load_radimagenet_vqa,   "tasks.vqa",        "vqa"),
        "label_extraction": (load_label_extraction,  "tasks.extraction", "extraction"),
        "radiorag":         (load_radiorag,           "tasks.mcq",        "mcq"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_benchmarks(config: dict, registry: dict) -> list:
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


def _sanitize_filename(name: str) -> str:
    """Replace characters that are invalid in filenames with underscores."""
    return re.sub(r"[^\w\-.]", "_", name)


def _build_judge_client(config: dict):
    judge_cfg = config.get("judge")
    if not judge_cfg:
        return None
    merged = {
        "server": judge_cfg,
        "benchmark_settings": config.get("benchmark_settings", {}),
    }
    try:
        return MedicalLLMClient(merged)
    except Exception as e:
        print(f"  Warning: could not create judge client: {e}")
        return None


def _run_one(benchmark: str, registry: dict, config: dict, client, judge_client,
             run_dir: str, logger=None) -> dict:
    """Load data, run the task, and evaluate for a single benchmark."""
    loader, task_module_path, eval_type = registry[benchmark]

    print(f"\n--- {benchmark.upper()} ---")

    limit = config.get("benchmark_settings", {}).get("limit_samples", None)
    data = loader(limit=limit)

    results_path = os.path.join(run_dir, f"{benchmark}_results.csv")
    report_path  = os.path.join(run_dir, f"{benchmark}_report.jsonl")

    task = importlib.import_module(task_module_path)
    task.run(config, client, data, results_path, logger=logger)

    return _evaluate(benchmark, eval_type, results_path, report_path, judge_client, logger=logger)


def _evaluate(benchmark: str, eval_type: str, results_path: str, report_path: str,
              judge_client, logger=None) -> dict:
    run_judge = judge_client is not None
    try:
        if eval_type == "mcq":
            from evaluate import write_report_jsonl, print_terminal_report
            report = write_report_jsonl(results_path, out_path=report_path, logger=logger)
            print_terminal_report(results_path)
            return {"accuracy_pct": report.get("accuracy_pct")}

        elif eval_type == "vqa":
            from evaluate import write_vqa_report_jsonl, print_vqa_terminal_report
            report = write_vqa_report_jsonl(results_path, out_path=report_path,
                                            client=judge_client, run_judge=run_judge, logger=logger)
            print_vqa_terminal_report(results_path, report=report)
            return {k: v for k, v in report.items() if k != "path"}

        elif eval_type == "extraction":
            from evaluate import write_extraction_report_jsonl, print_extraction_terminal_report
            report = write_extraction_report_jsonl(results_path, out_path=report_path, logger=logger)
            print_extraction_terminal_report(results_path)
            return {"micro_f1_pct": report.get("micro_f1_pct")}

        elif eval_type == "open_qa":
            from evaluate import write_open_qa_report_jsonl, print_open_qa_terminal_report
            report = write_open_qa_report_jsonl(results_path, out_path=report_path,
                                                client=judge_client, run_judge=run_judge, logger=logger)
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
        print("Hinweis: config.yaml nicht gefunden, nutze config.default.yaml.")

    registry = _registry()
    benchmarks = _resolve_benchmarks(config, registry)
    model_name = config.get("server", {}).get("model_name", "unknown")

    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-run subfolder for CSVs and JSONL reports
    run_dir = os.path.join("results", f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Log stays in results/ directly
    log_path = os.path.join("results", f"run_{ts}.log")

    with RunLogger(log_path) as logger:
        print(f"Run dir: {run_dir}")
        print(f"Log:     {log_path}")

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
                metrics = _run_one(benchmark, registry, config, client, judge_client,
                                   run_dir=run_dir, logger=logger)
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
                metric_str = "  ".join(
                    f"{k}: {v:.2f}%" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in metrics.items()
                )
                print(f"  {name:<22} {metric_str if metric_str else 'no metrics'}")
        print(f"{'='*60}")

        # Bar chart — stays in results/, named after model
        try:
            from core.plot import generate_results_chart
            chart_name = f"benchmark_results_{_sanitize_filename(model_name)}.png"
            chart_path = os.path.join("results", chart_name)
            generate_results_chart(summary, model_name, chart_path)
        except Exception as e:
            print(f"  Chart generation failed: {e}")


if __name__ == "__main__":
    main()

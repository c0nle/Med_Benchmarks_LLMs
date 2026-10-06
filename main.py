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
    from loaders.mamma_extraction import load_mamma_extraction
    from loaders.arm_extraction    import load_arm_extraction
    return {
        "medqa":                  (load_medqa,             "tasks.mcq",               "mcq"),
        "rar":                    (load_rar,                "tasks.mcq",               "mcq"),
        "radbench":               (load_radbench,           "tasks.vqa",               "vqa"),
        "vqa_med_2019":           (load_vqa_med_2019,       "tasks.vqa",               "vqa"),
        "radimagenet_vqa":        (load_radimagenet_vqa,    "tasks.vqa",               "vqa"),
        "label_extraction":       (load_label_extraction,   "tasks.extraction",        "extraction"),
        "radiorag":               (load_radiorag,            "tasks.mcq",               "mcq"),
        "label_extraction_mamma": (load_mamma_extraction,   "tasks.mamma_extraction",  "mamma_extraction"),
        "label_extraction_arm":   (load_arm_extraction,     "tasks.arm_extraction",    "arm_extraction"),
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

    # Loaders, die config kennen, bekommen sie übergeben (per inspect)
    import inspect
    sig = inspect.signature(loader)
    if "config" in sig.parameters:
        data = loader(limit=limit, config=config)
    else:
        data = loader(limit=limit)

    ids = [str(item.get("id")) for item in data]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{benchmark}: item ids are not unique ({len(set(ids))} unique of {len(ids)})")

    results_path = os.path.join(run_dir, f"{benchmark}_results.csv")
    report_path  = os.path.join(run_dir, f"{benchmark}_report.jsonl")
    _drop_error_rows(results_path)

    # Per-Task max_tokens-Override
    task_settings    = config.get("task_settings", {}).get(benchmark, {})
    orig_max_tokens  = client.max_tokens
    if "max_tokens" in task_settings:
        client.max_tokens = int(task_settings["max_tokens"])

    try:
        task = importlib.import_module(task_module_path)
        task.run(config, client, data, results_path, logger=logger)
    finally:
        client.max_tokens = orig_max_tokens

    n_done, n_errors = _count_rows(results_path)
    if n_done < len(data):
        print(f"  WARNING: {benchmark} incomplete – {n_done}/{len(data)} items in results "
              f"(resume with --run-dir {run_dir})")
    if n_errors:
        print(f"  WARNING: {n_errors} API errors in results (scored as wrong; "
              f"rerun with --run-dir {run_dir} to retry them)")

    metrics = _evaluate(benchmark, eval_type, results_path, report_path, judge_client,
                        config=config, logger=logger)
    return {"n_items": n_done, "n_expected": len(data), "n_api_errors": n_errors, **metrics}


def _error_mask(df):
    import pandas as pd
    mask = pd.Series(False, index=df.index)
    for col in ("model_answer", "model_raw"):
        if col in df.columns:
            mask |= df[col].fillna("").astype(str).str.startswith("Error:")
    return mask


def _drop_error_rows(results_path: str) -> None:
    """On resume, remove rows whose model call failed so they are asked again."""
    import pandas as pd
    if not (os.path.exists(results_path) and os.path.getsize(results_path) > 0):
        return
    df = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    mask = _error_mask(df)
    if mask.any():
        df[~mask].to_csv(results_path, index=False)
        print(f"  Resume: {int(mask.sum())} failed rows removed, will be retried")


def _count_rows(results_path: str):
    import pandas as pd
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        return 0, 0
    df = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    return len(df), int(_error_mask(df).sum())


def _evaluate(benchmark: str, eval_type: str, results_path: str, report_path: str,
              judge_client, config: dict = None, logger=None) -> dict:
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

        elif eval_type == "mamma_extraction":
            from evaluate import write_mamma_extraction_report_jsonl, print_mamma_extraction_terminal_report
            report = write_mamma_extraction_report_jsonl(
                results_path, out_path=report_path, config=config, logger=logger
            )
            print_mamma_extraction_terminal_report(results_path, report=report)
            return {k: v for k, v in report.items()
                    if k != "path" and isinstance(v, (int, float))}

        elif eval_type == "arm_extraction":
            from evaluate import write_arm_extraction_report_jsonl, print_arm_extraction_terminal_report
            report = write_arm_extraction_report_jsonl(
                results_path, out_path=report_path, logger=logger
            )
            print_arm_extraction_terminal_report(results_path, report=report)
            return {k: v for k, v in report.items()
                    if k != "path" and isinstance(v, (int, float))}

    except Exception as e:
        raise RuntimeError(f"evaluation failed ({results_path}): {e}") from e
    return {}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run medical LLM benchmarks (settings in config.yaml).")
    parser.add_argument("--run-dir", help="Existing or new run directory; an existing one is resumed "
                                          "(finished items are skipped, failed ones retried).")
    parser.add_argument("--benchmark", help="Comma-separated benchmarks, overrides config 'benchmark'.")
    parser.add_argument("--limit", help="Override benchmark_settings.limit_samples (number or 'all').")
    return parser.parse_args()


def _git_commit() -> str:
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"]) != 0
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _write_run_info(run_dir: str, config: dict, benchmarks: list) -> None:
    """Config snapshot (without API keys), benchmarks and code version for reproducibility."""
    import copy
    import json
    cfg = copy.deepcopy(config)
    for section in ("server", "judge"):
        if isinstance(cfg.get(section), dict):
            cfg[section].pop("api_key", None)
    info = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "benchmarks": benchmarks,
        "config": cfg,
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(run_dir, f"run_info_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    args = _parse_args()
    config_path = "config.yaml" if os.path.exists("config.yaml") else "config.default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config_path != "config.yaml":
        print("Hinweis: config.yaml nicht gefunden, nutze config.default.yaml.")
    if args.benchmark:
        config["benchmark"] = [b.strip() for b in args.benchmark.split(",") if b.strip()]
    if args.limit is not None:
        config.setdefault("benchmark_settings", {})["limit_samples"] = (
            None if args.limit.lower() in ("all", "none", "null") else int(args.limit)
        )

    registry = _registry()
    benchmarks = _resolve_benchmarks(config, registry)
    model_name = config.get("server", {}).get("model_name", "unknown")

    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-run subfolder for CSVs, JSONL reports, logs and run info
    run_dir = args.run_dir or os.path.join("results", f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"run_{ts}.log")
    _write_run_info(run_dir, config, benchmarks)

    with RunLogger(log_path) as logger:
        print(f"Run dir: {run_dir}")
        print(f"Log:     {log_path}")
        print(f"Limit:   {config.get('benchmark_settings', {}).get('limit_samples')}")

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
                import traceback
                print(f"\nError in benchmark '{benchmark}': {e}")
                traceback.print_exc()
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

        # Bar chart in the run directory
        try:
            from core.plot import generate_results_chart
            chart_name = f"benchmark_results_{_sanitize_filename(model_name)}.png"
            generate_results_chart(summary, model_name, os.path.join(run_dir, chart_name))
        except Exception as e:
            print(f"  Chart generation failed: {e}")


if __name__ == "__main__":
    main()

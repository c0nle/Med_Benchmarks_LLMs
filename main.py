"""
Medical Benchmark Runner

Supports running one or multiple benchmarks in a single call.

config.yaml options:
  benchmark: medqa                        # single benchmark
  benchmark: [medqa, rar, radbench]       # list
  benchmark: all                          # run every registered benchmark

Output layout (one folder per run, see README "Output Files"):
  results/run_<timestamp>/
    {benchmark}_results.csv / _report.jsonl / _status.json
    run_<ts>_<job>.log, run_info_<ts>_<job>.json, fingerprint.json
    benchmark_results_<model>.png
"""
import yaml
import os
import re
import sys
import importlib
import datetime

from core.client import MedicalLLMClient, ServerUnavailableError, ConfigurationError
from core.fileio import atomic_to_csv
from core.logger import RunLogger
from core.summary import (STATUS_KEYS, build_status, write_status, flatten_report,
                          merge_metrics, load_run_summary, format_metrics_line)
from core import run_utils

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
    """Judge client from the `judge:` block. Sampling settings come from that block
    (temperature default 0, max_tokens default 512), not from benchmark_settings."""
    judge_cfg = config.get("judge")
    if not judge_cfg:
        return None
    server = dict(judge_cfg)
    server.setdefault("seed", (config.get("server") or {}).get("seed", 42))
    merged = {
        "server": server,
        "benchmark_settings": {
            "temperature": judge_cfg.get("temperature", 0),
            "max_tokens": judge_cfg.get("max_tokens", 512),
        },
    }
    try:
        return MedicalLLMClient(merged)
    except (ValueError, KeyError, RuntimeError) as e:
        print(f"  Warning: could not create judge client: {e}")
        return None


class _PartialRun(Exception):
    """A benchmark stopped because the server is unavailable / misconfigured;
    carries the metrics of the partial evaluation."""

    def __init__(self, metrics: dict, headline: list, cause: Exception):
        super().__init__(str(cause))
        self.metrics, self.headline, self.cause = metrics, headline, cause


def _run_one(benchmark: str, registry: dict, config: dict, client, judge_client,
             run_dir: str, logger=None):
    """Load data, run the task, and evaluate for a single benchmark.

    Returns (metrics, headline_keys). Raises _PartialRun if the server became
    unavailable (the partial results are still evaluated and flagged)."""
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

    stop_exc = None
    try:
        task = importlib.import_module(task_module_path)
        task.run(config, client, data, results_path, logger=logger)
    except ServerUnavailableError as e:   # includes ConfigurationError
        stop_exc = e
        print(f"\n  STOPPED: {type(e).__name__}: {e}")
    finally:
        client.max_tokens = orig_max_tokens

    eval_path, n_done, n_errors, cleanup = _prepare_results_for_eval(results_path, set(ids), run_dir)
    stop_reason = f"{type(stop_exc).__name__}: {str(stop_exc)[:200]}" if stop_exc else None
    status = build_status(n_done, len(data), n_errors, stop_reason)
    write_status(run_dir, benchmark, status)
    if not status["complete"]:
        print(f"  WARNING: {benchmark} INCOMPLETE – {n_done}/{len(data)} items in results "
              f"(resume with --run-dir {run_dir})")
    if n_errors:
        print(f"  WARNING: {n_errors} API errors in results (scored as wrong; "
              f"rerun with --run-dir {run_dir} to retry them)")

    returned = {}
    try:
        if n_done > 0:
            returned = _evaluate(benchmark, eval_type, eval_path, report_path, judge_client,
                                 config=config, logger=logger)
        else:
            print("  No results to evaluate.")
    except Exception as e:
        if stop_exc is None:
            raise
        print(f"  WARNING: evaluation of the partial results failed: {e}")
    finally:
        cleanup()

    headline = [k for k, v in returned.items()
                if k != "path" and (v is None or isinstance(v, (int, float)))]
    flat = flatten_report(report_path) if returned else {}
    metrics = {**{k: status[k] for k in STATUS_KEYS},
               "stop_reason": status["stop_reason"],
               **merge_metrics(returned, flat)}
    if stop_exc is not None:
        raise _PartialRun(metrics, headline, stop_exc)
    return metrics, headline


def _error_mask(df):
    import pandas as pd
    mask = pd.Series(False, index=df.index)
    for col in ("model_answer", "model_raw"):
        if col in df.columns:
            mask |= df[col].fillna("").astype(str).str.startswith("Error:")
    return mask


def _read_results(results_path: str):
    import pandas as pd
    if not (os.path.exists(results_path) and os.path.getsize(results_path) > 0):
        return None
    return pd.read_csv(results_path, dtype=str, keep_default_na=False)


def _drop_error_rows(results_path: str) -> None:
    """On resume, remove rows whose model call failed so they are asked again."""
    df = _read_results(results_path)
    if df is None:
        return
    mask = _error_mask(df)
    if mask.any():
        atomic_to_csv(df[~mask], results_path)
        print(f"  Resume: {int(mask.sum())} failed rows removed, will be retried")


def _prepare_results_for_eval(results_path: str, current_ids: set, run_dir: str):
    """Deduplicate the results CSV by id (keep the last answer, rewrite atomically)
    and restrict evaluation to the ids of the current data.

    If the CSV holds answers for ids outside the current data (e.g. resumed with a
    smaller --limit), those rows stay in the CSV but a filtered copy is written to
    <run_dir>/.eval_subset/<bench>_results.csv and evaluated instead; its judge
    cache is a symlink to the real <bench>_judge_cache.csv so verdicts are shared.

    Returns (eval_path, n_done, n_api_errors, cleanup_fn); n_* count current ids only.
    """
    df = _read_results(results_path)
    if df is None or "id" not in df.columns:
        return results_path, 0, 0, lambda: None

    dup = df.duplicated(subset="id", keep="last")
    if dup.any():
        print(f"  WARNING: {int(dup.sum())} duplicate result rows (same id) removed, "
              f"keeping the latest answer per id")
        df = df[~dup]
        atomic_to_csv(df, results_path)

    in_scope = df["id"].isin(current_ids)
    n_done = int(in_scope.sum())
    n_errors = int((_error_mask(df) & in_scope).sum())
    if in_scope.all():
        return results_path, n_done, n_errors, lambda: None

    n_out = int((~in_scope).sum())
    print(f"  Note: {n_out} result rows belong to items outside the current data "
          f"(e.g. smaller --limit); they are kept but not evaluated")
    sub_dir = os.path.join(run_dir, ".eval_subset")
    os.makedirs(sub_dir, exist_ok=True)
    name = os.path.basename(results_path)
    eval_path = os.path.join(sub_dir, name)
    atomic_to_csv(df[in_scope], eval_path)
    base = name[: -len("_results.csv")] if name.endswith("_results.csv") else name
    cache_link = os.path.join(sub_dir, f"{base}_judge_cache.csv")
    if os.path.lexists(cache_link):
        os.unlink(cache_link)
    os.symlink(os.path.join("..", f"{base}_judge_cache.csv"), cache_link)

    def cleanup():
        for p in (eval_path, cache_link):
            if os.path.lexists(p):
                os.unlink(p)
        try:
            os.rmdir(sub_dir)
        except OSError:
            pass   # another benchmark still uses it
    return eval_path, n_done, n_errors, cleanup


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

def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Run medical LLM benchmarks (settings in config.yaml).")
    parser.add_argument("--config", help="YAML config to use (default: config.yaml, "
                                         "else config.default.yaml).")
    parser.add_argument("--model", help="Override server.model_name.")
    parser.add_argument("--run-dir", help="Existing or new run directory; an existing one is resumed "
                                          "(finished items are skipped, failed ones retried).")
    parser.add_argument("--benchmark", help="Comma-separated benchmarks, overrides config 'benchmark'.")
    parser.add_argument("--limit", help="Override benchmark_settings.limit_samples (number or 'all').")
    parser.add_argument("--force-resume", action="store_true",
                        help="Resume a run dir even if model_name / judge model differ from its "
                             "fingerprint.json (results become mixed!).")
    parser.add_argument("--skip-health-check", action="store_true",
                        help="Do not query GET /v1/models before the run.")
    return parser.parse_args(argv)


def _load_config(args):
    if args.config:
        config_path = args.config
        if not os.path.exists(config_path):
            raise SystemExit(f"Config file not found: {config_path}")
    else:
        config_path = "config.yaml" if os.path.exists("config.yaml") else "config.default.yaml"
        if config_path != "config.yaml":
            print("Hinweis: config.yaml nicht gefunden, nutze config.default.yaml.")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if args.model:
        config.setdefault("server", {})["model_name"] = args.model
    if args.benchmark:
        config["benchmark"] = [b.strip() for b in args.benchmark.split(",") if b.strip()]
    if args.limit is not None:
        config.setdefault("benchmark_settings", {})["limit_samples"] = (
            None if args.limit.lower() in ("all", "none", "null") else int(args.limit)
        )
    return config, config_path


def _print_results(summary: list, headlines: dict) -> None:
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    for name, metrics, err in summary:
        if err:
            print(f"  {name:<22} ERROR: {err}")
            continue
        keys = list(STATUS_KEYS[:3]) + list(headlines.get(name, []))
        line = format_metrics_line(metrics, headline_keys=keys)
        print(f"  {name:<22} {line if line else 'no metrics'}")
    print(f"{'='*60}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    config, config_path = _load_config(args)

    registry = _registry()
    benchmarks = _resolve_benchmarks(config, registry)
    model_name = config.get("server", {}).get("model_name", "unknown")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-run subfolder for CSVs, JSONL reports, logs and run info. Under SLURM the
    # job id is part of the default name, so jobs started in the same second differ.
    job = os.environ.get("SLURM_JOB_ID")
    run_dir = args.run_dir or os.path.join("results", f"run_{ts}" + (f"_job{job}" if job else ""))
    os.makedirs(run_dir, exist_ok=True)

    # One process per benchmark and run dir (two SLURM jobs on the same --run-dir
    # would otherwise append to the same CSV and overwrite each other's files).
    locks = run_utils.BenchmarkLocks(run_dir, benchmarks)
    try:
        locks.acquire()
    except run_utils.LockHeldError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    log_path = os.path.join(run_dir, f"run_{ts}_{run_utils.job_tag()}.log")
    exit_code = 0
    try:
        with RunLogger(log_path) as logger:
            try:
                exit_code = _main_logged(args, config, config_path, benchmarks, registry,
                                         model_name, run_dir, ts, log_path, logger)
            except Exception:
                import traceback
                traceback.print_exc()   # goes into the log via the stderr tee
                exit_code = 1
    finally:
        locks.release()
    return exit_code


def _main_logged(args, config, config_path, benchmarks, registry, model_name,
                 run_dir, ts, log_path, logger) -> int:
    print(f"Run dir: {run_dir}")
    print(f"Log:     {log_path}")
    print(f"Config:  {config_path}")
    print(f"Model:   {model_name}")
    print(f"Limit:   {config.get('benchmark_settings', {}).get('limit_samples')}")
    print(f"Concurrency: {config.get('benchmark_settings', {}).get('concurrency', 4)}")

    client = MedicalLLMClient(config)
    judge_client = _build_judge_client(config)
    if judge_client:
        print(f"Judge model: {judge_client.model} (temperature={judge_client.temperature}, "
              f"max_tokens={judge_client.max_tokens})")
    else:
        print("No judge model configured — LLM-as-a-Judge will be skipped.")
        print("To enable: add a 'judge:' section to config.yaml")

    served_models = None
    if not args.skip_health_check:
        try:
            served_models = client.health_check("model")
            if judge_client:
                judge_client.health_check("judge model")
        except ConfigurationError as e:
            print(f"ERROR: {e}")
            return 2
        if served_models is not None:
            print(f"Health check: {model_name} is served ({len(served_models)} models available)")

    # after the health check, so a typo in model_name does not end up in fingerprint.json
    try:
        run_utils.check_fingerprint(run_dir, config, benchmarks, force=args.force_resume)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2

    info =run_utils.build_run_info(config, benchmarks, args=args, config_path=config_path,
                                    served_models=served_models)
    info["log_file"] = os.path.basename(log_path)
    run_utils.write_run_info(run_dir, info, ts)

    summary, headlines = [], {}
    abort = None
    for benchmark in benchmarks:
        if abort:
            summary.append((benchmark, {}, f"skipped – {abort}"))
            continue
        try:
            metrics, headline = _run_one(benchmark, registry, config, client, judge_client,
                                         run_dir=run_dir, logger=logger)
            summary.append((benchmark, metrics, None))
            headlines[benchmark] = headline
        except _PartialRun as p:
            summary.append((benchmark, p.metrics, None))
            headlines[benchmark] = p.headline
            abort = f"{type(p.cause).__name__} in {benchmark}"
            print(f"\nStopping the run: {p.cause}")
        except Exception as e:
            import traceback
            print(f"\nError in benchmark '{benchmark}': {e}")
            traceback.print_exc()
            summary.append((benchmark, {}, str(e)))

    _print_results(summary, headlines)

    # run_info: completion and the model id the server actually answered with
    info["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    reported = sorted(client.reported_models)
    info["server_reported_models"] = reported
    if reported and reported != [model_name]:
        print(f"  Note: server reported model id(s) {reported} for requested {model_name!r}")
    if judge_client:
        info["judge_server_reported_models"] = sorted(judge_client.reported_models)
    info["status"] = {name: ({k: m.get(k) for k in STATUS_KEYS} if not err else {"error": err})
                      for name, m, err in summary}
    run_utils.write_run_info(run_dir, info, ts)

    # Bar chart of all benchmarks evaluated in this run dir (also those of other jobs)
    try:
        from core.plot import generate_results_chart
        by_name = {name: (name, m, None) for name, m, _ in load_run_summary(run_dir)}
        for name, m, err in summary:
            if not err and m:
                by_name[name] = (name, m, None)
        chart_name = f"benchmark_results_{_sanitize_filename(model_name)}.png"
        generate_results_chart(list(by_name.values()), model_name, os.path.join(run_dir, chart_name))
    except Exception as e:
        print(f"  Chart generation failed: {e}")

    failed = [name for name, m, err in summary if err or m.get("complete") is False]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


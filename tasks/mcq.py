"""
MCQ Task Runner – used for MedQA, RaR and RadioRAG.

The benchmarks share the same multiple-choice schema:
  item = { id, benchmark, question, options: [{key, value}], correct_answer, meta }

Results CSV columns:
  id, benchmark, question, correct_answer, option_keys, model_answer,
  finish_reason, completion_tokens
Evaluation: letter extraction → accuracy (handled by evaluate.py; only letters in
option_keys are accepted).

Note: RadBench is a VLM benchmark (X-ray images) and goes through tasks/vqa.py.

This module also contains the concurrent, resumable runner shared with tasks/vqa.py.
"""
import os
import csv
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd
from core.fileio import atomic_to_csv


META_FIELDS = ["finish_reason", "completion_tokens"]


def _parse_benchmark_settings(config: dict):
    s = config.get("benchmark_settings", {})
    sleep_s = float(s.get("sleep_s", 0) or 0)
    max_errors = s.get("max_errors", None)
    return sleep_s, int(max_errors) if max_errors is not None else None


def _n_workers(config: dict) -> int:
    return max(1, int((config.get("benchmark_settings", {}) or {}).get("concurrency", 1) or 1))


def prepare_results_file(results_path: str, fieldnames: list):
    """
    Read an existing results CSV for resuming. Returns (completed_ids, columns).

    The file is read as text (dtype=str, keep_default_na=False). If it exists but
    cannot be read, an error is raised – re-running everything silently would mix two
    runs. If it lacks columns of *fieldnames*, the file is rewritten with these columns
    added (empty for existing rows), so appended
    rows stay aligned with the header.
    """
    if not (os.path.exists(results_path) and os.path.getsize(results_path) > 0):
        return set(), list(fieldnames)
    try:
        existing = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    except Exception as e:
        raise RuntimeError(
            f"Existing results file {results_path} cannot be read ({e}). "
            "Repair or move it away; refusing to start over silently."
        ) from e
    if "id" not in existing.columns:
        raise RuntimeError(f"Existing results file {results_path} has no 'id' column; cannot resume.")
    columns = list(existing.columns)
    missing = [c for c in fieldnames if c not in columns]
    if missing:
        for c in missing:
            existing[c] = ""
        columns += missing
        atomic_to_csv(existing[columns], results_path)
        print(f"  Resume: added columns {missing} to {results_path}")
    completed = set(existing["id"].astype(str).tolist())
    if completed:
        print(f"Resume: {len(completed)} items already answered, skipping them.")
    return completed, columns


def _rate_str(n: int, elapsed: float) -> str:
    if n <= 0 or elapsed <= 0:
        return "-"
    rate = n / elapsed
    return f"{rate:.1f} q/s" if rate >= 1 else f"{elapsed / n:.1f} s/q"


def _meta_fields(client) -> dict:
    """finish_reason / completion_tokens of the last call in this thread (client.last_meta)."""
    meta = getattr(client, "last_meta", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    tokens = meta.get("completion_tokens")
    return {
        "finish_reason": "" if meta.get("finish_reason") is None else str(meta.get("finish_reason")),
        "completion_tokens": "" if tokens is None else str(tokens),
    }


def run_items(config: dict, client, data: list, results_path: str, fieldnames: list,
              ask, base_row, logger=None, describe=None, label: str = "questions",
              answer_field: str = "model_answer") -> str:
    """
    Ask all items not yet in *results_path* and append one CSV row per answer.

    ask(item) -> str         model call, executed in a worker thread
    base_row(item) -> dict   CSV fields except the answer / finish_reason / completion_tokens
    answer_field             CSV column of the model answer (default model_answer)
    describe(item) -> str    short tag for the verbose log (optional)

    Concurrency: benchmark_settings.concurrency worker threads (default 1). Only the
    main thread writes the CSV (one row per finished item, flushed). At most
    2 × workers items are in flight, so an abort leaves little unfinished work.
    core.client.ServerUnavailableError cancels pending items and is re-raised;
    benchmark_settings.max_errors stops submitting after that many API errors.
    """
    sleep_s, max_errors = _parse_benchmark_settings(config)
    n_workers = _n_workers(config)

    completed_ids, columns = prepare_results_file(results_path, fieldnames)

    total = len(data)
    position = {str(item.get("id")): i for i, item in enumerate(data, start=1)}
    todo = [item for item in data if str(item.get("id")) not in completed_ids]
    remaining = len(todo)
    print(f"  {total} {label}  ({remaining} remaining, {n_workers} worker{'s' if n_workers > 1 else ''})...")

    def _work(item):
        answer = ask(item)
        meta = _meta_fields(client)          # read right after the call, same thread
        if sleep_s > 0:
            time.sleep(sleep_s)
        return answer, meta

    start = time.time()
    processed_new = 0
    errors = 0
    stop = False

    file_exists = os.path.exists(results_path) and os.path.getsize(results_path) > 0
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        if not file_exists:
            writer.writeheader()
            f.flush()

        pool = ThreadPoolExecutor(max_workers=n_workers)
        pending = {}
        queue = iter(todo)
        try:
            def _fill():
                while not stop and len(pending) < 2 * n_workers:
                    item = next(queue, None)
                    if item is None:
                        return
                    pending[pool.submit(_work, item)] = item

            _fill()
            while pending:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                # `done` is unordered: write finished answers before re-raising a
                # failed future, so no completed answer is lost on abort.
                failed = None
                for fut in sorted(done, key=lambda d: d.exception() is not None):
                    item = pending.pop(fut)
                    if fut.exception() is not None:
                        failed = failed or fut
                        continue
                    answer, meta = fut.result()
                    item_id = str(item.get("id"))
                    is_error = isinstance(answer, str) and answer.startswith("Error:")
                    if is_error:
                        errors += 1

                    row = dict(base_row(item))
                    row.update({"id": item_id, answer_field: answer, **meta})
                    writer.writerow(row)
                    f.flush()
                    processed_new += 1

                    if logger:
                        status = f"ERROR: {answer}" if is_error else str(answer).strip()[:60]
                        if meta.get("finish_reason") == "length":
                            status += "  [truncated]"
                        tag = f" ({describe(item)})" if describe else ""
                        logger.verbose(f"[{position[item_id]:>{len(str(total))}}/{total}] {item_id}{tag}  →  {status}")

                    if is_error and max_errors is not None and errors >= max_errors and not stop:
                        print(f"Stopping: max_errors={max_errors} reached.")
                        stop = True

                    if processed_new % 50 == 0:
                        elapsed = time.time() - start
                        rate = processed_new / elapsed if elapsed > 0 else 0.0
                        eta_s = int((remaining - processed_new) / rate) if rate > 0 else -1
                        eta = f"{eta_s // 60:02d}:{eta_s % 60:02d}" if eta_s >= 0 else "?"
                        pct = int(processed_new / remaining * 100) if remaining > 0 else 100
                        print(f"  [{processed_new:>{len(str(remaining))}}/{remaining}] {pct:3d}%  "
                              f"{_rate_str(processed_new, elapsed)}  ETA {eta}  errors: {errors}")
                if failed is not None:
                    failed.result()                  # ServerUnavailableError propagates
                _fill()
        except BaseException:
            for fut in pending:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

    elapsed_total = time.time() - start
    print(f"  Done: {processed_new}/{remaining}  errors: {errors}  "
          f"({elapsed_total / 60:.1f} min, {_rate_str(processed_new, elapsed_total)})")
    return results_path


# ---------------------------------------------------------------------------
# MCQ benchmarks
# ---------------------------------------------------------------------------

FIELDNAMES = ["id", "benchmark", "question", "correct_answer", "option_keys", "model_answer"] + META_FIELDS


def build_prompt(item: dict) -> str:
    opts = item.get("options", [])
    options_str = ", ".join(f"{opt['key']}: {opt['value']}" for opt in opts)
    keys = "/".join(opt["key"] for opt in opts) if opts else "A/B/C/D"
    return (
        f"Question: {item['question']}\n"
        f"Options: {options_str}\n"
        f"Reply with only the correct letter ({keys})."
    )


def run(config: dict, client, data: list, results_path: str, logger=None) -> str:
    """
    Run an MCQ benchmark and write incremental results to *results_path*.

    Returns the path to the written CSV.
    """
    def _base_row(item):
        return {
            "benchmark": item.get("benchmark", ""),
            "question": item["question"],
            "correct_answer": item.get("correct_answer", ""),
            "option_keys": "".join(str(o["key"]) for o in item.get("options", []) or []),
        }

    return run_items(config, client, data, results_path, FIELDNAMES,
                     ask=lambda item: client.ask_question(build_prompt(item)),
                     base_row=_base_row, logger=logger)

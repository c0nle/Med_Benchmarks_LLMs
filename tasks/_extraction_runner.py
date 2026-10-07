"""
Shared, concurrent, resumable runner for the label-extraction tasks
(tasks/mamma_extraction.py, tasks/arm_extraction.py).

- Requests run in a ThreadPoolExecutor (benchmark_settings.concurrency, default 1).
  Each worker builds the prompt, calls the model, parses the answer and builds the
  complete CSV row (incl. anything that needs the report text, e.g. the Arm citation
  check), so the report text never has to leave the worker.
- The main thread is the only CSV writer: it consumes futures as they complete and
  appends + flushes one row per item.
- core.client.ServerUnavailableError: pending futures are cancelled, rows of requests
  that were already running are still written, then the error is re-raised.
- max_errors: after that many failed requests no new requests are started.
- Resume: the existing CSV is read with dtype=str / keep_default_na=False; an existing
  but unreadable CSV raises instead of silently starting a full re-run.
"""
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from core.fileio import atomic_to_csv

try:
    from core.client import ServerUnavailableError
except Exception:  # pragma: no cover - core.client always exists in this repo
    class ServerUnavailableError(RuntimeError):
        pass


def settings(config: dict):
    """(sleep_s, max_errors, n_workers) from benchmark_settings."""
    s = (config or {}).get("benchmark_settings", {}) or {}
    sleep_s = float(s.get("sleep_s", 0) or 0)
    max_errors = s.get("max_errors", None)
    max_errors = int(max_errors) if max_errors is not None else None
    n_workers = int(s.get("concurrency", 1) or 1)
    return sleep_s, max_errors, max(1, n_workers)


def call_model(client, prompt: str, system_prompt: str):
    """Ask the model; returns (answer, meta) with meta = client.last_meta of this thread."""
    answer = client.ask_question(prompt, system_prompt=system_prompt)
    meta = getattr(client, "last_meta", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    return answer, meta


def meta_columns(meta: dict) -> dict:
    """finish_reason / completion_tokens CSV columns from client.last_meta."""
    ct = meta.get("completion_tokens")
    return {
        "finish_reason": "" if meta.get("finish_reason") is None else str(meta.get("finish_reason")),
        "completion_tokens": "" if ct is None else str(ct),
    }


def format_rate(n_done: int, elapsed: float) -> str:
    """'3.2 items/s', or '12.5 s/item' below one item per second."""
    if n_done <= 0 or elapsed <= 0:
        return "? items/s"
    rate = n_done / elapsed
    if rate >= 1:
        return f"{rate:.1f} items/s"
    return f"{elapsed / n_done:.1f} s/item"


def _read_completed(results_path: str, fieldnames: list):
    """Return (completed_ids, file_has_rows). Adds missing columns to the header in place."""
    if not (os.path.exists(results_path) and os.path.getsize(results_path) > 0):
        return set(), False
    try:
        existing = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    except Exception as e:
        raise RuntimeError(
            f"Existing results file {results_path} is not readable ({e}); "
            "refusing to start a full re-run. Fix or move the file."
        ) from e
    if "id" not in existing.columns:
        raise RuntimeError(f"Existing results file {results_path} has no 'id' column.")
    missing_cols = [c for c in fieldnames if c not in existing.columns]
    if missing_cols:
        # Header without some of the current columns: add them (empty) so appended
        # rows line up with the header.
        for c in missing_cols:
            existing[c] = ""
        extra = [c for c in existing.columns if c not in fieldnames]
        atomic_to_csv(existing[fieldnames + extra], results_path)
    completed = {str(i) for i in existing["id"].tolist() if str(i)}
    return completed, True


def run_items(config: dict, data: list, results_path: str, fieldnames: list,
              process_item, logger=None, unit: str = "items") -> str:
    """
    process_item(item) -> (row: dict, is_error: bool, status: str); runs in a worker thread.
    """
    sleep_s, max_errors, n_workers = settings(config)

    completed_ids, file_exists = _read_completed(results_path, fieldnames)
    if completed_ids:
        print(f"Resume: {len(completed_ids)} {unit} already in results.")

    todo = [it for it in data if str(it.get("id")) not in completed_ids]
    total, remaining = len(data), len(todo)
    print(f"  {total} {unit}  ({remaining} remaining, {n_workers} worker(s))...")

    if file_exists:
        with open(results_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
    else:
        header = list(fieldnames)

    def _work(item):
        result = process_item(item)
        if sleep_s > 0:
            time.sleep(sleep_s)
        return result

    start = time.time()
    processed_new = errors = 0
    stop = False
    width = len(str(remaining))

    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
            f.flush()

        def _write(item, row, status):
            nonlocal processed_new
            writer.writerow(row)
            f.flush()
            processed_new += 1
            if logger:
                logger.verbose(f"[{processed_new:>{width}}/{remaining}] {item.get('id')}  →  {status}")
            if processed_new % 50 == 0:
                elapsed = time.time() - start
                rate = processed_new / elapsed if elapsed > 0 else 0.0
                eta_s = int((remaining - processed_new) / rate) if rate > 0 else -1
                eta = f"{eta_s // 60:02d}:{eta_s % 60:02d}" if eta_s >= 0 else "?"
                pct = int(processed_new / remaining * 100) if remaining > 0 else 100
                print(f"  [{processed_new:>{width}}/{remaining}] {pct:3d}%  "
                      f"{format_rate(processed_new, elapsed)}  ETA {eta}  errors: {errors}")

        pool = ThreadPoolExecutor(max_workers=n_workers)
        futures = {pool.submit(_work, item): item for item in todo}
        handled = set()
        try:
            for fut in as_completed(futures):
                handled.add(fut)
                if fut.cancelled():
                    continue
                item = futures[fut]
                try:
                    row, is_error, status = fut.result()
                except ServerUnavailableError:
                    for other in futures:
                        other.cancel()
                    # Keep the answers of requests that were already running
                    for other in futures:
                        if other in handled or other.cancelled():
                            continue
                        handled.add(other)
                        try:
                            o_row, o_err, o_status = other.result()
                        except Exception:
                            continue
                        if not o_err:
                            _write(futures[other], o_row, o_status)
                    raise
                if stop:
                    # max_errors reached: only keep successful answers still in flight
                    if not is_error:
                        _write(item, row, status)
                    continue
                if is_error:
                    errors += 1
                    if max_errors is not None and errors >= max_errors:
                        print(f"Abort: max_errors={max_errors} reached.")
                        stop = True
                        for other in futures:
                            other.cancel()
                        continue
                _write(item, row, status)
        finally:
            # On any exception: do not start the remaining requests
            for other in futures:
                other.cancel()
            pool.shutdown(wait=True)

    elapsed_total = time.time() - start
    print(f"  Done: {processed_new}/{remaining}  errors: {errors}  ({elapsed_total / 60:.1f} min, "
          f"{format_rate(processed_new, elapsed_total)})")
    return results_path

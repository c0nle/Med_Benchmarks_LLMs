"""
Run-directory bookkeeping: per-benchmark locks, resume fingerprint and run_info.
"""
import copy
import datetime
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from urllib.parse import urlparse

from core.fileio import atomic_write_json

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job_tag() -> str:
    """SLURM job id (or pid) – part of log / run_info file names so that two jobs
    started in the same second never write to the same file."""
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        return f"job{job}" + (f"_{task}" if task else "")
    return f"pid{os.getpid()}"


# ---------------------------------------------------------------------------
# Locks
# ---------------------------------------------------------------------------

class LockHeldError(RuntimeError):
    pass


class BenchmarkLocks:
    """Exclusive, non-blocking fcntl.flock on <run_dir>/.<benchmark>.lock for every
    benchmark. Raises LockHeldError if another process holds one of them. The
    lock is released automatically when the process exits (also on a crash)."""

    def __init__(self, run_dir: str, benchmarks):
        self.run_dir = run_dir
        self.benchmarks = list(benchmarks)
        self._files = []

    def acquire(self):
        try:
            for name in self.benchmarks:
                path = os.path.join(self.run_dir, f".{name}.lock")
                f = open(path, "a+", encoding="utf-8")
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    f.seek(0)
                    holder = f.read().strip() or "unknown process"
                    f.close()
                    raise LockHeldError(
                        f"benchmark '{name}' in {self.run_dir} is already running ({holder}). "
                        f"Wait for that job to finish or use another --run-dir."
                    )
                f.seek(0)
                f.truncate()
                f.write(f"{job_tag()} host={socket.gethostname()} "
                        f"since={datetime.datetime.now().isoformat(timespec='seconds')}\n")
                f.flush()
                self._files.append(f)
        except BaseException:
            self.release()
            raise
        return self

    def release(self):
        for f in self._files:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            finally:
                f.close()
        self._files = []

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()


# ---------------------------------------------------------------------------
# Resume fingerprint
# ---------------------------------------------------------------------------

FINGERPRINT_FILE = "fingerprint.json"
# Differences in these keys make resumed results incomparable -> refuse.
STRICT_KEYS = ("model_name", "judge_model")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def effective_max_tokens(config: dict, benchmark: str):
    task = (config.get("task_settings") or {}).get(benchmark) or {}
    if "max_tokens" in task:
        return int(task["max_tokens"])
    mt = (config.get("benchmark_settings") or {}).get("max_tokens")
    return int(mt) if mt is not None else None


def build_fingerprint(config: dict, benchmarks) -> dict:
    from core.client import DEFAULT_SYSTEM_PROMPT
    server = config.get("server") or {}
    judge = config.get("judge") or {}
    settings = config.get("benchmark_settings") or {}
    return {
        "model_name": server.get("model_name"),
        "server_host": urlparse(str(server.get("url") or "")).netloc,
        "temperature": settings.get("temperature", 0),
        "seed": server.get("seed", 42),
        "extra_body": server.get("extra_body") or {},
        "max_tokens": {b: effective_max_tokens(config, b) for b in benchmarks},
        "system_prompt_sha256": _sha256(DEFAULT_SYSTEM_PROMPT)[:16],
        "judge_model": judge.get("model_name") if judge else None,
    }


def check_fingerprint(run_dir: str, config: dict, benchmarks, force: bool = False) -> dict:
    """Write <run_dir>/fingerprint.json on the first run; on resume compare.

    Refuses (RuntimeError) if model_name or judge_model differ, unless *force*.
    Other differences only print a warning. Max_tokens of benchmarks that are new
    to this run dir are added to the stored fingerprint."""
    path = os.path.join(run_dir, FINGERPRINT_FILE)
    current = build_fingerprint(config, benchmarks)
    if not os.path.exists(path):
        has_results = any(f.endswith("_results.csv") for f in os.listdir(run_dir))
        if has_results:
            print(f"  WARNING: {run_dir} has results but no {FINGERPRINT_FILE}; "
                  f"cannot verify that the settings match the earlier run.")
        atomic_write_json(path, current)
        return current

    with open(path, encoding="utf-8") as f:
        stored = json.load(f)

    strict, soft = [], []
    for key, value in current.items():
        old = stored.get(key)
        if key == "max_tokens":
            for b, mt in value.items():
                if b in (old or {}) and old[b] != mt:
                    soft.append(f"max_tokens[{b}]: {old[b]} -> {mt}")
            continue
        if key not in stored or old == value:
            continue
        # a judge configured now but not before has no old verdicts to mix with
        is_strict = key in STRICT_KEYS and old is not None
        (strict if is_strict else soft).append(f"{key}: {old!r} -> {value!r}")

    if strict:
        msg = (f"run dir {run_dir} was started with different settings: " + "; ".join(strict)
               + ". Use a new --run-dir, or --force-resume to mix them anyway.")
        if not force:
            raise RuntimeError(msg)
        print(f"  WARNING (--force-resume): {msg}")
    for d in soft:
        print(f"  WARNING: setting differs from the earlier run in this dir: {d}")

    # remember max_tokens of benchmarks run here for the first time
    new_mt = {b: mt for b, mt in current["max_tokens"].items() if b not in (stored.get("max_tokens") or {})}
    if new_mt:
        stored.setdefault("max_tokens", {}).update(new_mt)
        atomic_write_json(path, stored)
    return stored


# ---------------------------------------------------------------------------
# run_info
# ---------------------------------------------------------------------------

_PACKAGES = ("pandas", "numpy", "requests", "openai", "httpx", "matplotlib", "nltk",
             "datasets", "pyarrow", "openpyxl", "Pillow", "PyYAML")
_SECRET_KEYS = {"api_key", "apikey", "authorization", "token", "password", "secret"}


def _package_versions() -> dict:
    from importlib import metadata
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, check=True,
                          timeout=30).stdout.decode("utf-8", errors="replace")


def git_info() -> dict:
    """Commit, porcelain status (incl. untracked files) and a hash of the diff.
    dirty = any tracked change or untracked (non-ignored) file."""
    try:
        commit = _git("rev-parse", "HEAD").strip()
        status = _git("status", "--porcelain")
        diff = _git("diff", "HEAD", "--binary")
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown"}
    dirty = bool(status.strip())
    return {
        "git_commit": commit + ("-dirty" if dirty else ""),
        "git_dirty": dirty,
        "git_status_porcelain": status.splitlines(),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def redact(obj):
    """Deep copy without secrets (api keys, tokens) at any depth."""
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items() if str(k).lower() not in _SECRET_KEYS}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return copy.deepcopy(obj)


def build_run_info(config: dict, benchmarks, args=None, config_path: str = None,
                   served_models=None) -> dict:
    info = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pid": os.getpid(),
        "argv": sys.argv,
        "cli_args": redact(vars(args)) if args is not None else None,
        "config_path": os.path.abspath(config_path) if config_path else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        **git_info(),
        "benchmarks": list(benchmarks),
        "served_models": served_models,
        "server_reported_models": None,
        "config": redact(config),
    }
    return info


def write_run_info(run_dir: str, info: dict, ts: str) -> str:
    path = os.path.join(run_dir, f"run_info_{ts}_{job_tag()}.json")
    atomic_write_json(path, info)
    return path

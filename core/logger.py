"""
RunLogger — tee all print() output (stdout and stderr) to a log file and add verbose-only detail.

Usage (context manager):
    with RunLogger("results/run_20240415_120000.log") as logger:
        ...
        logger.verbose("only in log file")
        print("goes to both terminal and log")

All print() calls and everything written to stderr (tracebacks, warnings)
inside the with-block are mirrored to the log file. The file is opened in
append mode, so a resumed run continues the same log.
logger.verbose() writes only to the file — not shown on terminal.
All writes are serialised by a lock (benchmarks may run worker threads).
"""
import sys
import datetime
import threading


class _Tee:
    """File-like object that writes to a terminal stream and the shared log file."""

    def __init__(self, terminal, owner: "RunLogger"):
        self._terminal = terminal
        self._owner = owner

    def write(self, msg: str):
        n = self._terminal.write(msg)
        self._owner._write_file(msg)
        return n

    def flush(self):
        self._terminal.flush()
        self._owner._flush_file()

    def isatty(self) -> bool:
        try:
            return self._terminal.isatty()
        except (AttributeError, ValueError):
            return False

    def fileno(self) -> int:
        return self._terminal.fileno()

    @property
    def encoding(self):
        return getattr(self._terminal, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._terminal, "errors", "strict")

    def __getattr__(self, name):
        # Everything else (buffer, mode, writable, ...) comes from the terminal stream
        return getattr(self._terminal, name)


class RunLogger:
    def __init__(self, path: str, tee_stderr: bool = True):
        self._terminal = sys.stdout
        self._terminal_err = sys.stderr
        self._tee_stderr = tee_stderr
        self._lock = threading.Lock()
        self._file = open(path, "a", encoding="utf-8")
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"=== Run Log {ts} ===\n\n")
        self._file.flush()
        self._stdout_tee = _Tee(self._terminal, self)
        self._stderr_tee = _Tee(self._terminal_err, self)

    def _write_file(self, msg: str):
        with self._lock:
            if not self._file.closed:
                self._file.write(msg)

    def _flush_file(self):
        with self._lock:
            if not self._file.closed:
                self._file.flush()

    # -- sys.stdout interface (kept for callers that use the logger directly) ----

    def write(self, msg: str):
        return self._stdout_tee.write(msg)

    def flush(self):
        self._stdout_tee.flush()

    def isatty(self) -> bool:
        return self._stdout_tee.isatty()

    def fileno(self) -> int:
        return self._stdout_tee.fileno()

    @property
    def encoding(self):
        return self._stdout_tee.encoding

    # -- verbose-only ---------------------------------------------------------

    def verbose(self, msg: str):
        """Write to log file only — not shown on terminal."""
        with self._lock:
            if not self._file.closed:
                self._file.write(msg + "\n")
                self._file.flush()

    # -- context manager ------------------------------------------------------

    def __enter__(self):
        sys.stdout = self._stdout_tee
        if self._tee_stderr:
            sys.stderr = self._stderr_tee
        return self

    def __exit__(self, *_):
        sys.stdout = self._terminal
        sys.stderr = self._terminal_err
        with self._lock:
            self._file.close()

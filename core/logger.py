"""
RunLogger — tee all print() output to a log file and add verbose-only detail.

Usage (context manager):
    with RunLogger("results/run_20240415_120000.log") as logger:
        ...
        logger.verbose("only in log file")
        print("goes to both terminal and log")

All print() calls inside the with-block are mirrored to the log file.
logger.verbose() writes only to the file — not shown on terminal.
"""
import sys
import datetime


class RunLogger:
    def __init__(self, path: str):
        self._terminal = sys.stdout
        self._file = open(path, "w", encoding="utf-8")
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"=== Run Log {ts} ===\n\n")
        self._file.flush()

    # -- sys.stdout interface --------------------------------------------------

    def write(self, msg: str):
        self._terminal.write(msg)
        self._file.write(msg)

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    # -- verbose-only ---------------------------------------------------------

    def verbose(self, msg: str):
        """Write to log file only — not shown on terminal."""
        self._file.write(msg + "\n")
        self._file.flush()

    # -- context manager ------------------------------------------------------

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._terminal
        self._file.close()

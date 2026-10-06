"""
Atomic file writes: write to a temporary file in the same directory, fsync,
then os.replace() it over the target. A crash or a concurrent reader never
sees a half-written file.
"""
import json
import os
import tempfile
from contextlib import contextmanager

# mkstemp creates 0600 files; give the result the permissions a plain open()
# would have given it (umask read once at import, os.umask is process-wide).
_UMASK = os.umask(0)
os.umask(_UMASK)


def _target_mode(path: str) -> int:
    try:
        return os.stat(path).st_mode & 0o777
    except OSError:
        return 0o666 & ~_UMASK


@contextmanager
def atomic_open(path: str, mode: str = "w", encoding: str = "utf-8", newline=None):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        os.chmod(tmp, _target_mode(path))
        kwargs = {} if "b" in mode else {"encoding": encoding, "newline": newline}
        with os.fdopen(fd, mode, **kwargs) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: str, text: str) -> None:
    with atomic_open(path) as f:
        f.write(text)


def atomic_write_json(path: str, obj) -> None:
    with atomic_open(path) as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def atomic_to_csv(df, path: str) -> None:
    """DataFrame.to_csv(path, index=False), atomically."""
    with atomic_open(path, newline="") as f:
        df.to_csv(f, index=False)

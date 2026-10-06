"""
Progress line for task runners.

    format_progress(done=50, total=285, elapsed_s=6.8, errors=0)
    -> "  [ 50/285]  17%  7.4 q/s  ETA 00:32  errors: 0"

Below 1 item/s the rate is shown as seconds per item ("12.3 s/q") instead of
a misleading "0.0 q/s"; the ETA switches to h:mm:ss above one hour.
"""


def _fmt_eta(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "?"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_rate(done: int, elapsed_s: float) -> str:
    if done <= 0 or elapsed_s <= 0:
        return "? q/s"
    rate = done / elapsed_s
    if rate >= 1.0:
        return f"{rate:.1f} q/s"
    return f"{elapsed_s / done:.1f} s/q"


def format_progress(done: int, total: int, elapsed_s: float, errors: int = 0) -> str:
    """One progress line; *done*/*total* count only the items of this session."""
    width = len(str(total))
    pct = int(done / total * 100) if total > 0 else 100
    eta = (total - done) * elapsed_s / done if done > 0 and elapsed_s > 0 else None
    return (f"  [{done:>{width}}/{total}] {pct:3d}%  {format_rate(done, elapsed_s)}  "
            f"ETA {_fmt_eta(eta)}  errors: {errors}")

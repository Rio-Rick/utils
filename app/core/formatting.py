"""Display helpers shared by tools, history and the UI (no Qt, no I/O)."""

from __future__ import annotations

import math

_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def human_size(num_bytes: float) -> str:
    """``1536`` -> ``"1.5 KB"`` (1024 based, like the ``target size`` options)."""
    if not math.isfinite(num_bytes):
        return "-"
    size = abs(float(num_bytes))
    unit = _UNITS[0]
    for unit in _UNITS:
        if round(size, 1) < 1024 or unit == _UNITS[-1]:
            break
        size /= 1024
    text = f"{size:.0f}" if unit == "B" else f"{size:.1f}"
    return f"{'-' if num_bytes < 0 else ''}{text} {unit}"


def human_duration(seconds: float) -> str:
    """``0.42`` -> ``"420 ms"``, ``75`` -> ``"1 min 15 s"``, ``3725`` -> ``"1 h 02 min"``."""
    if not math.isfinite(seconds) or seconds < 0:
        return "-"
    if seconds < 1:
        return f"{round(seconds * 1000)} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    whole = int(round(seconds))
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def timecode(seconds: float, *, fraction: bool = False) -> str:
    """``3725.5`` -> ``"01:02:05"`` (``"01:02:05.500"`` with ``fraction``)."""
    seconds = max(0.0, seconds if math.isfinite(seconds) else 0.0)
    total_ms = int(round(seconds * 1000)) if fraction else int(seconds) * 1000
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base}.{millis:03d}" if fraction else base


def reduction_text(before: float, after: float) -> str:
    """Size change as text: ``"-72%"`` (smaller), ``"+12%"`` (bigger), ``"0%"``."""
    if before <= 0:
        return "-"
    change = (after / before - 1) * 100
    if abs(change) < 0.5:
        return "0%"
    return f"{change:+.0f}%".replace("-", "\u2212")

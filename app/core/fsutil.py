"""Safe file helpers shared by configuration, presets and exports."""

from __future__ import annotations

import contextlib
import os
import secrets
import tempfile
from collections.abc import Iterator
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    """Write ``data`` to ``path`` so readers never observe a half-written file.

    The data goes to a temporary file in the same directory, is flushed to disk and then
    moved over the target with :func:`os.replace` (atomic on POSIX).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def atomic_write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    """UTF-8 text variant of :func:`atomic_write_bytes`."""
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


@contextlib.contextmanager
def part_file(final: Path) -> Iterator[Path]:
    """Yield a temporary sibling path; on success it replaces ``final``, on error it is removed.

    Used by engines that write through a library which wants a *path* (Pillow, FFmpeg).
    The temporary name keeps the real extension so format detection still works.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL + default mode: the kernel applies the user's umask, unlike tempfile.mkstemp (0600).
    while True:
        tmp = final.with_name(f".{final.stem[:100]}.{secrets.token_hex(4)}.part{final.suffix}")
        try:
            os.close(os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
            break
        except FileExistsError:
            continue
    try:
        yield tmp
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise

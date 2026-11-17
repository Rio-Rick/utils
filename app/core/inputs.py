"""Turning dropped files and folders into a de-duplicated, naturally sorted list of input files."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

_NATURAL_RE = re.compile(r"(\d+)")


def natural_key(text: str) -> list[tuple[int, int | str]]:
    """Sort key where ``img2`` comes before ``img10`` (case-insensitive)."""
    return [(0, int(part)) if part.isdigit() else (1, part.casefold()) for part in _NATURAL_RE.split(text) if part]


@dataclass(slots=True)
class ScanResult:
    """Outcome of :func:`collect_files`.

    ``files`` holds ``(file, base_dir)`` pairs; ``base_dir`` is the dropped folder the file
    was found in (``None`` for files that were added directly).
    """

    files: list[tuple[Path, Path | None]] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    ignored: int = 0  # files of other types found inside dropped folders (not worth listing one by one)
    truncated: bool = False

    @property
    def paths(self) -> list[Path]:
        return [path for path, _ in self.files]


def extension_of(path: Path) -> str:
    """Lower-case extension without the dot (``''`` when there is none)."""
    return path.suffix[1:].lower()


def collect_files(
    paths: Iterable[Path | str],
    *,
    extensions: Iterable[str] | None = None,
    recursive: bool = True,
    limit: int = 0,
    include_hidden: bool = False,
    skip_dirs: Iterable[str] = (),
    cancelled: Callable[[], bool] | None = None,
) -> ScanResult:
    """Expand files and folders into the input files to process.

    Folders are walked without following symbolic links (no loops, no escaping the dropped
    folder); hidden entries and ``skip_dirs`` (typically the output sub-folder) are ignored,
    as are files of other types inside folders (counted in ``ignored``). Explicitly added
    files are kept even if hidden, but must match ``extensions`` (empty/None means any file)
    or they are reported in ``skipped``. ``limit`` stops the scan after that many files
    (0 = unlimited).
    """
    wanted = frozenset(e.lower().lstrip(".") for e in extensions) if extensions else frozenset()
    skip = frozenset(skip_dirs)
    result = ScanResult()
    seen: set[Path] = set()

    def matches(path: Path) -> bool:
        return not wanted or extension_of(path) in wanted

    def full() -> bool:
        if limit and len(result.files) >= limit:
            result.truncated = True
            return True
        return False

    def accept(path: Path, base: Path | None) -> None:
        try:
            key = path.resolve()
        except (OSError, RuntimeError):
            key = path.absolute()
        if key in seen:
            return
        seen.add(key)
        result.files.append((path, base))

    for raw in paths:
        if (cancelled and cancelled()) or full():
            break
        path = Path(os.path.expanduser(str(raw)))
        try:
            if path.is_dir():
                _walk(path, accept, matches, recursive, include_hidden, skip, full, cancelled, result)
            elif path.is_file():
                if matches(path):
                    accept(path, None)
                else:
                    result.skipped.append((path, "unsupported file type"))
            else:
                result.skipped.append((path, "not found"))
        except OSError as exc:
            result.skipped.append((path, exc.strerror or "cannot be read"))
    return result


def _walk(
    root: Path,
    accept: Callable[[Path, Path | None], None],
    matches: Callable[[Path], bool],
    recursive: bool,
    include_hidden: bool,
    skip: frozenset[str],
    full: Callable[[], bool],
    cancelled: Callable[[], bool] | None,
    result: ScanResult,
) -> None:
    def on_error(error: OSError) -> None:
        result.skipped.append((Path(error.filename or root), error.strerror or "cannot be read"))

    for current, dirs, files in os.walk(root, followlinks=False, onerror=on_error):
        if cancelled and cancelled():
            return
        dirs[:] = sorted(
            (d for d in dirs if d not in skip and (include_hidden or not d.startswith("."))),
            key=natural_key,
        ) if recursive else []
        for number, name in enumerate(sorted(files, key=natural_key)):
            if cancelled and number % 32 == 0 and cancelled():
                return
            if not include_hidden and name.startswith("."):
                continue
            candidate = Path(current) / name
            if not matches(candidate):
                result.ignored += 1
                continue
            if full():
                return
            if candidate.is_symlink() and not candidate.exists():
                continue  # dangling link
            accept(candidate, root)

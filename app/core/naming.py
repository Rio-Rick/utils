"""Where a result is written and what it is called.

One implementation of the output rules (location, naming template, overwrite policy)
shared by every tool, so behaviour and safety checks are never duplicated:

* the template language is tiny and safe - ``{name}``, ``{index:03d}`` ... - no attribute
  access, no code;
* file names are sanitised (no path separators, control characters, over-long names);
* "rename" reserves the target atomically (``O_EXCL``) so parallel workers never pick the
  same name, and the source file can never be overwritten.
"""

from __future__ import annotations

import contextlib
import os
import re
import string
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from app.core.errors import ToolError

Overwrite = Literal["rename", "overwrite", "skip"]
Location = Literal["subfolder", "same", "custom"]

MAX_NAME_BYTES = 200
_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PAD_RE = re.compile(r"^0?\d{1,2}d?$")
_WINDOWS_RESERVED = frozenset({"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)})
STANDARD_FIELDS = ("name", "ext", "index", "date", "time", "datetime", "parent")


def sanitize_filename(name: str, fallback: str = "output") -> str:
    """Make ``name`` safe as a single path component (no separators/control characters, bounded length)."""
    cleaned = unicodedata.normalize("NFC", name)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() and ch not in '/\\\x00<>:"|?*')
    cleaned = cleaned.strip().strip(".").strip()
    if cleaned.lower() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    encoded = cleaned.encode("utf-8")
    if len(encoded) > MAX_NAME_BYTES:
        cleaned = encoded[:MAX_NAME_BYTES].decode("utf-8", "ignore").rstrip()
    return cleaned or fallback


def render_template(template: str, fields: Mapping[str, Any]) -> str:
    """Fill ``{field}`` / ``{field:03d}`` placeholders. Unknown fields raise :class:`ToolError`."""
    parts: list[str] = []
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ToolError(f"Invalid naming template: {exc}", hint="Use fields such as {name}, {index:03d}, {date}.") from exc
    for literal, field_name, spec, conversion in parsed:
        parts.append(literal)
        if field_name is None:
            continue
        if conversion or not _FIELD_RE.match(field_name):
            raise ToolError(f"Naming template field {{{field_name}}} is not allowed", hint=_available(fields))
        if field_name not in fields:
            raise ToolError(f"Unknown naming template field {{{field_name}}}", hint=_available(fields))
        value = fields[field_name]
        if spec:
            if isinstance(value, int) and not isinstance(value, bool) and _PAD_RE.match(spec):
                parts.append(format(value, spec if spec.endswith("d") else spec + "d"))
                continue
            raise ToolError(f"Format '{spec}' is not allowed for {{{field_name}}}", hint="Only zero padding such as {index:03d} is supported.")
        parts.append(str(value))
    return "".join(parts)


def _available(fields: Mapping[str, Any]) -> str:
    return "Available fields: " + ", ".join("{" + key + "}" for key in fields)


@dataclass(slots=True)
class OutputTarget:
    """A planned output file. Use as a context manager so a failed item leaves no empty placeholder."""

    path: Path
    skip: bool = False
    reserved: bool = False

    def __enter__(self) -> OutputTarget:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self.discard()

    def discard(self) -> None:
        """Remove the placeholder created by the reservation when nothing was written into it."""
        if self.reserved:
            with contextlib.suppress(OSError):
                if self.path.stat().st_size == 0:
                    self.path.unlink()
        self.reserved = False


@dataclass(frozen=True, slots=True)
class OutputPlanner:
    """Turns ``(source, extension, fields)`` into a free output path according to the user's rules."""

    location: Location = "subfolder"
    subfolder_name: str = "converted"
    directory: str = ""
    fallback_directory: str = ""
    template: str = "{name}"
    overwrite: Overwrite = "rename"
    preserve_structure: bool = True

    @classmethod
    def from_options(cls, options: Mapping[str, Any], *, fallback_directory: str = "") -> OutputPlanner:
        """Build from resolved ``output.*`` options (``processing.output`` keys without the prefix)."""
        return cls(
            location=options.get("output.location", "subfolder"),
            subfolder_name=sanitize_filename(str(options.get("output.subfolder_name", "converted")), "converted"),
            directory=str(options.get("output.directory", "") or ""),
            fallback_directory=fallback_directory,
            template=str(options.get("output.naming_template", "{name}") or "{name}"),
            overwrite=options.get("output.overwrite", "rename"),
            preserve_structure=bool(options.get("output.preserve_structure", True)),
        )

    # ------------------------------------------------------------------ directories
    def output_dir(self, source: Path | None, base_dir: Path | None = None) -> Path:
        """Directory results for ``source`` go to (not created yet)."""
        source_dir = source.parent if source is not None else Path.cwd()
        relative = Path()
        if self.preserve_structure and base_dir is not None and source is not None:
            with contextlib.suppress(ValueError):
                relative = source.parent.relative_to(base_dir)
        if self.location == "same":
            return source_dir
        if self.location == "custom":
            chosen = self.directory or self.fallback_directory
            if chosen:
                return Path(os.path.expanduser(chosen)) / relative
            return source_dir
        # "subfolder": a single folder at the root of a dropped folder, otherwise next to each file
        if self.preserve_structure and base_dir is not None and source is not None:
            return base_dir / self.subfolder_name / relative
        return source_dir / self.subfolder_name

    # ------------------------------------------------------------------ files
    def plan(
        self,
        source: Path | None,
        extension: str,
        fields: Mapping[str, Any] | None = None,
        *,
        base_dir: Path | None = None,
        index: int = 1,
        now: datetime | None = None,
    ) -> OutputTarget:
        """Pick (and under ``rename`` reserve) the output path for one result.

        ``extension`` has no leading dot. ``fields`` adds tool specific template fields
        (``width``, ``format``, ``page`` ...) to the standard ones.
        """
        out_dir = self.output_dir(source, base_dir)
        stamp = now or datetime.now()
        values: dict[str, Any] = {
            "name": source.stem if source is not None else "output",
            "ext": extension,
            "index": index,
            "date": stamp.strftime("%Y-%m-%d"),
            "time": stamp.strftime("%H%M%S"),
            "datetime": stamp.strftime("%Y%m%d-%H%M%S"),
            "parent": source.parent.name if source is not None else "",
        }
        values.update(fields or {})
        stem = sanitize_filename(render_template(self.template, values))
        suffix = f".{extension.lstrip('.')}" if extension else ""
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(f"Cannot create the output folder {out_dir}: {exc.strerror or exc}", hint="Check permissions or choose another output folder.") from exc
        return self._claim(out_dir, stem, suffix, source)

    def sibling_dir(self, source: Path | None, name: str, *, base_dir: Path | None = None) -> Path:
        """A fresh folder ``<output dir>/<name>`` (e.g. for extracted frames), unique unless overwriting."""
        out_dir = self.output_dir(source, base_dir)
        stem = sanitize_filename(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate = out_dir / stem
        if self.overwrite == "overwrite":
            candidate.mkdir(exist_ok=True)
            return candidate
        counter = 0
        while True:
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                counter += 1
                candidate = out_dir / f"{stem} ({counter})"

    def _claim(self, out_dir: Path, stem: str, suffix: str, source: Path | None) -> OutputTarget:
        source_real = _real(source) if source is not None else None
        candidate = out_dir / f"{stem}{suffix}"
        counter = 0
        while True:
            is_source = source_real is not None and _real(candidate) == source_real
            exists = candidate.exists() or candidate.is_symlink()
            if self.overwrite == "overwrite" and not is_source:
                if candidate.is_dir():
                    raise ToolError(f"{candidate.name} is a folder; cannot write a file with that name")
                return OutputTarget(candidate)
            if self.overwrite == "skip" and exists and not is_source:
                return OutputTarget(candidate, skip=True)
            if not exists:
                try:
                    fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
                except FileExistsError:
                    pass  # lost a race with another worker: try the next number
                except OSError as exc:
                    raise ToolError(f"Cannot create {candidate}: {exc.strerror or exc}", hint="Check permissions and free disk space.") from exc
                else:
                    os.close(fd)
                    return OutputTarget(candidate, reserved=True)
            counter += 1
            candidate = out_dir / f"{stem} ({counter}){suffix}"
            if counter > 100_000:
                raise ToolError(f"Could not find a free file name for {stem}{suffix}")


def _real(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()

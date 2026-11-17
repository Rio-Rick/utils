"""The tool abstraction every utility implements.

A *tool* is a small, self-contained unit: metadata (id, name, category, required
dependencies), its configurable options (as :class:`~app.core.config_schema.Setting`
objects, so the settings forms and the documentation are generated from them) and its
processing logic. Tools never touch the GUI; the UI renders any tool from this
description and runs it through the job manager.

Adding a utility::

    @register_tool
    class MyTool(BatchTool):
        id = "image.my_tool"
        name = "My tool"
        description = "..."
        category = Category.IMAGE
        config_domain = "image"
        config_section = "my_tool"
        options = (Setting("strength", Kind.INT, 5, "Strength", minimum=0, maximum=10),)

        def run_item(self, item, options, ctx):
            ...
            return ItemResult(outputs=[out])
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from app.core import inputs
from app.core.config_defs import PROCESSING_SETTINGS
from app.core.config_schema import Kind, Setting
from app.core.control import JobControl
from app.core.errors import ToolError

if TYPE_CHECKING:
    from app.core.services import Services

TOOL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


class Category(StrEnum):
    """Sidebar groups. The values double as navigation route names."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"
    OTHER = "other"

    @property
    def title(self) -> str:
        return CATEGORY_TITLES[self]


CATEGORY_TITLES: dict[Category, str] = {
    Category.IMAGE: "Image Tools",
    Category.VIDEO: "Video Tools",
    Category.AUDIO: "Audio Tools",
    Category.FILE: "File Tools",
    Category.OTHER: "Other Utilities",
}


class InputMode(StrEnum):
    """How dropped files map to work items."""

    FILES = "files"   # one item per file (batch)
    MERGE = "merge"   # all files together form a single item (merge, image -> PDF)
    NONE = "none"     # the tool needs no input files


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One unit of work: usually one input file."""

    source: Path | None = None
    base_dir: Path | None = None
    label: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Short display name."""
        if self.label:
            return self.label
        return self.source.name if self.source else "-"

    @property
    def sources(self) -> tuple[Path, ...]:
        """All input files (several for merge items)."""
        extra = self.extra.get("sources")
        if extra:
            return tuple(Path(p) for p in extra)
        return (self.source,) if self.source else ()


@dataclass(slots=True)
class ItemResult:
    """What a tool reports for one finished work item."""

    outputs: list[Path] = field(default_factory=list)
    message: str = ""
    skipped: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    report: list[tuple[str, str]] = field(default_factory=list)  # label/value rows for inspector tools

    @property
    def primary_output(self) -> Path | None:
        return self.outputs[0] if self.outputs else None


@dataclass(slots=True)
class RunContext:
    """Per-item execution context handed to :meth:`BatchTool.run_item`."""

    services: Services
    control: JobControl = field(default_factory=JobControl)
    on_progress: Callable[[float, str], None] | None = None
    on_log: Callable[[str], None] | None = None
    label: str = "job"
    _temp: Path | None = field(default=None, init=False, repr=False)

    @property
    def config(self) -> Any:
        return self.services.config

    def progress(self, fraction: float, message: str = "") -> None:
        """Report progress (0..1); also a cancellation point."""
        self.control.check()
        if self.on_progress:
            self.on_progress(min(1.0, max(0.0, fraction)), message)

    def log(self, text: str) -> None:
        """Append a line to the job log."""
        if self.on_log:
            self.on_log(text)

    def temp_dir(self) -> Path:
        """A private scratch directory, removed by :meth:`cleanup` unless the user disabled cleanup."""
        if self._temp is None:
            configured = str(self.services.config.get("app", "directories.temp", "") or "")
            base = Path(os.path.expanduser(configured)) if configured else None
            if base is not None:
                base.mkdir(parents=True, exist_ok=True)
            self._temp = Path(tempfile.mkdtemp(prefix="lmt-", dir=base))
        return self._temp

    def cleanup(self) -> None:
        """Remove the scratch directory (honours ``processing.cleanup_temp``)."""
        if self._temp is not None and self.services.config.get("processing", "cleanup_temp", True):
            shutil.rmtree(self._temp, ignore_errors=True)
        self._temp = None


def output_settings() -> tuple[Setting, ...]:
    """The shared output options, each inheriting the matching ``processing.output.*`` default."""
    return tuple(
        dataclasses.replace(setting, inherit=f"processing.{setting.key}")
        for setting in PROCESSING_SETTINGS
        if setting.key.startswith("output.")
    )


class Tool:
    """Metadata and options common to every tool. Subclass :class:`BatchTool`."""

    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    category: ClassVar[Category]
    icon: ClassVar[str] = "tool"
    keywords: ClassVar[tuple[str, ...]] = ()
    config_domain: ClassVar[str]
    config_section: ClassVar[str] = ""
    config_title: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()
    options: ClassVar[tuple[Setting, ...]] = ()
    ui_page: ClassVar[str] = ""          # "package.module:Class" of a custom page; empty = generic page
    order: ClassVar[int] = 100
    phase: ClassVar[int] = 1

    # ------------------------------------------------------------------ options
    @classmethod
    def option_map(cls) -> dict[str, Setting]:
        """Option definitions by relative key."""
        return {s.key: s for s in cls.options}

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """Built-in defaults (without configuration files)."""
        return {s.key: s.default_value() for s in cls.options}

    def resolve_options(
        self,
        services: Services,
        *,
        preset: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Effective options: config files, then ``advanced.tool_overrides``, then preset, then ``overrides``."""
        values = {
            key: value
            for key, value in services.config.section(self.config_domain, self.config_section).items()
            if key in self.option_map()
        }
        tool_values, _issues = services.config.validate_tool_overrides(self.id, self.options)
        values.update(tool_values)
        if preset:
            values.update({k: v for k, v in preset.items() if k in self.option_map()})
        if overrides:
            values.update(overrides)
        return self.validate_options(values)

    def validate_options(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Coerce every option through its definition; unknown keys are dropped. Raises :class:`ToolError`."""
        cleaned: dict[str, Any] = {}
        problems: list[str] = []
        for setting in self.options:
            raw = values.get(setting.key, setting.default)
            try:
                cleaned[setting.key] = setting.coerce(raw)
            except ValueError as exc:
                problems.append(f"{setting.label}: {exc}")
        if problems:
            raise ToolError("Invalid settings", detail="\n".join(problems), hint="Fix the highlighted settings and try again.")
        self.check_options(cleaned)
        return cleaned

    def check_options(self, options: Mapping[str, Any]) -> None:
        """Cross-field validation hook; raise :class:`ToolError` when the combination is invalid."""

    @classmethod
    def available_choices(cls, key: str) -> tuple[str, ...] | None:
        """Choices of option ``key`` that work on this system (``None`` = all declared choices)."""
        return None

    def settings_for_config(self) -> tuple[Setting, ...]:
        """Definitions to register in the configuration domain (already relative to ``config_section``)."""
        return self.options


class BatchTool(Tool):
    """A tool that turns input files into output files (or a report) one item at a time."""

    input_mode: ClassVar[InputMode] = InputMode.FILES
    input_extensions: ClassVar[frozenset[str]] = frozenset()  # lower-case, no dot; empty = any file
    accepts_folders: ClassVar[bool] = True
    preview: ClassVar[str] = "none"      # "compare" | "frames" | "none"
    results_kind: ClassVar[str] = "files"  # "files" | "report"
    file_hint: ClassVar[str] = ""

    def supported_extensions(self) -> frozenset[str]:
        """Extensions (lower-case, no dot) this tool accepts; empty = any file. Override when it depends on the system."""
        return self.input_extensions

    def expand_inputs(
        self,
        paths: Sequence[Path | str],
        options: Mapping[str, Any] | None = None,
        *,
        limit: int = 0,
        cancelled: Callable[[], bool] | None = None,
    ) -> inputs.ScanResult:
        """Files and folders -> the files this tool can process (the UI shows ``skipped`` to the user)."""
        skip_dirs: list[str] = []
        if options and options.get("output.location") == "subfolder":
            skip_dirs.append(str(options.get("output.subfolder_name") or "converted"))
        return inputs.collect_files(
            paths,
            extensions=self.supported_extensions() or None,
            recursive=self.accepts_folders,
            limit=limit,
            skip_dirs=skip_dirs,
            cancelled=cancelled,
        )

    def make_items(self, scan: inputs.ScanResult, options: Mapping[str, Any]) -> list[WorkItem]:
        """Scan result -> work items. Override for tools that group inputs (merge, per-page)."""
        if self.input_mode is InputMode.MERGE and scan.files:
            return [WorkItem(scan.files[0][0], scan.files[0][1], f"{len(scan.files)} files", {"sources": [str(p) for p in scan.paths]})]
        return [WorkItem(path, base, extra={"index": number}) for number, (path, base) in enumerate(scan.files, start=1)]

    def run_item(self, item: WorkItem, options: Mapping[str, Any], ctx: RunContext) -> ItemResult:
        """Process one item. Raise :class:`ToolError` for expected failures and honour ``ctx.control``."""
        raise NotImplementedError

    def make_preview(self, source: Path, options: Mapping[str, Any], services: Services, control: JobControl) -> object | None:
        """Build the preview shown next to the options (``preview = "compare"``: an :class:`~app.engines.image_engine.Preview`).

        Runs on a worker thread; return ``None`` when the tool has nothing to show.
        """
        return None

    def idle_summary(self) -> str:
        """One-line hint shown in the empty drop area."""
        if self.file_hint:
            return self.file_hint
        extensions = self.supported_extensions()
        if extensions:
            return "Supported: " + ", ".join(sorted(extensions))
        return "Drop any files here"


_T = TypeVar("_T", bound=type[Tool])
_REGISTRY: dict[str, type[Tool]] = {}


def register_tool(cls: _T) -> _T:
    """Class decorator adding a tool to the global registry (picked up by the plugin manager)."""
    tool_id = getattr(cls, "id", "")
    if not TOOL_ID_RE.match(tool_id):
        raise ValueError(f"invalid tool id {tool_id!r}: use lower-case 'category.name'")
    for attr in ("name", "description", "category", "config_domain"):
        if not hasattr(cls, attr):
            raise ValueError(f"tool {tool_id} is missing the attribute {attr!r}")
    existing = _REGISTRY.get(tool_id)
    if existing is not None and existing.__module__ != cls.__module__:
        raise ValueError(f"tool id {tool_id!r} is already registered by {existing.__module__}")
    _REGISTRY[tool_id] = cls
    return cls


def registered_tool_classes() -> dict[str, type[Tool]]:
    """A copy of the global registry."""
    return dict(_REGISTRY)


def unregister_tool(tool_id: str) -> None:
    """Remove a tool (used when a plugin fails half-way through loading, and by tests)."""
    with contextlib.suppress(KeyError):
        del _REGISTRY[tool_id]


__all__ = [
    "BatchTool", "Category", "InputMode", "ItemResult", "Kind", "RunContext", "Setting", "Tool",
    "WorkItem", "output_settings", "register_tool", "registered_tool_classes", "unregister_tool",
]

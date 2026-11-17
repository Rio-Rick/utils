"""Discovery and registry of tools (built-in, user plugins, pip-installed plugins).

Loading is defensive: a plugin that fails to import is recorded in
:attr:`PluginManager.failures` (shown in *Settings*), never raised, and the tools it
registered before failing are removed again.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import logging
import pkgutil
import sys
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app import paths
from app.tools import base as tool_base
from app.tools.base import Category, Tool

if TYPE_CHECKING:
    from app.core.config_manager import ConfigManager
    from app.core.dependency_manager import DependencyManager

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "linux_media_toolkit.tools"
BUILTIN_PACKAGE = "app.tools"


@dataclass(frozen=True, slots=True)
class PluginFailure:
    """A module that could not be loaded."""

    source: str
    error: str


@dataclass(frozen=True, slots=True)
class Availability:
    """Whether a tool can run now, and what is missing if not."""

    ok: bool
    missing: tuple[str, ...] = ()
    message: str = ""


class PluginManager:
    """Registry of tool instances with ordering, enable/disable and availability checks."""

    def __init__(self, config: ConfigManager | None = None, deps: DependencyManager | None = None) -> None:
        self._config = config
        self._deps = deps
        self._tools: dict[str, Tool] = {}
        self._origin: dict[str, str] = {}
        self.failures: list[PluginFailure] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ discovery
    def discover(self, *, builtin: bool = True, user_plugins: bool | None = None, entry_points: bool | None = None) -> None:
        """Import all tool modules and instantiate the registered tools.

        ``user_plugins`` / ``entry_points`` default to the ``plugins.*`` configuration.
        """
        with self._lock:
            if builtin:
                self._import_builtin()
            if user_plugins is None:
                user_plugins = bool(self._config.get("plugins", "user_plugins.enabled", True)) if self._config else True
            if entry_points is None:
                entry_points = bool(self._config.get("plugins", "entry_points.enabled", True)) if self._config else True
            if user_plugins:
                self._import_user_plugins()
            if entry_points:
                self._import_entry_points()
            self._instantiate()
            self._register_settings()

    def _import_builtin(self) -> None:
        package = importlib.import_module(BUILTIN_PACKAGE)

        def on_error(name: str) -> None:
            log.exception("failed to import tool package %s", name)
            self._record(name, "import failed (see the log)")

        for info in pkgutil.walk_packages(package.__path__, prefix=f"{BUILTIN_PACKAGE}.", onerror=on_error):
            if info.name.rsplit(".", 1)[-1] in {"base"}:
                continue
            before = set(tool_base.registered_tool_classes())
            try:
                importlib.import_module(info.name)
            except Exception as exc:  # a broken built-in module must not take the app down
                log.exception("failed to load tool module %s", info.name)
                self._fail(info.name, exc, before)

    def _plugin_dir(self) -> Path:
        configured = str(self._config.get("plugins", "user_plugins.directory", "") or "") if self._config else ""
        return Path(configured).expanduser() if configured else paths.user_plugins_dir()

    def _import_user_plugins(self) -> None:
        directory = self._plugin_dir()
        if not directory.is_dir():
            return
        for file in sorted(directory.glob("*.py")):
            if file.name.startswith(("_", ".")):
                continue
            module_name = f"lmt_user_plugin_{file.stem}"
            before = set(tool_base.registered_tool_classes())
            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load {file}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                for tool_id in set(tool_base.registered_tool_classes()) - before:
                    self._origin[tool_id] = f"plugin:{file.name}"
            except Exception as exc:
                sys.modules.pop(module_name, None)
                log.warning("user plugin %s failed to load: %s", file.name, exc)
                self._fail(str(file), exc, before)

    def _import_entry_points(self) -> None:
        try:
            entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception as exc:
            log.warning("cannot read entry points: %s", exc)
            return
        for entry in entries:
            before = set(tool_base.registered_tool_classes())
            try:
                entry.load()
                for tool_id in set(tool_base.registered_tool_classes()) - before:
                    self._origin[tool_id] = f"package:{entry.name}"
            except Exception as exc:
                log.warning("plugin entry point %s failed: %s", entry.name, exc)
                self._fail(f"entry point {entry.name}", exc, before)

    def _record(self, source: str, error: str) -> None:
        failure = PluginFailure(source, error)
        if failure not in self.failures:
            self.failures.append(failure)

    def _fail(self, source: str, exc: BaseException, before: set[str]) -> None:
        for tool_id in set(tool_base.registered_tool_classes()) - before:
            tool_base.unregister_tool(tool_id)
        self._record(source, f"{type(exc).__name__}: {exc}")

    def _instantiate(self) -> None:
        for tool_id, cls in sorted(tool_base.registered_tool_classes().items()):
            if tool_id in self._tools:
                continue
            try:
                self._tools[tool_id] = cls()
                self._origin.setdefault(tool_id, "builtin")
            except Exception as exc:
                log.warning("tool %s cannot be created: %s", tool_id, exc)
                self._record(tool_id, f"{type(exc).__name__}: {exc}")

    def _register_settings(self) -> None:
        """Make every tool's options part of the configuration (YAML files, Settings page, docs)."""
        if self._config is None:
            return
        for tool in self._tools.values():
            if getattr(tool, "_settings_registered", False):
                continue
            try:
                self._config.register_domain(tool.config_domain, tool.settings_for_config(), section=tool.config_section)
                tool._settings_registered = True  # type: ignore[attr-defined]
            except Exception as exc:  # conflicting plugin definitions
                log.warning("settings of %s rejected: %s", tool.id, exc)
                self._record(tool.id, f"settings: {exc}")

    def register(self, tool: Tool, origin: str = "runtime") -> None:
        """Add an already created tool instance (tests, embedding)."""
        with self._lock:
            self._tools[tool.id] = tool
            self._origin[tool.id] = origin
            self._register_settings()

    # ------------------------------------------------------------------ queries
    def get(self, tool_id: str) -> Tool | None:
        return self._tools.get(tool_id)

    def origin(self, tool_id: str) -> str:
        return self._origin.get(tool_id, "unknown")

    def domain_titles(self) -> dict[str, str]:
        """``{config domain: title}`` contributed by tools."""
        titles: dict[str, str] = {}
        for tool in self._tools.values():
            if tool.config_title and tool.config_domain not in titles:
                titles[tool.config_domain] = tool.config_title
        return titles

    def disabled_ids(self) -> set[str]:
        return {str(x) for x in self._config.get("plugins", "disabled", [])} if self._config else set()

    def favorites(self) -> list[str]:
        if not self._config:
            return []
        return [t for t in self._config.get("plugins", "favorites", []) if t in self._tools]

    def all_tools(self, *, include_disabled: bool = False) -> list[Tool]:
        """Tools in display order: ``plugins.order`` first, then by ``order``/name."""
        configured = [str(x) for x in self._config.get("plugins", "order", [])] if self._config else []
        rank = {tool_id: index for index, tool_id in enumerate(configured)}
        disabled = set() if include_disabled else self.disabled_ids()
        tools = [t for t in self._tools.values() if t.id not in disabled]
        tools.sort(key=lambda t: (rank.get(t.id, len(rank)), t.order, t.name.casefold()))
        return tools

    def by_category(self, category: Category, *, include_disabled: bool = False) -> list[Tool]:
        return [t for t in self.all_tools(include_disabled=include_disabled) if t.category == category]

    def search(self, text: str) -> list[Tool]:
        """Tools whose name, description, id or keywords contain every word of ``text``."""
        words = text.casefold().split()
        if not words:
            return self.all_tools()
        found = []
        for tool in self.all_tools():
            haystack = " ".join((tool.name, tool.description, tool.id, *tool.keywords)).casefold()
            if all(word in haystack for word in words):
                found.append(tool)
        return found

    # ------------------------------------------------------------------ state changes
    def set_favorite(self, tool_id: str, favorite: bool) -> None:
        if self._config is None or tool_id not in self._tools:
            return
        current = [str(x) for x in self._config.get("plugins", "favorites", [])]
        if favorite and tool_id not in current:
            current.append(tool_id)
        elif not favorite and tool_id in current:
            current.remove(tool_id)
        self._config.set("plugins", "favorites", current)

    def set_enabled(self, tool_id: str, enabled: bool) -> None:
        if self._config is None:
            return
        disabled = [str(x) for x in self._config.get("plugins", "disabled", [])]
        if enabled and tool_id in disabled:
            disabled.remove(tool_id)
        elif not enabled and tool_id not in disabled:
            disabled.append(tool_id)
        self._config.set("plugins", "disabled", disabled)

    def set_order(self, tool_ids: Iterable[str]) -> None:
        if self._config is not None:
            self._config.set("plugins", "order", list(dict.fromkeys(tool_ids)))

    # ------------------------------------------------------------------ availability
    def availability(self, tool: Tool | str, *, probe: bool = False) -> Availability:
        """Are the tool's dependencies present?

        With ``probe=False`` (GUI thread) only cached results are used: dependencies that have
        not been checked yet count as available. Use ``probe=True`` from worker threads.
        """
        resolved = self._tools.get(tool) if isinstance(tool, str) else tool
        if resolved is None:
            return Availability(False, (), "Unknown tool")
        if self._deps is None or not resolved.requires:
            return Availability(True)
        missing: list[str] = []
        for requirement in resolved.requires:
            states = []
            for alternative in requirement.split("|"):
                if probe:
                    states.append(self._deps.available(alternative))
                else:
                    cached = self._deps.cached(alternative)
                    states.append(True if cached is None else cached.available)
            if not any(states):
                missing.append(requirement)
        if not missing:
            return Availability(True)
        return Availability(False, tuple(missing), "Missing: " + ", ".join(missing))

    def clear(self) -> None:
        with self._lock, contextlib.suppress(Exception):
            self._tools.clear()
            self._origin.clear()
            self.failures.clear()

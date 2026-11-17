"""The object graph shared by the UI, the job manager and the tools.

Created once at start-up (:meth:`Services.create`) and passed around explicitly, so there
are no module level singletons and every part can be tested with a temporary directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app import __version__, paths
from app.core.config_manager import ConfigManager
from app.core.dependency_manager import DependencyManager
from app.core.logger import configure_from_config
from app.core.plugin_manager import PluginManager

if TYPE_CHECKING:
    from app.core.history_manager import HistoryManager

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    """Configuration, dependency detection, tool registry and history."""

    config: ConfigManager
    deps: DependencyManager
    plugins: PluginManager
    history: HistoryManager | None = None

    @classmethod
    def create(
        cls,
        *,
        user_dir: Path | None = None,
        defaults_dir: Path | None = None,
        discover: bool = True,
        with_history: bool = True,
        configure_logging: bool = True,
        user_plugins: bool | None = None,
        entry_points: bool | None = None,
    ) -> Services:
        """Build the whole object graph from the standard (or given) locations."""
        config = ConfigManager(user_dir or paths.user_config_dir(), defaults_dir or paths.defaults_dir(), app_version=__version__)
        if configure_logging:
            configure_from_config(config)
        deps = DependencyManager(config)
        plugins = PluginManager(config, deps)
        if discover:
            plugins.discover(user_plugins=user_plugins, entry_points=entry_points)
        history = None
        if with_history:
            from app.core.history_manager import HistoryManager

            history = HistoryManager(paths.data_dir() / "history.sqlite3", config)
        return cls(config, deps, plugins, history)

    def close(self) -> None:
        """Release resources (history database)."""
        if self.history is not None:
            self.history.close()

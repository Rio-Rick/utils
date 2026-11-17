"""Filesystem locations used by the application.

Every path is resolved here so the rest of the code base never hard-codes a
location. Functions are stateless (they read the environment on each call) so
tests and packagers can redirect them with environment variables:

``LMT_CONFIG_DIR``    user configuration (overrides)
``LMT_DEFAULTS_DIR``  shipped default configuration
``LMT_DATA_DIR``      history database, user plugins
``LMT_STATE_DIR``     logs
``LMT_FORCE_XDG``     never use the portable ``config/user`` directory of a source checkout
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "linux-media-toolkit"


def project_root() -> Path:
    """Directory that contains the ``app`` package (the repository root in development)."""
    return Path(__file__).resolve().parent.parent


def resources_dir() -> Path:
    """Static resources shipped with the application (icons, translations)."""
    return Path(__file__).resolve().parent / "resources"


def builtin_presets_dir() -> Path:
    """Presets shipped with the application."""
    return Path(__file__).resolve().parent / "presets"


def _xdg_base(env_var: str, fallback: str) -> Path:
    value = os.environ.get(env_var, "")
    if value and os.path.isabs(value):
        return Path(value) / APP_DIR_NAME
    return Path.home() / fallback / APP_DIR_NAME


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(path, os.W_OK | os.X_OK)


def defaults_dir() -> Path | None:
    """Return the shipped default-configuration directory, or ``None`` when it cannot be found.

    The application still works without it because every setting also has a
    built-in default in the settings schema.
    """
    candidates: list[Path] = []
    override = os.environ.get("LMT_DEFAULTS_DIR")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(project_root() / "config" / "defaults")
    candidates.append(Path(sys.prefix) / "share" / APP_DIR_NAME / "config" / "defaults")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def user_config_dir() -> Path:
    """Directory holding the user's configuration overrides.

    A writable source checkout keeps them in ``config/user`` (portable
    development mode). Installed copies (AppImage, packages, pip) use
    ``$XDG_CONFIG_HOME/linux-media-toolkit``.
    """
    override = os.environ.get("LMT_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    portable = project_root() / "config" / "user"
    in_checkout = (project_root() / "config" / "defaults").is_dir()
    if in_checkout and not os.environ.get("LMT_FORCE_XDG") and _is_writable_dir(portable):
        return portable
    return _xdg_base("XDG_CONFIG_HOME", ".config")


def data_dir() -> Path:
    """Persistent application data (history database, user plugins)."""
    override = os.environ.get("LMT_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg_base("XDG_DATA_HOME", ".local/share")


def state_dir() -> Path:
    """Runtime state such as log files."""
    override = os.environ.get("LMT_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg_base("XDG_STATE_HOME", ".local/state")


def log_dir() -> Path:
    """Directory for rotating log files."""
    return state_dir() / "logs"


def user_plugins_dir() -> Path:
    """Default directory scanned for user-written tool plugins."""
    return data_dir() / "plugins"

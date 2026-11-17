"""Application logging.

Configured from the ``app.logging.*`` settings. The log never contains file
contents; by default the user's home directory is replaced by ``~`` and the
``basename`` mode reduces every absolute path to its file name.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from app import paths

if TYPE_CHECKING:
    from app.core.config_manager import ConfigManager

LOGGER_NAME = "app"
LOG_FILE_NAME = "linux-media-toolkit.log"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_ABS_PATH_RE = re.compile(r"(?<![\w:/.~-])/(?:[^\s'\"\\:;,()<>|*?\[\]{}]*)")
_HANDLER_MARK = "_lmt_handler"


class RedactingFilter(logging.Filter):
    """Removes personal path information from records before they are written."""

    def __init__(self, mode: str = "home", home: str | None = None) -> None:
        super().__init__()
        self.mode = mode
        self.home = home if home is not None else str(Path.home())

    def redact(self, text: str) -> str:
        """Return ``text`` with paths reduced according to the configured mode."""
        if self.mode == "home":
            if self.home and self.home != "/":
                return re.sub(re.escape(self.home) + r"(?![\w.-])", "~", text)
            return text
        if self.mode == "basename":
            return _ABS_PATH_RE.sub(_basename_of, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if self.mode == "none":
            return True
        try:
            message = record.getMessage()
        except Exception:  # a malformed log call must never break logging
            return True
        record.msg = self.redact(message)
        record.args = ()
        if record.exc_text:
            record.exc_text = self.redact(record.exc_text)
        return True


def _basename_of(match: re.Match[str]) -> str:
    path = match.group(0)
    name = os.path.basename(path.rstrip("/"))
    return name or path


def log_file_path(log_dir: Path | None = None) -> Path:
    """Location of the current log file."""
    return (log_dir or paths.log_dir()) / LOG_FILE_NAME


def configure_logging(
    level: str = "INFO",
    *,
    to_file: bool = True,
    log_dir: Path | None = None,
    max_size_mb: int = 5,
    backup_count: int = 3,
    redact: str = "home",
    console: bool = True,
) -> logging.Logger:
    """(Re)configure the ``app`` logger. Safe to call repeatedly."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()

    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)
    redactor = RedactingFilter(redact)

    def attach(handler: logging.Handler) -> None:
        handler.setFormatter(formatter)
        handler.addFilter(redactor)
        setattr(handler, _HANDLER_MARK, True)
        logger.addHandler(handler)

    if console:
        attach(logging.StreamHandler(sys.stderr))
    if to_file:
        target = log_file_path(log_dir)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            attach(
                logging.handlers.RotatingFileHandler(
                    target,
                    maxBytes=max(1, max_size_mb) * 1024 * 1024,
                    backupCount=max(0, backup_count),
                    encoding="utf-8",
                    delay=True,
                )
            )
        except OSError as exc:
            logger.warning("cannot write the log file %s: %s", target.name, exc)
    return logger


def configure_from_config(config: ConfigManager, *, console: bool = True) -> None:
    """Configure logging from ``app.logging.*`` and keep it in sync with later changes."""

    def apply() -> None:
        configure_logging(
            str(config.get("app", "logging.level")),
            to_file=bool(config.get("app", "logging.to_file")),
            max_size_mb=int(config.get("app", "logging.max_size_mb")),
            backup_count=int(config.get("app", "logging.backup_count")),
            redact=str(config.get("app", "logging.redact_paths")),
            console=console,
        )

    apply()

    def on_change(domain: str, keys: frozenset[str]) -> None:
        if domain == "app" and any(key.startswith("logging.") for key in keys):
            apply()

    config.subscribe(on_change)

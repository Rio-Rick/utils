"""Tests for logging configuration and path redaction."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.config_manager import ConfigManager
from app.core.logger import LOGGER_NAME, RedactingFilter, configure_from_config, configure_logging, log_file_path


def ours(logger: logging.Logger) -> list[logging.Handler]:
    """Handlers installed by configure_logging (pytest adds its own capture handlers)."""
    return [h for h in logger.handlers if getattr(h, "_lmt_handler", False)]


@pytest.fixture(autouse=True)
def reset_logger() -> Iterator[None]:
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for handler in ours(logger):
        logger.removeHandler(handler)
        handler.close()


class TestRedaction:
    def test_home_mode(self) -> None:
        f = RedactingFilter("home", home="/home/alice")
        assert f.redact("open /home/alice/Videos/a.mp4") == "open ~/Videos/a.mp4"
        assert f.redact("/home/alice2/x") == "/home/alice2/x"  # a different user is left alone
        assert f.redact("nothing here") == "nothing here"

    def test_basename_mode(self) -> None:
        f = RedactingFilter("basename", home="/home/alice")
        assert f.redact("convert /home/alice/Pictures/holiday.png -> /tmp/out/x.webp done") == "convert holiday.png -> x.webp done"
        assert f.redact("ratio 3/4 and http://example.com/a") == "ratio 3/4 and http://example.com/a"

    def test_none_mode_and_root_home(self) -> None:
        assert RedactingFilter("none").redact("/home/alice/a") == "/home/alice/a"
        assert RedactingFilter("home", home="/").redact("/etc/passwd") == "/etc/passwd"

    def test_filter_rewrites_args_and_exceptions(self) -> None:
        f = RedactingFilter("home", home="/home/alice")
        try:
            raise ValueError("bad /home/alice/secret.txt")
        except ValueError:
            import sys

            record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed %s", ("/home/alice/a.png",), sys.exc_info())
        logging.Formatter().format(record)  # populates exc_text like a handler would
        assert f.filter(record)
        assert record.getMessage() == "failed ~/a.png"
        assert "/home/alice" not in (record.exc_text or "")


class TestConfigure:
    def test_writes_rotating_file_and_honours_level(self, tmp_path: Path) -> None:
        logger = configure_logging("WARNING", log_dir=tmp_path / "logs", console=False, redact="none")
        logging.getLogger("app.test").info("hidden")
        logging.getLogger("app.test").warning("visible")
        for handler in logger.handlers:
            handler.flush()
        text = log_file_path(tmp_path / "logs").read_text()
        assert "visible" in text and "hidden" not in text

    def test_reconfigure_replaces_handlers(self, tmp_path: Path) -> None:
        configure_logging(log_dir=tmp_path, console=True)
        logger = configure_logging(log_dir=tmp_path, console=True)
        assert len(ours(logger)) == 2

    def test_unwritable_directory_does_not_crash(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x")
        logger = configure_logging(log_dir=blocker / "logs", console=False)
        logging.getLogger("app.test").error("still works")
        assert not ours(logger)

    def test_follows_configuration_changes(self, user_dir: Path, tmp_path: Path) -> None:
        config = ConfigManager(user_dir)
        configure_from_config(config, console=False)
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.INFO
        config.set("app", "logging.level", "DEBUG")
        assert logger.level == logging.DEBUG
        config.set("app", "logging.to_file", False)
        assert not ours(logger)

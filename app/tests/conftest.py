"""Shared pytest fixtures.

Every test runs against a throw-away configuration/data/state directory so the
developer's real settings, history and logs are never touched.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Redirect every per-user location into ``tmp_path``."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LMT_CONFIG_DIR", str(tmp_path / "config-user"))
    monkeypatch.setenv("LMT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LMT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("LMT_DEFAULTS_DIR", raising=False)
    yield


@pytest.fixture
def user_dir(tmp_path: Path) -> Path:
    """The (initially empty) user configuration directory."""
    path = tmp_path / "config-user"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture(scope="session")
def qapp() -> Iterator[object]:
    """One QApplication for the whole test session (offscreen platform)."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["lmt-tests"])
    yield app


@pytest.fixture
def wait_until(qapp: object) -> Callable[..., bool]:
    """Pump the Qt event loop until ``predicate()`` is true or ``timeout`` seconds passed."""
    from PySide6.QtCore import QCoreApplication

    def _wait(predicate: Callable[[], bool], timeout: float = 10.0, interval: float = 0.005) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            QCoreApplication.processEvents()
            if predicate():
                return True
            time.sleep(interval)
        QCoreApplication.processEvents()
        return predicate()

    return _wait

"""Shared pytest fixtures.

Every test runs against a throw-away configuration/data/state directory so the
developer's real settings, history and logs are never touched.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.core.services import Services

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


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Restore the global tool registry (and plugin modules) after a test that registers tools."""
    import sys

    from app.tools import base

    saved = dict(base._REGISTRY)
    modules = set(sys.modules)
    yield
    base._REGISTRY.clear()
    base._REGISTRY.update(saved)
    for name in set(sys.modules) - modules:
        if name.startswith("lmt_user_plugin_"):
            del sys.modules[name]


@pytest.fixture
def services(user_dir: Path) -> Iterator[Services]:
    """A fully discovered object graph (built-in tools only) over the temporary directories."""
    from app.core.services import Services

    graph = Services.create(
        user_dir=user_dir, with_history=False, configure_logging=False, user_plugins=False, entry_points=False
    )
    yield graph
    graph.close()


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

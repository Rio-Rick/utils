from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app import paths
from app.core.plugin_manager import PluginManager
from app.core.services import Services
from app.tools.base import BatchTool, Category

GOOD_PLUGIN = '''
from app.core.config_schema import Kind, Setting
from app.tools.base import BatchTool, Category, register_tool


@register_tool
class Hello(BatchTool):
    id = "other.hello_plugin"
    name = "Hello plugin"
    description = "Says hello"
    category = Category.OTHER
    config_domain = "plugin_demo"
    config_section = "hello"
    options = (Setting("greeting", Kind.STR, "hi", "Greeting"),)
'''

HALF_LOADED_PLUGIN = '''
from app.tools.base import BatchTool, Category, register_tool


@register_tool
class Half(BatchTool):
    id = "other.half_loaded"
    name = "Half"
    description = "registered, then the module blows up"
    category = Category.OTHER
    config_domain = "plugin_demo"

raise RuntimeError("boom")
'''


def toy(tool_id: str, name: str, **attrs: Any) -> BatchTool:
    cls = type(
        "T" + tool_id.replace(".", "_"),
        (BatchTool,),
        {"id": tool_id, "name": name, "description": f"{name} description", "category": Category.OTHER,
         "config_domain": "toys", "config_section": tool_id.split(".")[-1], **attrs},
    )
    return cls()


def write_plugins(files: dict[str, str]) -> Path:
    directory = paths.user_plugins_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for name, code in files.items():
        (directory / name).write_text(code)
    return directory


# ----------------------------------------------------------------------------- discovery
def test_builtin_tools_are_discovered_and_their_settings_registered(services):
    tool = services.plugins.get("image.converter")
    assert tool is not None and services.plugins.origin("image.converter") == "builtin"
    assert services.config.get("image", "converter.quality") == 85
    assert "image" in services.plugins.domain_titles()
    assert services.plugins.failures == []
    assert tool in services.plugins.by_category(Category.IMAGE)
    assert services.plugins.by_category(Category.AUDIO) == [] or all(t.category == Category.AUDIO for t in services.plugins.by_category(Category.AUDIO))


def test_user_plugins_load_and_broken_ones_are_isolated(user_dir, clean_registry):
    write_plugins({
        "good.py": GOOD_PLUGIN,
        "half.py": HALF_LOADED_PLUGIN,
        "syntax.py": "def (:\n",
        "_private.py": "raise SystemExit('never imported')",
    })
    graph = Services.create(user_dir=user_dir, with_history=False, configure_logging=False, user_plugins=True, entry_points=False)
    try:
        plugins = graph.plugins
        assert plugins.get("other.hello_plugin") is not None
        assert plugins.origin("other.hello_plugin") == "plugin:good.py"
        assert graph.config.get("plugin_demo", "hello.greeting") == "hi"
        assert plugins.get("other.half_loaded") is None  # removed again after the failed import
        failed = {Path(f.source).name: f.error for f in plugins.failures}
        assert "boom" in failed["half.py"] and "SyntaxError" in failed["syntax.py"]
        assert "_private.py" not in failed
        assert plugins.get("image.converter") is not None  # one broken plugin never affects the rest
    finally:
        graph.close()


def test_user_plugins_can_be_turned_off_by_configuration(user_dir, clean_registry):
    write_plugins({"good.py": GOOD_PLUGIN})
    config_off = Services.create(user_dir=user_dir, with_history=False, configure_logging=False, entry_points=False)
    try:
        config_off.config.set("plugins", "user_plugins.enabled", False)
    finally:
        config_off.close()
    from app.tools import base

    base.unregister_tool("other.hello_plugin")
    again = Services.create(user_dir=user_dir, with_history=False, configure_logging=False, entry_points=False)
    try:
        assert again.plugins.get("other.hello_plugin") is None
    finally:
        again.close()


def test_discovery_twice_does_not_duplicate_failures(user_dir, clean_registry):
    write_plugins({"half.py": HALF_LOADED_PLUGIN})
    graph = Services.create(user_dir=user_dir, with_history=False, configure_logging=False, user_plugins=True, entry_points=False)
    try:
        graph.plugins.discover(user_plugins=True, entry_points=False)
        assert len([f for f in graph.plugins.failures if f.source.endswith("half.py")]) == 1
    finally:
        graph.close()


# ----------------------------------------------------------------------------- ordering and state
def test_order_disable_favorite_and_search(services):
    plugins = services.plugins
    plugins.register(toy("other.zeta", "Zeta", order=1, keywords=("shiny",)))
    plugins.register(toy("other.alpha", "Alpha", order=1))
    ids = [t.id for t in plugins.all_tools()]
    assert ids.index("other.alpha") < ids.index("other.zeta")  # same order value: by name

    plugins.set_order(["other.zeta", "other.alpha"])
    ids = [t.id for t in plugins.all_tools()]
    assert ids.index("other.zeta") < ids.index("other.alpha")

    plugins.set_enabled("other.alpha", False)
    assert "other.alpha" not in [t.id for t in plugins.all_tools()]
    assert "other.alpha" in [t.id for t in plugins.all_tools(include_disabled=True)]
    plugins.set_enabled("other.alpha", True)
    assert "other.alpha" in [t.id for t in plugins.all_tools()]

    plugins.set_favorite("other.zeta", True)
    plugins.set_favorite("other.zeta", True)
    plugins.set_favorite("does.not_exist", True)
    assert plugins.favorites() == ["other.zeta"]
    plugins.set_favorite("other.zeta", False)
    assert plugins.favorites() == []

    assert [t.id for t in plugins.search("shiny")] == ["other.zeta"]
    assert [t.id for t in plugins.search("alpha description")] == ["other.alpha"]
    assert plugins.search("nonexistent-thing") == []
    assert len(plugins.search("")) == len(plugins.all_tools())


def test_order_and_favorites_persist_in_the_config_file(services, user_dir: Path):
    services.plugins.register(toy("other.alpha", "Alpha"))
    services.plugins.set_favorite("other.alpha", True)
    assert "other.alpha" in (user_dir / "plugins.yaml").read_text()


# ----------------------------------------------------------------------------- availability
class FakeDeps:
    def __init__(self, installed: dict[str, bool], cached: dict[str, bool] | None = None) -> None:
        self.installed = installed
        self.cache = cached or {}
        self.probed: list[str] = []

    def cached(self, dep_id: str):
        return SimpleNamespace(available=self.cache[dep_id]) if dep_id in self.cache else None

    def available(self, dep_id: str) -> bool:
        self.probed.append(dep_id)
        return self.installed.get(dep_id, False)


def test_availability_uses_cache_and_never_probes_on_the_gui_thread():
    deps = FakeDeps({"ffmpeg": False}, cached={"ffmpeg": False})
    manager = PluginManager(None, deps)  # type: ignore[arg-type]
    tool = toy("video.needs_ffmpeg", "Needs ffmpeg", requires=("ffmpeg", "pillow"))
    manager.register(tool)
    result = manager.availability(tool)
    assert not result.ok and result.missing == ("ffmpeg",) and "ffmpeg" in result.message
    assert deps.probed == []  # uncached "pillow" counts as available, nothing was probed


def test_availability_alternatives_and_probe_mode():
    deps = FakeDeps({"rsvg_convert": True, "cairosvg": False})
    manager = PluginManager(None, deps)  # type: ignore[arg-type]
    tool = toy("image.svg", "SVG", requires=("cairosvg|rsvg_convert",))
    manager.register(tool)
    assert manager.availability(tool, probe=True).ok
    assert set(deps.probed) == {"cairosvg", "rsvg_convert"}
    deps.installed["rsvg_convert"] = False
    assert not manager.availability("image.svg", probe=True).ok
    assert not manager.availability("unknown.tool").ok
    assert PluginManager(None, None).availability(tool).ok  # type: ignore[arg-type]


def test_tool_without_requirements_is_always_available():
    manager = PluginManager(None, FakeDeps({}))  # type: ignore[arg-type]
    tool = toy("other.free", "Free")
    manager.register(tool)
    assert manager.availability(tool, probe=True).ok

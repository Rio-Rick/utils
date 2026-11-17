from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.core.config_schema import Kind, Setting
from app.core.errors import ToolError
from app.core.inputs import collect_files
from app.tools import base
from app.tools.base import BatchTool, Category, InputMode, ItemResult, RunContext, WorkItem, output_settings, register_tool


def make_tool(**attrs: Any) -> type[BatchTool]:
    body: dict[str, Any] = {
        "id": "other.toy",
        "name": "Toy",
        "description": "A toy tool",
        "category": Category.OTHER,
        "config_domain": "toy",
        "config_section": "toy",
        "options": (
            Setting("level", Kind.INT, 5, "Level", minimum=0, maximum=10),
            Setting("mode", Kind.CHOICE, "a", "Mode", choices=("a", "b")),
            Setting("tint", Kind.COLOR, "#112233", "Tint"),
        ),
    }
    body.update(attrs)
    return type("Toy", (BatchTool,), body)


# ----------------------------------------------------------------------------- registry
def test_register_validates_id_and_required_attributes(clean_registry):
    with pytest.raises(ValueError):
        register_tool(make_tool(id="NoDots"))
    with pytest.raises(ValueError):
        register_tool(make_tool(id="Bad.Id"))
    incomplete = type("Broken", (BatchTool,), {"id": "other.broken"})
    with pytest.raises(ValueError):
        register_tool(incomplete)
    cls = register_tool(make_tool())
    assert base.registered_tool_classes()["other.toy"] is cls


def test_duplicate_id_from_another_module_is_rejected(clean_registry):
    first = make_tool()
    first.__module__ = "plugin_one"
    register_tool(first)
    second = make_tool()
    second.__module__ = "plugin_two"
    with pytest.raises(ValueError):
        register_tool(second)
    same_module = make_tool()
    same_module.__module__ = "plugin_one"
    register_tool(same_module)  # reloading a module replaces its own registration


def test_unregister_is_idempotent(clean_registry):
    register_tool(make_tool())
    base.unregister_tool("other.toy")
    base.unregister_tool("other.toy")
    assert "other.toy" not in base.registered_tool_classes()


# ----------------------------------------------------------------------------- options
def test_validate_options_coerces_and_drops_unknown_keys():
    tool = make_tool()()
    cleaned = tool.validate_options({"level": "7", "mode": "B", "tint": "#abc", "bogus": 1})
    assert cleaned == {"level": 7, "mode": "b", "tint": "#aabbcc"}
    assert tool.validate_options({}) == tool.defaults()


def test_validate_options_reports_every_problem():
    tool = make_tool()()
    with pytest.raises(ToolError) as info:
        tool.validate_options({"level": 99, "mode": "zzz"})
    assert "Level" in info.value.detail and "Mode" in info.value.detail


def test_check_options_hook_is_called():
    class Picky(make_tool()):  # type: ignore[misc]
        def check_options(self, options):
            if options["level"] == 3:
                raise ToolError("three is not allowed")

    tool = Picky()
    assert tool.validate_options({"level": 4})["level"] == 4
    with pytest.raises(ToolError, match="three"):
        tool.validate_options({"level": 3})


def test_resolve_options_layers(services, clean_registry):
    tool_cls = register_tool(make_tool())
    tool = tool_cls()
    services.plugins.register(tool)
    config = services.config
    assert tool.resolve_options(services)["level"] == 5

    config.set("toy", "toy.level", 6)                                   # config file
    assert tool.resolve_options(services)["level"] == 6

    config.set("advanced", "tool_overrides", {"other.toy": {"level": 7}})  # per-tool override
    assert tool.resolve_options(services)["level"] == 7

    assert tool.resolve_options(services, preset={"level": 8, "mode": "b"})["level"] == 8   # preset
    resolved = tool.resolve_options(services, preset={"level": 8}, overrides={"level": 9})  # form
    assert resolved["level"] == 9 and resolved["mode"] == "a"


def test_invalid_tool_override_is_ignored_not_fatal(services, clean_registry):
    tool = make_tool()()
    services.plugins.register(tool)
    services.config.set("advanced", "tool_overrides", {"other.toy": {"level": 999}})
    assert tool.resolve_options(services)["level"] == 5


def test_output_settings_inherit_processing_defaults(services):
    settings = {s.key: s for s in output_settings()}
    assert set(settings) >= {"output.location", "output.naming_template", "output.overwrite"}
    assert all(s.inherit == f"processing.{s.key}" for s in settings.values())


# ----------------------------------------------------------------------------- items
def test_make_items_numbers_files(tmp_path: Path):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("x")
    tool = make_tool()()
    items = tool.make_items(collect_files([tmp_path]), {})
    assert [i.name for i in items] == ["a.txt", "b.txt"]
    assert [i.extra["index"] for i in items] == [1, 2]
    assert all(i.base_dir == tmp_path for i in items)


def test_merge_mode_builds_one_item_with_all_sources(tmp_path: Path):
    files = []
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("x")
        files.append(tmp_path / name)
    tool = make_tool(input_mode=InputMode.MERGE)()
    items = tool.make_items(collect_files(files), {})
    assert len(items) == 1 and items[0].sources == tuple(files) and items[0].name == "3 files"


def test_expand_inputs_skips_the_output_subfolder(tmp_path: Path):
    (tmp_path / "in.txt").write_text("x")
    (tmp_path / "converted").mkdir()
    (tmp_path / "converted" / "old.txt").write_text("x")
    tool = make_tool()()
    scan = tool.expand_inputs([tmp_path], {"output.location": "subfolder", "output.subfolder_name": "converted"})
    assert [p.name for p in scan.paths] == ["in.txt"]
    scan = tool.expand_inputs([tmp_path], {"output.location": "same"})
    assert {p.name for p in scan.paths} == {"in.txt", "old.txt"}


def test_work_item_and_result_helpers(tmp_path: Path):
    assert WorkItem().name == "-" and WorkItem().sources == ()
    assert WorkItem(tmp_path / "x.png").name == "x.png"
    assert WorkItem(tmp_path / "x.png", label="nice").name == "nice"
    assert ItemResult().primary_output is None
    assert ItemResult(outputs=[tmp_path / "o"]).primary_output == tmp_path / "o"


# ----------------------------------------------------------------------------- run context
def test_run_context_progress_is_a_cancellation_point(services):
    seen: list[tuple[float, str]] = []
    ctx = RunContext(services, on_progress=lambda f, m: seen.append((f, m)))
    ctx.progress(1.7, "x")
    ctx.progress(-1, "y")
    assert seen == [(1.0, "x"), (0.0, "y")]
    ctx.control.cancel()
    from app.core.errors import Cancelled

    with pytest.raises(Cancelled):
        ctx.progress(0.5)


def test_run_context_temp_dir_and_cleanup(services, tmp_path: Path):
    services.config.set("app", "directories.temp", str(tmp_path / "scratch"))
    ctx = RunContext(services)
    scratch = ctx.temp_dir()
    assert scratch.parent == tmp_path / "scratch" and scratch.is_dir()
    assert ctx.temp_dir() == scratch
    ctx.cleanup()
    assert not scratch.exists()

    services.config.set("processing", "cleanup_temp", False)
    ctx = RunContext(services)
    kept = ctx.temp_dir()
    ctx.cleanup()
    assert kept.exists()

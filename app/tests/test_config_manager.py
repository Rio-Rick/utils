"""Tests for the layered YAML configuration manager."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.core.config_manager import BUNDLE_FORMAT, ConfigManager
from app.core.config_schema import ConfigError, Kind, Setting

SETTINGS = {
    "demo": (
        Setting("quality", Kind.INT, 80, "Quality", minimum=1, maximum=100),
        Setting("resize.mode", Kind.CHOICE, "fit", "Resize mode", choices=("fit", "fill")),
        Setting("path.ffmpeg", Kind.FILE, "", "ffmpeg", sensitive=True),
        Setting("out.location", Kind.CHOICE, "subfolder", "Where", choices=("subfolder", "same")),
        Setting("tags", Kind.LIST, ["a"], "Tags"),
    ),
    "other": (
        Setting("size", Kind.INT, 1, "Size", inherit="demo.quality"),
        Setting("loc", Kind.CHOICE, "same", "Loc", choices=("subfolder", "same"), inherit="demo.out.location"),
    ),
}


def make(user_dir: Path, defaults: Path | None = None) -> ConfigManager:
    return ConfigManager(user_dir, defaults, settings=SETTINGS)


def read_user(user_dir: Path, domain: str = "demo") -> dict:
    return yaml.safe_load((user_dir / f"{domain}.yaml").read_text())


class TestResolution:
    def test_defaults_without_any_file(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        assert cfg.get("demo", "quality") == 80
        assert cfg.source_of("demo", "quality") == "default"
        assert cfg.tree("demo")["resize"] == {"mode": "fit"}
        assert cfg.get("demo", "missing", None) is None
        with pytest.raises(KeyError):
            cfg.get("demo", "missing")

    def test_layer_order_default_shipped_user(self, tmp_path: Path, user_dir: Path) -> None:
        defaults = tmp_path / "defaults"
        defaults.mkdir()
        (defaults / "demo.yaml").write_text("quality: 70\n")
        cfg = make(user_dir, defaults)
        assert cfg.get("demo", "quality") == 70
        assert cfg.source_of("demo", "quality") == "shipped"
        cfg.set("demo", "quality", 90)
        assert cfg.get("demo", "quality") == 90
        assert cfg.source_of("demo", "quality") == "user"
        cfg.reset("demo", "quality")
        assert cfg.get("demo", "quality") == 70

    def test_inherit_follows_parent_until_set(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        assert cfg.get("other", "size") == 80
        assert cfg.source_of("other", "size") == "inherited"
        cfg.set("demo", "quality", 55)
        assert cfg.get("other", "size") == 55
        cfg.set("other", "size", 10)
        cfg.set("demo", "quality", 99)
        assert cfg.get("other", "size") == 10
        # an explicit value equal to the parent is still kept (it pins the setting)
        cfg.set("other", "size", 99)
        assert cfg.source_of("other", "size") == "user"

    def test_get_returns_private_copies(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        value = cfg.get("demo", "tags")
        value.append("zzz")
        assert cfg.get("demo", "tags") == ["a"]

    def test_section_returns_relative_keys(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        assert cfg.section("demo", "resize") == {"mode": "fit"}


class TestWriting:
    def test_only_changed_values_are_saved(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set("demo", "quality", 80)  # equals default -> nothing stored
        assert not (user_dir / "demo.yaml").exists()
        cfg.set("demo", "quality", 60)
        assert read_user(user_dir) == {"quality": 60}
        cfg.set("demo", "quality", 80)
        assert not (user_dir / "demo.yaml").exists()  # back to default -> file removed

    def test_invalid_value_raises_and_changes_nothing(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        with pytest.raises(ConfigError):
            cfg.set_many("demo", {"quality": 50, "resize.mode": "zoom"})
        assert cfg.get("demo", "quality") == 80
        with pytest.raises(ConfigError):
            cfg.set("demo", "nope", 1)

    def test_set_many_reports_changed_keys_and_notifies(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        events: list[tuple[str, frozenset[str]]] = []
        unsubscribe = cfg.subscribe(lambda d, k: events.append((d, k)))
        changed = cfg.set_many("demo", {"quality": 10, "resize.mode": "fill"})
        assert changed == {"quality", "resize.mode"}
        assert events == [("demo", frozenset({"quality", "resize.mode"}))]
        assert cfg.set_many("demo", {"quality": 10}) == set()
        unsubscribe()
        cfg.set("demo", "quality", 11)
        assert len(events) == 1

    def test_listener_errors_do_not_break_changes(self, user_dir: Path) -> None:
        cfg = make(user_dir)

        def boom(domain: str, keys: frozenset[str]) -> None:
            raise RuntimeError("listener bug")

        cfg.subscribe(boom)
        cfg.set("demo", "quality", 5)
        assert cfg.get("demo", "quality") == 5

    def test_no_temp_files_left_behind(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set("demo", "quality", 5)
        assert sorted(p.name for p in user_dir.iterdir()) == ["demo.yaml"]

    def test_user_file_permissions_are_private(self, user_dir: Path) -> None:
        make(user_dir).set("demo", "quality", 5)
        assert (user_dir / "demo.yaml").stat().st_mode & 0o077 == 0


class TestRobustness:
    def test_invalid_and_unknown_values_are_kept_on_disk(self, user_dir: Path) -> None:
        (user_dir / "demo.yaml").write_text("quality: 500\nmystery: 1\nresize:\n  mode: fill\n")
        cfg = make(user_dir)
        assert cfg.get("demo", "quality") == 80  # invalid -> default
        assert cfg.get("demo", "resize.mode") == "fill"
        issues = {(i.key, i.severity) for i in cfg.issues("demo")}
        assert issues == {("quality", "error"), ("mystery", "warning")}
        cfg.set("demo", "resize.mode", "fit")
        saved = read_user(user_dir)
        assert saved["quality"] == 500 and saved["mystery"] == 1  # never silently dropped

    def test_broken_yaml_is_backed_up_not_lost(self, user_dir: Path) -> None:
        path = user_dir / "demo.yaml"
        path.write_text("quality: [unclosed\n")
        cfg = make(user_dir)
        assert cfg.get("demo", "quality") == 80
        assert any(i.severity == "error" and i.domain == "demo" for i in cfg.issues())
        assert path.read_text() == "quality: [unclosed\n"  # untouched until the user changes something
        cfg.set("demo", "quality", 42)
        assert read_user(user_dir) == {"quality": 42}
        stashed = list((user_dir / "backups").glob("demo.broken-*.yaml"))
        assert len(stashed) == 1 and stashed[0].read_text() == "quality: [unclosed\n"

    def test_non_mapping_top_level_is_an_error(self, user_dir: Path) -> None:
        (user_dir / "demo.yaml").write_text("- 1\n- 2\n")
        assert make(user_dir).issues("demo")[0].severity == "error"

    def test_broken_shipped_default_is_reported(self, tmp_path: Path, user_dir: Path) -> None:
        defaults = tmp_path / "defaults"
        defaults.mkdir()
        (defaults / "demo.yaml").write_text("quality: nope\n")
        cfg = make(user_dir, defaults)
        assert cfg.get("demo", "quality") == 80
        assert any(i.key == "quality" for i in cfg.issues("demo"))

    def test_reload_if_changed_picks_up_manual_edits_but_not_own_writes(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        events: list[tuple[str, frozenset[str]]] = []
        cfg.subscribe(lambda d, k: events.append((d, k)))
        cfg.set("demo", "quality", 20)
        events.clear()
        assert cfg.reload_if_changed() == set()  # our own write is not an external change
        (user_dir / "demo.yaml").write_text("quality: 30\n")
        assert cfg.reload_if_changed() == {"demo"}
        assert cfg.get("demo", "quality") == 30
        assert events == [("demo", frozenset({"quality"}))]

    def test_registering_more_settings_keeps_unsaved_state(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set("demo", "quality", 33, save=False)
        cfg.register_domain("demo", [Setting("extra", Kind.INT, 1, "Extra")], section="ext")
        assert cfg.get("demo", "quality") == 33
        assert cfg.get("demo", "ext.extra") == 1

    def test_conflicting_registration_rejected(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        with pytest.raises(ConfigError):
            cfg.register_domain("demo", [Setting("quality", Kind.STR, "x", "Quality")])
        with pytest.raises(ConfigError):
            cfg.register_domain("Bad Name", [])

    def test_validate_flags_missing_paths(self, user_dir: Path) -> None:
        (user_dir / "demo.yaml").write_text("path:\n  ffmpeg: /definitely/not/here\n")
        issues = make(user_dir).validate()
        assert any(i.key == "path.ffmpeg" and "does not exist" in i.message for i in issues)

    def test_tool_overrides_validation(self, user_dir: Path) -> None:
        cfg = ConfigManager(user_dir, None)
        cfg.set("advanced", "tool_overrides", {"image.convert": {"quality": 50, "bogus": 1, "mode": "zoom"}})
        options = [
            Setting("quality", Kind.INT, 80, "Q", minimum=1, maximum=100),
            Setting("mode", Kind.CHOICE, "fit", "M", choices=("fit", "fill")),
        ]
        valid, issues = cfg.validate_tool_overrides("image.convert", options)
        assert valid == {"quality": 50}
        assert {i.key.rsplit(".", 1)[-1] for i in issues} == {"bogus", "mode"}


class TestExportImport:
    def test_roundtrip_overrides_only(self, tmp_path: Path, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set_many("demo", {"quality": 12, "resize.mode": "fill"})
        out = cfg.export_bundle(tmp_path / "export.yaml")
        bundle = yaml.safe_load(out.read_text())
        assert bundle["format"] == BUNDLE_FORMAT
        assert bundle["domains"] == {"demo": {"quality": 12, "resize": {"mode": "fill"}}}

        other_dir = tmp_path / "other-user"
        other = make(other_dir)
        report = other.import_bundle(out)
        assert sorted(report.applied) == ["demo.quality", "demo.resize.mode"]
        assert other.get("demo", "quality") == 12 and other.get("demo", "resize.mode") == "fill"

    def test_export_leaves_out_sensitive_unless_asked(self, tmp_path: Path, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set_many("demo", {"quality": 12, "path.ffmpeg": "/usr/bin/ffmpeg"})
        safe = yaml.safe_load(cfg.export_bundle(tmp_path / "a.yaml").read_text())
        assert "path" not in safe["domains"]["demo"]
        full = yaml.safe_load(cfg.export_bundle(tmp_path / "b.yaml", include_sensitive=True).read_text())
        assert full["domains"]["demo"]["path"] == {"ffmpeg": "/usr/bin/ffmpeg"}

    def test_export_effective_includes_defaults(self, tmp_path: Path, user_dir: Path) -> None:
        cfg = make(user_dir)
        bundle = yaml.safe_load(cfg.export_bundle(tmp_path / "e.yaml", include_defaults=True).read_text())
        assert bundle["domains"]["demo"]["quality"] == 80

    def test_import_skips_sensitive_by_default(self, tmp_path: Path, user_dir: Path) -> None:
        src = tmp_path / "in.yaml"
        src.write_text(yaml.safe_dump({"format": BUNDLE_FORMAT, "version": 1, "domains": {
            "demo": {"quality": 5, "path": {"ffmpeg": "/tmp/evil"}}}}))
        cfg = make(user_dir)
        report = cfg.import_bundle(src)
        assert report.skipped_sensitive == ["demo.path.ffmpeg"]
        assert cfg.get("demo", "path.ffmpeg") == ""
        assert cfg.get("demo", "quality") == 5
        cfg.import_bundle(src, allow_sensitive=True)
        assert cfg.get("demo", "path.ffmpeg") == "/tmp/evil"

    def test_dry_run_changes_nothing(self, tmp_path: Path, user_dir: Path) -> None:
        src = tmp_path / "in.yaml"
        src.write_text("demo:\n  quality: 5\n")  # hand-written plain mapping is accepted
        cfg = make(user_dir)
        report = cfg.import_bundle(src, dry_run=True)
        assert report.applied == ["demo.quality"]
        assert cfg.get("demo", "quality") == 80
        assert not (user_dir / "demo.yaml").exists()

    def test_replace_mode_keeps_existing_sensitive_values(self, tmp_path: Path, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set_many("demo", {"quality": 1, "resize.mode": "fill", "path.ffmpeg": "/usr/bin/ffmpeg"})
        src = tmp_path / "in.yaml"
        src.write_text("demo:\n  quality: 5\n")
        cfg.import_bundle(src, mode="replace")
        assert cfg.get("demo", "quality") == 5
        assert cfg.get("demo", "resize.mode") == "fit"  # replaced
        assert cfg.get("demo", "path.ffmpeg") == "/usr/bin/ffmpeg"  # machine-specific, survives

    def test_import_reports_invalid_values_and_rejects_garbage(self, tmp_path: Path, user_dir: Path) -> None:
        cfg = make(user_dir)
        src = tmp_path / "in.yaml"
        src.write_text("demo:\n  quality: 5000\n  tags: [x]\nnothing: 1\n")
        report = cfg.import_bundle(src)
        assert report.applied == ["demo.tags"]
        assert any(i.key == "quality" and i.severity == "error" for i in report.issues)
        junk = tmp_path / "junk.yaml"
        junk.write_text("hello: world\n")
        with pytest.raises(ConfigError):
            cfg.import_bundle(junk)
        broken = tmp_path / "broken.yaml"
        broken.write_text("a: [")
        with pytest.raises(ConfigError):
            cfg.import_bundle(broken)

    def test_import_makes_a_backup_of_existing_files(self, tmp_path: Path, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set("demo", "quality", 7)
        src = tmp_path / "in.yaml"
        src.write_text("demo:\n  quality: 9\n")
        report = cfg.import_bundle(src)
        assert report.backup is not None
        assert yaml.safe_load((report.backup / "demo.yaml").read_text()) == {"quality": 7}


class TestResetAndBackups:
    def test_reset_domain_and_restore_defaults(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set_many("demo", {"quality": 7, "resize.mode": "fill"})
        cfg.set("other", "size", 3)
        assert cfg.reset("demo") == {"quality", "resize.mode"}
        assert not (user_dir / "demo.yaml").exists()
        assert (user_dir / "other.yaml").exists()
        backup = cfg.reset_all()
        assert backup is not None and (backup / "other.yaml").exists()
        assert not (user_dir / "other.yaml").exists()

    def test_restore_backup(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set("demo", "quality", 7)
        cfg.reset_all()
        assert cfg.get("demo", "quality") == 80
        newest_with_demo = next(b for b in cfg.list_backups() if (b / "demo.yaml").exists())
        cfg.restore_backup(newest_with_demo)
        assert cfg.get("demo", "quality") == 7

    def test_restore_rejects_empty_folder(self, tmp_path: Path, user_dir: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ConfigError):
            make(user_dir).restore_backup(empty)

    def test_backups_are_pruned(self, user_dir: Path) -> None:
        cfg = make(user_dir)
        cfg.set("demo", "quality", 7)
        for i in range(30):
            cfg.backup(f"b{i}")
        assert len(cfg.list_backups()) <= 20

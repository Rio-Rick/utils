"""Tests for the setting schema, validation and nested-dict helpers."""

from __future__ import annotations

import pytest

from app.core.config_defs import CORE_SETTINGS, validate_environment
from app.core.config_schema import (
    ConfigError,
    Kind,
    Setting,
    build_defaults,
    delete_path,
    deep_merge,
    get_path,
    iter_leaf_paths,
    normalise_color,
    normalise_shortcut,
    set_path,
    split_args,
    validate_tree,
)


class TestCoercion:
    def test_int_range_and_text(self) -> None:
        setting = Setting("quality", Kind.INT, 80, "Quality", minimum=1, maximum=100)
        assert setting.coerce("75") == 75
        assert setting.coerce(75.0) == 75
        for bad in (0, 101, "abc", 7.5, True, None):
            with pytest.raises(ValueError):
                setting.coerce(bad)

    def test_float_and_bool(self) -> None:
        assert Setting("x", Kind.FLOAT, 1.0, "X", minimum=0).coerce("2.5") == 2.5
        flag = Setting("flag", Kind.BOOL, False, "Flag")
        assert flag.coerce("yes") is True
        assert flag.coerce("off") is False
        with pytest.raises(ValueError):
            flag.coerce("maybe")

    def test_choice_is_case_insensitive_and_canonical(self) -> None:
        setting = Setting("mode", Kind.CHOICE, "fit", "Mode", choices=("fit", "fill"))
        assert setting.coerce(" FIT ") == "fit"
        with pytest.raises(ValueError):
            setting.coerce("zoom")

    def test_choice_requires_choices(self) -> None:
        with pytest.raises(ValueError):
            Setting("mode", Kind.CHOICE, "a", "Mode")

    def test_invalid_key_rejected(self) -> None:
        with pytest.raises(ValueError):
            Setting("Bad Key", Kind.STR, "", "Bad")

    def test_color_and_shortcut_normalisation(self) -> None:
        assert normalise_color("#ABC") == "#aabbcc"
        assert normalise_color("#11223344") == "#11223344"
        with pytest.raises(ValueError):
            normalise_color("red")
        assert normalise_shortcut("ctrl+shift+o") == "Ctrl+Shift+O"
        assert normalise_shortcut("") == ""
        with pytest.raises(ValueError):
            normalise_shortcut("Ctrl+Banana")

    def test_list_with_choices(self) -> None:
        setting = Setting("items", Kind.LIST, [], "Items", choices=("a", "b"))
        assert setting.coerce(["a"]) == ["a"]
        with pytest.raises(ValueError):
            setting.coerce(["c"])
        with pytest.raises(ValueError):
            setting.coerce("a")

    def test_args_use_shlex_not_shell(self) -> None:
        assert split_args("-vf 'scale=1:1' -an") == ["-vf", "scale=1:1", "-an"]
        setting = Setting("args", Kind.ARGS, "", "Args")
        assert setting.coerce("  -an  ") == "-an"
        with pytest.raises(ValueError):
            setting.coerce("-vf 'unterminated")
        with pytest.raises(ConfigError):
            split_args("'unterminated")

    def test_nul_rejected_in_text(self) -> None:
        with pytest.raises(ValueError):
            Setting("s", Kind.STR, "", "S").coerce("a\x00b")

    def test_custom_validator_message(self) -> None:
        setting = Setting("m", Kind.MAP, {}, "M", validator=validate_environment)
        assert setting.coerce({"FOO": "bar"}) == {"FOO": "bar"}
        for name in ("PATH", "LD_PRELOAD", "PYTHONPATH", "1BAD"):
            with pytest.raises(ValueError):
                setting.coerce({name: "x"})


class TestVisibility:
    def test_visible_if_and_prefix(self) -> None:
        setting = Setting("lossless", Kind.BOOL, False, "Lossless", visible_if={"format": ["webp"]})
        assert setting.is_visible({"format": "webp"})
        assert not setting.is_visible({"format": "png"})
        prefixed = setting.with_prefix("formats")
        assert prefixed.key == "formats.lossless"
        assert prefixed.visible_if == {"formats.format": ["webp"]}


class TestNestedHelpers:
    def test_set_get_delete_prune(self) -> None:
        data: dict = {}
        set_path(data, "a.b.c", 1)
        set_path(data, "a.d", 2)
        assert get_path(data, "a.b.c") == 1
        assert get_path(data, "a.x", "fallback") == "fallback"
        with pytest.raises(KeyError):
            get_path(data, "a.x")
        assert delete_path(data, "a.b.c")
        assert data == {"a": {"d": 2}}  # empty parent "b" pruned
        assert not delete_path(data, "a.zzz")

    def test_deep_merge_does_not_mutate(self) -> None:
        base = {"a": {"b": 1, "c": 2}, "l": [1]}
        merged = deep_merge(base, {"a": {"b": 9}, "l": [2]})
        assert merged == {"a": {"b": 9, "c": 2}, "l": [2]}
        assert base == {"a": {"b": 1, "c": 2}, "l": [1]}

    def test_iter_leaf_paths_keeps_empty_mappings_as_leaves(self) -> None:
        leaves = dict(iter_leaf_paths({"a": {"b": 1, "c": {}}}))
        assert leaves == {"a.b": 1, "a.c": {}}


class TestValidateTree:
    def test_valid_invalid_unknown(self) -> None:
        settings = {s.key: s for s in (
            Setting("a.q", Kind.INT, 1, "Q", minimum=1, maximum=5),
            Setting("a.name", Kind.STR, "", "Name"),
        )}
        valid, issues = validate_tree("t", settings, {"a": {"q": 99, "name": "x", "extra": 1}, "zzz": 1}, source="s")
        assert valid == {"a.name": "x"}
        severities = {(i.key, i.severity) for i in issues}
        assert severities == {("a.q", "error"), ("a.extra", "warning"), ("zzz", "warning")}

    def test_section_given_as_scalar_is_an_error(self) -> None:
        settings = {"a.q": Setting("a.q", Kind.INT, 1, "Q")}
        valid, issues = validate_tree("t", settings, {"a": 5}, source="s")
        assert valid == {}
        assert issues[0].severity == "error"


class TestCoreDefinitions:
    def test_defaults_are_valid_and_unique(self) -> None:
        for domain, settings in CORE_SETTINGS.items():
            keys = [s.key for s in settings]
            assert len(keys) == len(set(keys)), f"duplicate keys in {domain}"
            for setting in settings:
                setting.coerce(setting.default)  # the shipped default must itself be valid

    def test_inherit_targets_exist(self) -> None:
        index = {(d, s.key) for d, items in CORE_SETTINGS.items() for s in items}
        for domain, settings in CORE_SETTINGS.items():
            for setting in settings:
                if setting.inherit:
                    other_domain, _, other_key = setting.inherit.partition(".")
                    assert (other_domain, other_key) in index, f"{domain}.{setting.key} inherits unknown {setting.inherit}"

    def test_build_defaults_nested(self) -> None:
        tree = build_defaults(CORE_SETTINGS["app"])
        assert tree["max_concurrent_jobs"] == 2
        assert tree["logging"]["level"] in {"DEBUG", "INFO", "WARNING", "ERROR"}

"""Typed description of every configuration setting.

A :class:`Setting` is the single definition used by

* the configuration manager (defaults, validation, coercion),
* the generic settings / options forms in the GUI,
* the generated documentation (``docs/configuration.md``) and default YAML files.

Tools contribute their own settings (see :mod:`app.tools.base`), so a new
utility becomes configurable without touching the core.
"""

from __future__ import annotations

import copy
import re
import shlex
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Literal

KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"false", "no", "off", "0"})
_MODIFIERS = {"ctrl": "Ctrl", "control": "Ctrl", "shift": "Shift", "alt": "Alt", "meta": "Meta"}
_NAMED_KEYS = frozenset(
    {
        "return", "enter", "esc", "escape", "tab", "backspace", "delete", "del", "space", "up", "down",
        "left", "right", "home", "end", "pgup", "pgdown", "pageup", "pagedown", "ins", "insert", "menu",
        "comma", "period", "plus", "minus", "slash", "backslash",
    }
)


class ConfigError(ValueError):
    """Raised when a configuration value or file is invalid."""


class Kind(StrEnum):
    """Value type of a setting; decides validation and which widget the GUI shows."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    TEXT = "text"          # multi-line string
    CHOICE = "choice"
    COLOR = "color"        # #rgb, #rrggbb or #rrggbbaa
    PATH = "path"          # any filesystem path (file or directory)
    DIR = "dir"
    FILE = "file"
    SHORTCUT = "shortcut"  # key sequence such as Ctrl+Shift+O (empty = disabled)
    LIST = "list"          # list of strings
    MAP = "map"            # free-form mapping validated by ``Setting.validator``
    ARGS = "args"          # command line arguments, parsed with shlex (never a shell)


@dataclass(frozen=True, slots=True, eq=False)
class Setting:
    """Definition of one configuration setting.

    ``key`` is a dotted path inside the YAML file of the domain (``resize.mode``).
    ``inherit`` points at another ``domain.key`` whose value is used when this
    setting was not explicitly configured (shared output defaults, for example).
    ``visible_if`` maps another key of the same form to the values for which this
    setting is shown (progressive disclosure).
    """

    key: str
    kind: Kind
    default: Any
    label: str
    help: str = ""
    choices: tuple[str, ...] = ()
    choice_labels: Mapping[str, str] | None = None
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""
    advanced: bool = False
    sensitive: bool = False
    restart: bool = False
    group: str = ""
    inherit: str | None = None
    visible_if: Mapping[str, Sequence[Any]] | None = None
    validator: Callable[[Any], str | None] | None = None

    def __post_init__(self) -> None:
        if not KEY_RE.match(self.key):
            raise ValueError(f"invalid setting key: {self.key!r}")
        if self.kind is Kind.CHOICE and not self.choices:
            raise ValueError(f"choice setting {self.key!r} needs choices")

    # ------------------------------------------------------------------ helpers
    def with_prefix(self, section: str) -> Setting:
        """Return a copy whose key (and visibility references) are prefixed by ``section``."""
        if not section:
            return self
        visible = None
        if self.visible_if:
            visible = {f"{section}.{k}": v for k, v in self.visible_if.items()}
        return replace(self, key=f"{section}.{self.key}", visible_if=visible)

    def label_for(self, choice: str) -> str:
        """Human readable label of a choice value."""
        if self.choice_labels and choice in self.choice_labels:
            return self.choice_labels[choice]
        return choice

    def is_visible(self, values: Mapping[str, Any]) -> bool:
        """Evaluate ``visible_if`` against the current values of the form."""
        if not self.visible_if:
            return True
        return all(values.get(key) in allowed for key, allowed in self.visible_if.items())

    def default_value(self) -> Any:
        """A private copy of the default (safe to mutate)."""
        return copy.deepcopy(self.default)

    def constraints(self) -> str:
        """Short text describing the accepted values (used in docs and tooltips)."""
        kind = self.kind
        if kind is Kind.CHOICE:
            return "one of: " + ", ".join(self.choices)
        if kind in (Kind.INT, Kind.FLOAT):
            lo = "" if self.minimum is None else f"{_num(self.minimum)}"
            hi = "" if self.maximum is None else f"{_num(self.maximum)}"
            span = f"{lo}..{hi}" if lo or hi else "number"
            return f"{kind.value} {span}{(' ' + self.unit) if self.unit else ''}".strip()
        if kind is Kind.COLOR:
            return "hex colour (#rrggbb or #rrggbbaa)"
        if kind is Kind.SHORTCUT:
            return "key sequence such as Ctrl+Shift+O (empty disables it)"
        if kind is Kind.LIST:
            return "list of: " + ", ".join(self.choices) if self.choices else "list of text values"
        if kind is Kind.ARGS:
            return "command line arguments (quote with shell syntax; no shell is used)"
        if kind is Kind.BOOL:
            return "true or false"
        return kind.value

    # ------------------------------------------------------------------ validation
    def coerce(self, value: Any) -> Any:
        """Validate ``value`` and return its normalised form. Raises :class:`ValueError`."""
        result = self._coerce_kind(value)
        if self.validator is not None:
            message = self.validator(result)
            if message:
                raise ValueError(message)
        return result

    def _coerce_kind(self, value: Any) -> Any:  # noqa: C901 - one branch per kind is clearest
        kind = self.kind
        if kind is Kind.BOOL:
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in _TRUE:
                return True
            if isinstance(value, str) and value.strip().lower() in _FALSE:
                return False
            raise ValueError("expected true or false")
        if kind is Kind.INT:
            number = _to_int(value)
            self._check_range(number)
            return number
        if kind is Kind.FLOAT:
            number_f = _to_float(value)
            self._check_range(number_f)
            return number_f
        if kind in (Kind.STR, Kind.TEXT, Kind.ARGS, Kind.PATH, Kind.DIR, Kind.FILE):
            text = "" if value is None else value
            if not isinstance(text, str):
                if isinstance(text, (int, float)) and not isinstance(text, bool):
                    text = str(text)
                else:
                    raise ValueError("expected text")
            if "\x00" in text:
                raise ValueError("text must not contain NUL characters")
            if kind in (Kind.PATH, Kind.DIR, Kind.FILE, Kind.ARGS):
                text = text.strip()
            if kind is Kind.ARGS and text:
                try:
                    shlex.split(text)
                except ValueError as exc:
                    raise ValueError(f"cannot parse arguments: {exc}") from exc
            return text
        if kind is Kind.CHOICE:
            if not isinstance(value, str):
                raise ValueError("expected one of: " + ", ".join(self.choices))
            lowered = {c.lower(): c for c in self.choices}
            canonical = lowered.get(value.strip().lower())
            if canonical is None:
                raise ValueError("must be one of: " + ", ".join(self.choices))
            return canonical
        if kind is Kind.COLOR:
            return normalise_color(value)
        if kind is Kind.SHORTCUT:
            return normalise_shortcut(value)
        if kind is Kind.LIST:
            if value is None:
                return []
            if not isinstance(value, (list, tuple)):
                raise ValueError("expected a list")
            items: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    raise ValueError("list items must be text")
                if self.choices and item not in self.choices:
                    raise ValueError(f"unknown item {item!r}; allowed: " + ", ".join(self.choices))
                items.append(item)
            return items
        if kind is Kind.MAP:
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ValueError("expected a mapping")
            if not all(isinstance(k, str) for k in value):
                raise ValueError("mapping keys must be text")
            return copy.deepcopy(dict(value))
        raise ValueError(f"unsupported kind {kind}")  # pragma: no cover

    def _check_range(self, number: float) -> None:
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"must be at least {_num(self.minimum)}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"must be at most {_num(self.maximum)}")


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """A problem found while loading or validating configuration."""

    severity: Literal["error", "warning"]
    domain: str
    key: str
    message: str
    source: str = ""
    value: Any = None

    def __str__(self) -> str:
        where = f"{self.domain}.{self.key}" if self.key else self.domain
        return f"[{self.severity}] {where}: {self.message}"


# ---------------------------------------------------------------------- coercion helpers
def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("expected a whole number")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError("expected a whole number")
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ValueError("expected a whole number")


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("expected a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ValueError("expected a number")


def normalise_color(value: Any) -> str:
    """Return ``value`` as lower-case ``#rrggbb`` / ``#rrggbbaa``; raises ``ValueError`` otherwise."""
    if not isinstance(value, str) or not _COLOR_RE.match(value.strip()):
        raise ValueError("expected a hex colour such as #1e90ff or #1e90ff80")
    text = value.strip().lower()
    if len(text) in (4, 5):  # #rgb / #rgba -> #rrggbb / #rrggbbaa
        text = "#" + "".join(ch * 2 for ch in text[1:])
    return text


def normalise_shortcut(value: Any) -> str:
    """Normalise a key sequence (``ctrl+shift+o`` -> ``Ctrl+Shift+O``); empty disables the shortcut."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("expected a key sequence such as Ctrl+Shift+O")
    text = value.strip()
    if not text:
        return ""
    parts = [p.strip() for p in text.split("+")]
    if text.endswith("++"):  # "Ctrl++" means the plus key
        parts = [p.strip() for p in text[:-2].split("+")] + ["Plus"]
    if not parts or any(not p for p in parts):
        raise ValueError("invalid key sequence")
    *mods, key = parts
    normalised: list[str] = []
    for mod in mods:
        canonical = _MODIFIERS.get(mod.lower())
        if canonical is None:
            raise ValueError(f"unknown modifier {mod!r}")
        if canonical in normalised:
            raise ValueError(f"duplicate modifier {mod!r}")
        normalised.append(canonical)
    lowered = key.lower()
    if len(key) == 1:
        key_out = key.upper() if key.isalpha() else key
    elif re.fullmatch(r"f([1-9]|[12][0-9]|3[0-5])", lowered):
        key_out = key.upper()
    elif lowered in _NAMED_KEYS:
        key_out = key[:1].upper() + key[1:].lower()
        if lowered in {"pgup", "pgdown"}:
            key_out = {"pgup": "PgUp", "pgdown": "PgDown"}[lowered]
    else:
        raise ValueError(f"unknown key {key!r}")
    return "+".join([*normalised, key_out])


def split_args(text: str) -> list[str]:
    """Split user supplied command line arguments (POSIX quoting, no shell involved)."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise ConfigError(f"cannot parse arguments {text!r}: {exc}") from exc


# ---------------------------------------------------------------------- nested dict helpers
_MISSING = object()


def get_path(data: Mapping[str, Any], key: str, default: Any = _MISSING) -> Any:
    """Read ``a.b.c`` from nested mappings. Raises ``KeyError`` when absent and no default is given."""
    node: Any = data
    for part in key.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            if default is _MISSING:
                raise KeyError(key)
            return default
    return node


def has_path(data: Mapping[str, Any], key: str) -> bool:
    """True when ``a.b.c`` exists in the nested mapping."""
    return get_path(data, key, _MISSING) is not _MISSING


def set_path(data: dict[str, Any], key: str, value: Any) -> None:
    """Write ``a.b.c`` into nested dicts, creating (or replacing non-dict) parents."""
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def delete_path(data: dict[str, Any], key: str) -> bool:
    """Remove ``a.b.c`` and prune parents that became empty. Returns whether something was removed."""
    parts = key.split(".")
    stack: list[tuple[dict[str, Any], str]] = []
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False
        stack.append((node, part))
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return False
    del node[parts[-1]]
    while stack and not node:
        parent, part = stack.pop()
        del parent[part]
        node = parent
    return True


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated with ``override``; nested mappings merge, everything else is replaced."""
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def build_defaults(settings: Sequence[Setting]) -> dict[str, Any]:
    """Nested dict made of every setting's own default (inherit-only settings are skipped)."""
    tree: dict[str, Any] = {}
    for setting in settings:
        if setting.inherit is None:
            set_path(tree, setting.key, setting.default_value())
    return tree


def iter_leaf_paths(data: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every non-mapping leaf of a nested mapping."""
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            yield from iter_leaf_paths(value, path)
        else:
            yield path, value


def validate_tree(
    domain: str,
    settings: Mapping[str, Setting],
    data: Mapping[str, Any],
    *,
    source: str,
) -> tuple[dict[str, Any], list[ConfigIssue]]:
    """Validate nested ``data`` against ``settings`` (keyed by dotted path).

    Returns ``(valid, issues)`` where ``valid`` maps dotted keys to coerced values.
    Invalid values are reported and left out; unknown keys only produce a warning
    (they are preserved on disk, never deleted).
    """
    valid: dict[str, Any] = {}
    issues: list[ConfigIssue] = []
    prefixes: set[str] = set()
    for key in settings:
        parts = key.split(".")
        for i in range(1, len(parts)):
            prefixes.add(".".join(parts[:i]))

    def walk(node: Mapping[str, Any], prefix: str) -> None:
        for raw_key, value in node.items():
            if not isinstance(raw_key, str):
                issues.append(ConfigIssue("error", domain, prefix, f"non-text key {raw_key!r}", source))
                continue
            path = f"{prefix}.{raw_key}" if prefix else raw_key
            setting = settings.get(path)
            if setting is not None:
                try:
                    valid[path] = setting.coerce(value)
                except ValueError as exc:
                    issues.append(ConfigIssue("error", domain, path, str(exc), source, value))
            elif path in prefixes:
                if isinstance(value, Mapping):
                    walk(value, path)
                else:
                    issues.append(ConfigIssue("error", domain, path, "expected a section (mapping)", source, value))
            else:
                issues.append(ConfigIssue("warning", domain, path, "unknown setting (ignored)", source, value))

    walk(data, "")
    return valid, issues

"""Layered, validated YAML configuration.

Effective value of a setting, lowest to highest priority::

    built-in default (Setting.default)
      < shipped default  (config/defaults/<domain>.yaml)
      < inherited value  (Setting.inherit, e.g. processing.output.location)
      < user override    (config/user/<domain>.yaml  or  ~/.config/linux-media-toolkit/)

Only the user's overrides are ever written, so shipped defaults can change
between releases without touching the user's choices. Unknown keys and values
that fail validation are reported but never deleted from the user's files; the
application simply keeps using the default for them.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import shutil
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from app.core.config_defs import CORE_SETTINGS
from app.core.config_schema import (
    _MISSING,
    ConfigError,
    ConfigIssue,
    Kind,
    Setting,
    delete_path,
    get_path,
    set_path,
    validate_tree,
)
from app.core.fsutil import atomic_write_text

log = logging.getLogger(__name__)

BUNDLE_FORMAT = "linux-media-toolkit-config"
BUNDLE_VERSION = 1
DOMAIN_RE = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_BACKUPS = 20
USER_HEADER = (
    "# Linux Media & Utility Toolkit - user configuration.\n"
    "# Only values that differ from config/defaults are stored here.\n"
    "# Safe to edit by hand; the Settings page writes this file too (comments are not preserved).\n"
)

Listener = Callable[[str, frozenset[str]], None]
Source = Literal["user", "shipped", "inherited", "default"]


def _as_int(value: Any) -> int:
    """Best-effort integer for version fields in imported files (0 when not a number)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


@dataclass(slots=True)
class ImportReport:
    """Outcome of :meth:`ConfigManager.import_bundle`."""

    applied: list[str] = field(default_factory=list)
    skipped_sensitive: list[str] = field(default_factory=list)
    issues: list[ConfigIssue] = field(default_factory=list)
    backup: Path | None = None


class ConfigManager:
    """Loads, validates, merges and saves configuration for all registered domains."""

    def __init__(
        self,
        user_dir: Path,
        defaults_dir: Path | None = None,
        *,
        settings: Mapping[str, Sequence[Setting]] | None = None,
        app_version: str = "",
    ) -> None:
        self.user_dir = Path(user_dir)
        self.defaults_dir = Path(defaults_dir) if defaults_dir else None
        self.app_version = app_version
        self._lock = threading.RLock()
        self._registry: dict[str, dict[str, Setting]] = {}
        self._shipped_raw: dict[str, dict[str, Any]] = {}
        self._shipped: dict[str, dict[str, Any]] = {}
        self._user_raw: dict[str, dict[str, Any]] = {}
        self._user: dict[str, dict[str, Any]] = {}
        self._issues_file: dict[str, list[ConfigIssue]] = {}
        self._issues_shipped: dict[str, list[ConfigIssue]] = {}
        self._shipped_errors: dict[str, list[ConfigIssue]] = {}
        self._issues_user: dict[str, list[ConfigIssue]] = {}
        self._signatures: dict[str, tuple[int, int] | None] = {}
        self._broken: set[str] = set()
        self._listeners: list[Listener] = []
        for domain, items in (settings if settings is not None else CORE_SETTINGS).items():
            self.register_domain(domain, items)

    # ------------------------------------------------------------------ registration
    def register_domain(self, domain: str, settings: Iterable[Setting], *, section: str = "") -> None:
        """Add settings to ``domain`` (optionally below ``section``) and (re)load the domain's files."""
        if not DOMAIN_RE.match(domain):
            raise ConfigError(f"invalid configuration domain name: {domain!r}")
        with self._lock:
            registry = self._registry.setdefault(domain, {})
            for setting in settings:
                prefixed = setting.with_prefix(section)
                existing = registry.get(prefixed.key)
                if existing is not None and existing is not setting and existing != prefixed:
                    if existing.kind != prefixed.kind or existing.default != prefixed.default:
                        raise ConfigError(f"conflicting definitions of setting {domain}.{prefixed.key}")
                registry[prefixed.key] = prefixed if existing is None else existing
            if domain in self._user_raw:  # already loaded: only re-validate against the new definitions
                self._validate_domain(domain)
            else:
                self._load_domain(domain)

    def domains(self) -> list[str]:
        """Registered domain names in registration order."""
        with self._lock:
            return list(self._registry)

    def settings(self, domain: str) -> list[Setting]:
        """Settings of ``domain`` in registration order."""
        with self._lock:
            return list(self._registry_for(domain).values())

    def setting(self, domain: str, key: str) -> Setting | None:
        """Look up a single setting definition."""
        with self._lock:
            return self._registry.get(domain, {}).get(key)

    def _registry_for(self, domain: str) -> dict[str, Setting]:
        try:
            return self._registry[domain]
        except KeyError:
            raise ConfigError(f"unknown configuration domain: {domain!r}") from None

    # ------------------------------------------------------------------ loading
    def reload(self) -> list[ConfigIssue]:
        """Re-read every file from disk; returns all issues found."""
        changed: dict[str, frozenset[str]] = {}
        with self._lock:
            for domain in list(self._registry):
                changed[domain] = self._reload_domain(domain)
        for domain, keys in changed.items():
            if keys:
                self._notify(domain, keys)
        return self.issues()

    def _reload_domain(self, domain: str) -> frozenset[str]:
        before = self._snapshot(domain)
        self._load_domain(domain)
        after = self._snapshot(domain)
        return frozenset(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))

    def reload_if_changed(self) -> set[str]:
        """Reload domains whose user file changed on disk (e.g. edited by hand). Returns their names."""
        changed: dict[str, frozenset[str]] = {}
        with self._lock:
            for domain in list(self._registry):
                if self._signature(self._user_path(domain)) != self._signatures.get(domain):
                    changed[domain] = self._reload_domain(domain)
        for domain, keys in changed.items():
            if keys:
                self._notify(domain, keys)
        return set(changed)

    def _user_path(self, domain: str) -> Path:
        return self.user_dir / f"{domain}.yaml"

    @staticmethod
    def _signature(path: Path) -> tuple[int, int] | None:
        try:
            st = path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    @staticmethod
    def _read_yaml(path: Path) -> tuple[dict[str, Any] | None, str | None]:
        """Return ``(data, error)``; a missing file is an empty mapping."""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}, None
        except (OSError, UnicodeDecodeError) as exc:
            return None, f"cannot read file: {exc}"
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            detail = " ".join(str(exc).split())
            return None, f"invalid YAML: {detail[:300]}"
        if data is None:
            return {}, None
        if not isinstance(data, dict):
            return None, "the top level of the file must be a mapping (key: value)"
        return data, None

    def _load_domain(self, domain: str) -> None:
        """Read the shipped and user files of ``domain`` from disk and validate them."""
        shipped_raw: dict[str, Any] = {}
        shipped_errors: list[ConfigIssue] = []
        if self.defaults_dir is not None:
            path = self.defaults_dir / f"{domain}.yaml"
            data, error = self._read_yaml(path)
            if error:
                shipped_errors.append(ConfigIssue("error", domain, "", f"default file {path.name}: {error}", str(path)))
            else:
                shipped_raw = data or {}
        self._shipped_raw[domain] = shipped_raw
        self._shipped_errors[domain] = shipped_errors

        path = self._user_path(domain)
        self._signatures[domain] = self._signature(path)
        data, error = self._read_yaml(path)
        file_issues: list[ConfigIssue] = []
        if error:
            file_issues.append(ConfigIssue("error", domain, "", f"{path.name}: {error}", str(path)))
            self._broken.add(domain)
            data = {}
        else:
            self._broken.discard(domain)
        self._user_raw[domain] = data or {}
        self._issues_file[domain] = file_issues
        self._validate_domain(domain)

    def _validate_domain(self, domain: str) -> None:
        registry = self._registry[domain]
        if self.defaults_dir is not None:
            source = str(self.defaults_dir / f"{domain}.yaml")
            self._shipped[domain], found = validate_tree(domain, registry, self._shipped_raw.get(domain, {}), source=source)
            self._issues_shipped[domain] = self._shipped_errors.get(domain, []) + found
        else:
            self._shipped[domain] = {}
            self._issues_shipped[domain] = []
        self._revalidate_user(domain)

    def _revalidate_user(self, domain: str) -> None:
        registry = self._registry[domain]
        valid, issues = validate_tree(domain, registry, self._user_raw[domain], source=str(self._user_path(domain)))
        self._user[domain] = valid
        self._issues_user[domain] = issues

    # ------------------------------------------------------------------ reading values
    def get(self, domain: str, key: str, default: Any = _MISSING) -> Any:
        """Effective value of ``domain.key`` (a private copy)."""
        with self._lock:
            try:
                return copy.deepcopy(self._resolve(domain, key))
            except KeyError:
                if default is _MISSING:
                    raise KeyError(f"{domain}.{key}") from None
                return default

    def _resolve(self, domain: str, key: str, depth: int = 0) -> Any:
        if key in self._user.get(domain, {}):
            return self._user[domain][key]
        if key in self._shipped.get(domain, {}):
            return self._shipped[domain][key]
        setting = self._registry.get(domain, {}).get(key)
        if setting is None:
            raise KeyError(f"{domain}.{key}")
        if setting.inherit and depth < 4:
            other_domain, _, other_key = setting.inherit.partition(".")
            if other_key in self._registry.get(other_domain, {}):
                return self._resolve(other_domain, other_key, depth + 1)
        return setting.default

    def source_of(self, domain: str, key: str) -> Source:
        """Which layer provides the effective value (drives the 'modified' markers in Settings)."""
        with self._lock:
            if key in self._user.get(domain, {}):
                return "user"
            if key in self._shipped.get(domain, {}):
                return "shipped"
            setting = self._registry.get(domain, {}).get(key)
            if setting is not None and setting.inherit:
                return "inherited"
            return "default"

    def is_overridden(self, domain: str, key: str) -> bool:
        """True when the user has an explicit value for ``domain.key``."""
        return self.source_of(domain, key) == "user"

    def default_of(self, domain: str, key: str) -> Any:
        """The value ``key`` would have without any user override."""
        with self._lock:
            return copy.deepcopy(self._baseline(domain, key))

    def _baseline(self, domain: str, key: str) -> Any:
        """Shipped/inherited/built-in value, ignoring the user layer (caller holds the lock)."""
        saved = self._user.get(domain, {}).pop(key, _MISSING)
        try:
            return self._resolve(domain, key)
        finally:
            if saved is not _MISSING:
                self._user[domain][key] = saved

    def section(self, domain: str, section: str = "") -> dict[str, Any]:
        """Flat ``{relative_key: value}`` of every setting below ``section``."""
        prefix = f"{section}." if section else ""
        with self._lock:
            return {
                key[len(prefix):]: copy.deepcopy(self._resolve(domain, key))
                for key in self._registry_for(domain)
                if key.startswith(prefix)
            }

    def tree(self, domain: str) -> dict[str, Any]:
        """Nested dict with the effective value of every setting of ``domain``."""
        result: dict[str, Any] = {}
        for key, value in self.section(domain).items():
            set_path(result, key, value)
        return result

    def tool_overrides(self, tool_id: str) -> dict[str, Any]:
        """Per-tool defaults from ``advanced.tool_overrides`` (not yet validated against the tool)."""
        overrides = self.get("advanced", "tool_overrides", {})
        value = overrides.get(tool_id, {}) if isinstance(overrides, dict) else {}
        return dict(value) if isinstance(value, dict) else {}

    def _snapshot(self, domain: str) -> dict[str, Any]:
        return {key: self._resolve(domain, key) for key in self._registry.get(domain, {})}

    # ------------------------------------------------------------------ writing values
    def set(self, domain: str, key: str, value: Any, *, save: bool = True) -> bool:
        """Set one value. Raises :class:`ConfigError` when it is invalid. Returns True when it changed."""
        return bool(self.set_many(domain, {key: value}, save=save))

    def set_many(self, domain: str, values: Mapping[str, Any], *, save: bool = True) -> set[str]:
        """Validate and apply several values atomically (all or nothing). Returns the changed keys."""
        changed: set[str] = set()
        with self._lock:
            registry = self._registry_for(domain)
            coerced: dict[str, Any] = {}
            errors: list[str] = []
            for key, value in values.items():
                setting = registry.get(key)
                if setting is None:
                    errors.append(f"{domain}.{key}: unknown setting")
                    continue
                try:
                    coerced[key] = setting.coerce(value)
                except ValueError as exc:
                    errors.append(f"{domain}.{key}: {exc}")
            if errors:
                raise ConfigError("; ".join(errors))
            for key, value in coerced.items():
                if self._apply(domain, key, value):
                    changed.add(key)
            if values and save:
                self._save_domain_locked(domain)
        if changed:
            self._notify(domain, frozenset(changed))
        return changed

    def _apply(self, domain: str, key: str, value: Any) -> bool:
        setting = self._registry[domain][key]
        before = self._resolve(domain, key)
        raw = self._user_raw.setdefault(domain, {})
        baseline = self._baseline(domain, key)
        if setting.inherit is None and value == baseline:
            delete_path(raw, key)
        else:
            set_path(raw, key, copy.deepcopy(value))
        self._revalidate_user(domain)
        return self._resolve(domain, key) != before

    def reset(self, domain: str, key: str | None = None, *, save: bool = True) -> set[str]:
        """Remove the user's override of one key (or of the whole domain). Returns the changed keys."""
        with self._lock:
            self._registry_for(domain)
            before = self._snapshot(domain)
            raw = self._user_raw.setdefault(domain, {})
            if key is None:
                self.backup(f"reset-{domain}", domains=[domain])
                raw.clear()
            else:
                delete_path(raw, key)
            self._revalidate_user(domain)
            changed = frozenset(k for k, v in self._snapshot(domain).items() if before.get(k) != v)
            if save:
                self._save_domain_locked(domain)
        if changed:
            self._notify(domain, changed)
        return set(changed)

    def reset_all(self) -> Path | None:
        """Restore factory defaults for every domain. The previous files are backed up first."""
        with self._lock:
            backup = self.backup("restore-defaults")
            before = {d: self._snapshot(d) for d in self._registry}
            for domain in self._registry:
                self._user_raw[domain] = {}
                self._revalidate_user(domain)
                self._save_domain_locked(domain)
            changes = {
                d: frozenset(k for k, v in self._snapshot(d).items() if before[d].get(k) != v)
                for d in self._registry
            }
        for domain, keys in changes.items():
            if keys:
                self._notify(domain, keys)
        return backup

    # ------------------------------------------------------------------ saving
    def save(self, domain: str | None = None) -> None:
        """Write the user file of one domain (or all domains)."""
        with self._lock:
            for name in [domain] if domain else list(self._registry):
                self._save_domain_locked(name)

    def _save_domain_locked(self, domain: str) -> None:
        path = self._user_path(domain)
        raw = self._user_raw.get(domain, {})
        if domain in self._broken and path.exists():
            self._stash_broken(path)
        self._broken.discard(domain)
        if not raw:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ConfigError(f"cannot remove {path}: {exc}") from exc
            self._signatures[domain] = None
            return
        text = USER_HEADER + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, default_flow_style=False)
        try:
            atomic_write_text(path, text, mode=0o600)
        except OSError as exc:
            raise ConfigError(f"cannot save {path}: {exc}") from exc
        self._signatures[domain] = self._signature(path)

    def _stash_broken(self, path: Path) -> None:
        target_dir = self.user_dir / "backups"
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        target = target_dir / f"{path.stem}.broken-{stamp}.yaml"
        try:
            shutil.copy2(path, target)
            log.warning("kept a copy of the unreadable configuration file %s as %s", path.name, target)
        except OSError:
            log.exception("could not back up the unreadable file %s", path)

    # ------------------------------------------------------------------ validation
    def issues(self, domain: str | None = None) -> list[ConfigIssue]:
        """Problems found while loading (errors first)."""
        with self._lock:
            out: list[ConfigIssue] = []
            for name in [domain] if domain else list(self._registry):
                out += self._issues_file.get(name, [])
                out += self._issues_shipped.get(name, [])
                out += self._issues_user.get(name, [])
        return sorted(out, key=lambda i: (i.severity != "error", i.domain, i.key))

    def validate(self, *, check_paths: bool = True) -> list[ConfigIssue]:
        """Re-read the files from disk and report every problem (including missing paths)."""
        self.reload()
        issues = self.issues()
        if check_paths:
            with self._lock:
                for domain, registry in self._registry.items():
                    for key, setting in registry.items():
                        if setting.kind not in (Kind.FILE, Kind.DIR, Kind.PATH):
                            continue
                        value = self._user.get(domain, {}).get(key)
                        if value and not Path(os.path.expanduser(value)).exists():
                            issues.append(ConfigIssue("warning", domain, key, f"path does not exist: {value}", "user", value))
        return issues

    def validate_tool_overrides(self, tool_id: str, settings: Iterable[Setting]) -> tuple[dict[str, Any], list[ConfigIssue]]:
        """Validate ``advanced.tool_overrides[tool_id]`` against the tool's option definitions."""
        by_key = {s.key: s for s in settings}
        valid: dict[str, Any] = {}
        issues: list[ConfigIssue] = []
        for key, value in self.tool_overrides(tool_id).items():
            setting = by_key.get(key)
            if setting is None:
                issues.append(ConfigIssue("warning", "advanced", f"tool_overrides.{tool_id}.{key}", "unknown option", "user", value))
                continue
            try:
                valid[key] = setting.coerce(value)
            except ValueError as exc:
                issues.append(ConfigIssue("error", "advanced", f"tool_overrides.{tool_id}.{key}", str(exc), "user", value))
        return valid, issues

    # ------------------------------------------------------------------ export / import / backup
    def export_bundle(
        self,
        path: Path,
        *,
        include_defaults: bool = False,
        include_sensitive: bool = False,
        domains: Iterable[str] | None = None,
    ) -> Path:
        """Write a single YAML file with the user's overrides (or the complete effective configuration).

        Machine-specific settings (executable paths, environment variables, custom
        arguments) are left out unless ``include_sensitive`` so the file can be shared safely.
        """
        path = Path(path)
        with self._lock:
            names = list(domains) if domains is not None else list(self._registry)
            payload: dict[str, Any] = {}
            for name in names:
                registry = self._registry_for(name)
                if include_defaults:
                    values = self.section(name)
                else:
                    values = {k: copy.deepcopy(v) for k, v in self._user.get(name, {}).items()}
                nested: dict[str, Any] = {}
                for key, value in values.items():
                    if registry[key].sensitive and not include_sensitive:
                        continue
                    set_path(nested, key, value)
                if nested:
                    payload[name] = nested
            bundle = {
                "format": BUNDLE_FORMAT,
                "version": BUNDLE_VERSION,
                "app_version": self.app_version,
                "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "contents": "effective" if include_defaults else "overrides",
                "domains": payload,
            }
        text = "# Linux Media & Utility Toolkit configuration export\n" + yaml.safe_dump(
            bundle, sort_keys=False, allow_unicode=True, default_flow_style=False
        )
        try:
            atomic_write_text(path, text)
        except OSError as exc:
            raise ConfigError(f"cannot write {path}: {exc}") from exc
        return path

    def import_bundle(
        self,
        path: Path,
        *,
        mode: Literal["merge", "replace"] = "merge",
        allow_sensitive: bool = False,
        dry_run: bool = False,
    ) -> ImportReport:
        """Apply an exported bundle.

        Sensitive settings (executable paths, environment, custom arguments, plugin
        loading) are skipped unless ``allow_sensitive`` - an imported file from someone
        else must not be able to point the app at arbitrary programs without the user
        noticing. ``dry_run`` only reports what would happen.
        """
        data, error = self._read_yaml(Path(path))
        if error or data is None:
            raise ConfigError(f"cannot import {path}: {error}")
        domains_data = data.get("domains") if data.get("format") == BUNDLE_FORMAT else None
        if domains_data is None:  # plain {domain: {...}} mapping written by hand
            domains_data = {k: v for k, v in data.items() if k in self._registry}
        if not isinstance(domains_data, dict) or not domains_data:
            raise ConfigError("the file does not contain any configuration for this application")

        report = ImportReport()
        accepted: dict[str, dict[str, Any]] = {}
        if data.get("format") == BUNDLE_FORMAT and _as_int(data.get("version")) > BUNDLE_VERSION:
            report.issues.append(
                ConfigIssue("warning", "", "", "the file was written by a newer version of the application", "import")
            )
        with self._lock:
            for domain, tree in domains_data.items():
                if domain not in self._registry:
                    report.issues.append(ConfigIssue("warning", str(domain), "", "unknown configuration domain (ignored)", "import"))
                    continue
                if not isinstance(tree, dict):
                    report.issues.append(ConfigIssue("error", domain, "", "expected a mapping", "import"))
                    continue
                valid, issues = validate_tree(domain, self._registry[domain], tree, source="import")
                report.issues.extend(issues)
                keep: dict[str, Any] = {}
                for key, value in valid.items():
                    if self._registry[domain][key].sensitive and not allow_sensitive:
                        report.skipped_sensitive.append(f"{domain}.{key}")
                    else:
                        keep[key] = value
                accepted[domain] = keep
            report.applied = [f"{d}.{k}" for d, kv in accepted.items() for k in kv]
            if dry_run:
                return report

            report.backup = self.backup("pre-import", domains=list(accepted))
            before = {d: self._snapshot(d) for d in accepted}
            for domain, values in accepted.items():
                if mode == "replace":
                    kept = {} if allow_sensitive else {
                        k: v for k, v in self._user.get(domain, {}).items() if self._registry[domain][k].sensitive
                    }
                    self._user_raw[domain] = {}
                    for key, value in kept.items():  # machine-specific settings survive a replace-import
                        set_path(self._user_raw[domain], key, copy.deepcopy(value))
                for key, value in values.items():
                    self._apply(domain, key, value)
                self._revalidate_user(domain)
                self._save_domain_locked(domain)
            changes = {
                d: frozenset(k for k, v in self._snapshot(d).items() if before[d].get(k) != v) for d in accepted
            }
        for domain, keys in changes.items():
            if keys:
                self._notify(domain, keys)
        return report

    def backup(self, label: str = "manual", *, domains: Iterable[str] | None = None) -> Path | None:
        """Copy the user's YAML files to ``<user_dir>/backups/<timestamp>-<label>/``. None when there is nothing."""
        with self._lock:
            names = list(domains) if domains is not None else list(self._registry)
            files = [self._user_path(d) for d in names if self._user_path(d).is_file()]
            if not files:
                return None
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label)[:40] or "backup"
            target = self.user_dir / "backups" / f"{stamp}-{safe_label}"
            try:
                target.mkdir(parents=True, exist_ok=True)
                for file in files:
                    shutil.copy2(file, target / file.name)
            except OSError:
                log.exception("configuration backup failed")
                return None
            self._prune_backups()
            return target

    def list_backups(self) -> list[Path]:
        """Existing backup folders, newest first."""
        root = self.user_dir / "backups"
        if not root.is_dir():
            return []
        return sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)

    def restore_backup(self, backup: Path) -> None:
        """Replace the current user files by the ones saved in ``backup`` (the current ones are backed up)."""
        backup = Path(backup)
        files = [p for p in backup.glob("*.yaml") if DOMAIN_RE.match(p.stem)]
        if not files:
            raise ConfigError(f"{backup} does not contain configuration files")
        with self._lock:
            self.backup("pre-restore")
            self.user_dir.mkdir(parents=True, exist_ok=True)
            for domain in self._registry:
                if self._user_path(domain).exists() and not (backup / f"{domain}.yaml").exists():
                    self._user_path(domain).unlink()
            for file in files:
                shutil.copy2(file, self._user_path(file.stem))
        self.reload()

    def _prune_backups(self) -> None:
        for old in self.list_backups()[MAX_BACKUPS:]:
            shutil.rmtree(old, ignore_errors=True)

    # ------------------------------------------------------------------ change notification
    def subscribe(self, callback: Listener) -> Callable[[], None]:
        """Call ``callback(domain, changed_keys)`` after every change. Returns an unsubscribe function."""
        with self._lock:
            self._listeners.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._listeners:
                    self._listeners.remove(callback)

        return unsubscribe

    def _notify(self, domain: str, keys: frozenset[str]) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(domain, keys)
            except Exception:  # a broken listener must not break configuration changes
                log.exception("configuration listener failed")

    def _notify_all(self) -> None:
        for domain in self.domains():
            self._notify(domain, frozenset(self._registry[domain]))

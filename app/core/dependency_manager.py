"""Detection of optional programs and Python packages.

Nothing here installs anything or runs ``sudo``: the manager only *finds*
dependencies, reports versions and builds the install command for the user's
distribution so it can be copied and run by hand.

A tool declares what it needs with dependency ids (``requires = ("ffmpeg",)``);
``"a|b"`` means any one of them is enough.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from app.core.config_manager import ConfigManager

log = logging.getLogger(__name__)

PROBE_TIMEOUT = 8.0
Family = Literal["apt", "dnf", "pacman", "zypper", "unknown"]
Source = Literal["custom", "system", "python", "missing"]


@dataclass(frozen=True, slots=True)
class DependencySpec:
    """What the application knows about one dependency."""

    id: str
    name: str
    description: str
    kind: Literal["binary", "python"]
    binaries: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("-version",)
    version_regex: str = r"(\d+(?:\.\d+)+\S*)"
    module: str = ""
    distributions: tuple[str, ...] = ()
    packages: Mapping[str, str] = field(default_factory=dict)
    pip: str = ""
    config_key: str = ""
    required: bool = False
    homepage: str = ""
    note: str = ""


def _spec(**kwargs: Any) -> DependencySpec:
    return DependencySpec(**kwargs)


SPECS: tuple[DependencySpec, ...] = (
    _spec(
        id="pillow", name="Pillow", kind="python", required=True, module="PIL", distributions=("Pillow", "pillow"),
        description="Image decoding, editing and encoding (PNG, JPEG, WebP, AVIF, GIF, BMP, TIFF, ICO).",
        pip="Pillow", packages={"apt": "python3-pil", "dnf": "python3-pillow", "pacman": "python-pillow", "zypper": "python3-Pillow"},
        homepage="https://python-pillow.org",
    ),
    _spec(
        id="ffmpeg", name="FFmpeg", kind="binary", binaries=("ffmpeg",), config_key="ffmpeg",
        version_args=("-version",), version_regex=r"version\s+(\S+)",
        description="Video and audio decoding/encoding. Needed for contact sheets and every video/audio tool.",
        packages={"apt": "ffmpeg", "dnf": "ffmpeg-free", "pacman": "ffmpeg", "zypper": "ffmpeg"},
        homepage="https://ffmpeg.org",
        note="On Fedora, 'ffmpeg-free' has a reduced codec set; install 'ffmpeg' from RPM Fusion for everything.",
    ),
    _spec(
        id="ffprobe", name="FFprobe", kind="binary", binaries=("ffprobe",), config_key="ffprobe",
        version_args=("-version",), version_regex=r"version\s+(\S+)",
        description="Reads video/audio metadata and keyframe positions. Installed together with FFmpeg.",
        packages={"apt": "ffmpeg", "dnf": "ffmpeg-free", "pacman": "ffmpeg", "zypper": "ffmpeg"},
        homepage="https://ffmpeg.org",
        note="Without FFprobe the application falls back to parsing FFmpeg's own output (no keyframe listing).",
    ),
    _spec(
        id="imagemagick", name="ImageMagick", kind="binary", binaries=("magick", "convert"), config_key="imagemagick",
        version_args=("-version",), version_regex=r"ImageMagick\s+(\S+)",
        description="Optional backend: opens formats Pillow cannot (HEIC, PSD, ...) and renders SVG.",
        packages={"apt": "imagemagick", "dnf": "ImageMagick", "pacman": "imagemagick", "zypper": "ImageMagick"},
        homepage="https://imagemagick.org",
    ),
    _spec(
        id="rsvg_convert", name="rsvg-convert (librsvg)", kind="binary", binaries=("rsvg-convert",), config_key="rsvg_convert",
        version_args=("--version",), version_regex=r"version\s+(\S+)",
        description="Optional: renders SVG vector images to bitmaps.",
        packages={"apt": "librsvg2-bin", "dnf": "librsvg2-tools", "pacman": "librsvg", "zypper": "rsvg-convert"},
        homepage="https://wiki.gnome.org/Projects/LibRsvg",
    ),
    _spec(
        id="cairosvg", name="CairoSVG", kind="python", module="cairosvg", distributions=("CairoSVG", "cairosvg"),
        description="Optional: renders SVG vector images to bitmaps (needs the cairo library).",
        pip="cairosvg", packages={"apt": "python3-cairosvg", "dnf": "python3-cairosvg", "pacman": "python-cairosvg"},
        homepage="https://cairosvg.org",
    ),
    _spec(
        id="opencv", name="OpenCV", kind="python", module="cv2",
        distributions=("opencv-python-headless", "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"),
        description="Optional backend: reads video frames when FFmpeg is missing and decodes QR codes.",
        pip="opencv-python-headless", packages={"apt": "python3-opencv", "dnf": "python3-opencv", "pacman": "python-opencv"},
        homepage="https://opencv.org",
    ),
    _spec(
        id="pillow_heif", name="pillow-heif", kind="python", module="pillow_heif", distributions=("pillow-heif",),
        description="Optional: HEIC/HEIF image support for Pillow.",
        pip="pillow-heif", homepage="https://github.com/bigcat88/pillow_heif",
    ),
    _spec(
        id="exiftool", name="ExifTool", kind="binary", binaries=("exiftool",), config_key="exiftool",
        version_args=("-ver",), version_regex=r"(\d+(?:\.\d+)+)",
        description="Optional: complete metadata viewer/editor for images, video and audio.",
        packages={"apt": "libimage-exiftool-perl", "dnf": "perl-Image-ExifTool", "pacman": "perl-image-exiftool", "zypper": "exiftool"},
        homepage="https://exiftool.org",
    ),
    _spec(
        id="poppler", name="Poppler (pdftoppm)", kind="binary", binaries=("pdftoppm",), config_key="poppler",
        version_args=("-v",), version_regex=r"version\s+(\S+)",
        description="Optional: renders PDF pages to images.",
        packages={"apt": "poppler-utils", "dnf": "poppler-utils", "pacman": "poppler", "zypper": "poppler-tools"},
        homepage="https://poppler.freedesktop.org",
    ),
    _spec(
        id="sevenzip", name="7-Zip", kind="binary", binaries=("7zz", "7z", "7za"), config_key="sevenzip",
        version_args=(), version_regex=r"7-Zip\b.*?(\d+\.\d+)",
        description="Optional: create and extract 7z archives.",
        packages={"apt": "7zip", "dnf": "7zip", "pacman": "7zip", "zypper": "7zip"},
        homepage="https://www.7-zip.org",
        note="Older Debian/Ubuntu releases call the package 'p7zip-full'.",
    ),
)
SPEC_BY_ID: dict[str, DependencySpec] = {spec.id: spec for spec in SPECS}


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    """Result of probing one dependency."""

    id: str
    name: str
    available: bool
    source: Source = "missing"
    path: str = ""
    version: str = ""
    warning: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Short human readable state, e.g. ``"6.1.1"`` or ``"not found"``."""
        if not self.available:
            return "not found"
        return self.version or "installed"


@dataclass(frozen=True, slots=True)
class InstallHint:
    """Text the user can copy to install a dependency. The application never runs it."""

    dependency: str
    family: Family
    command: str
    pip_command: str = ""
    note: str = ""


@dataclass(frozen=True, slots=True)
class Distro:
    """The Linux distribution the application runs on."""

    id: str = ""
    name: str = "Linux"
    like: tuple[str, ...] = ()
    family: Family = "unknown"


_FAMILY_IDS: tuple[tuple[Family, frozenset[str]], ...] = (
    ("apt", frozenset({"debian", "ubuntu", "linuxmint", "mint", "pop", "elementary", "zorin", "kali", "raspbian", "neon"})),
    ("dnf", frozenset({"fedora", "rhel", "centos", "rocky", "almalinux", "nobara", "ol"})),
    ("pacman", frozenset({"arch", "archlinux", "cachyos", "manjaro", "endeavouros", "garuda", "artix", "arcolinux"})),
    ("zypper", frozenset({"suse", "opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles"})),
)
_INSTALL_PREFIX: dict[str, str] = {
    "apt": "sudo apt install",
    "dnf": "sudo dnf install",
    "pacman": "sudo pacman -S",
    "zypper": "sudo zypper install",
}


def detect_distro(os_release: Path | None = Path("/etc/os-release")) -> Distro:
    """Parse ``/etc/os-release`` (ID / ID_LIKE) into a package-manager family."""
    values: dict[str, str] = {}
    try:
        text = Path(os_release).read_text(encoding="utf-8", errors="replace") if os_release else ""
    except OSError:
        text = ""
    for line in text.splitlines():
        key, sep, raw = line.partition("=")
        if sep and key.strip() in {"ID", "ID_LIKE", "NAME", "PRETTY_NAME"}:
            values[key.strip()] = raw.strip().strip("'\"")
    distro_id = values.get("ID", "").lower()
    like = tuple(values.get("ID_LIKE", "").lower().split())
    candidates = (distro_id, *like)
    family: Family = "unknown"
    for fam, ids in _FAMILY_IDS:
        if any(c in ids for c in candidates):
            family = fam
            break
    return Distro(distro_id, values.get("PRETTY_NAME") or values.get("NAME") or "Linux", like, family)


class DependencyManager:
    """Finds dependencies, caches the result and explains how to install what is missing."""

    def __init__(
        self,
        config: ConfigManager | None = None,
        *,
        specs: Iterable[DependencySpec] = SPECS,
        distro: Distro | None = None,
        which: Callable[[str, str | None], str | None] | None = None,
    ) -> None:
        self._config = config
        self._specs: dict[str, DependencySpec] = {spec.id: spec for spec in specs}
        self._distro = distro
        self._which = which or (lambda name, path: shutil.which(name, path=path))
        self._cache: dict[str, DependencyStatus] = {}
        self._lock = threading.RLock()
        self._unsubscribe: Callable[[], None] | None = None
        if config is not None:
            self._unsubscribe = config.subscribe(self._on_config_change)

    # ------------------------------------------------------------------ specs
    def specs(self) -> list[DependencySpec]:
        """All known dependencies in display order."""
        return list(self._specs.values())

    def spec(self, dep_id: str) -> DependencySpec:
        """Specification of ``dep_id``; raises ``KeyError`` for unknown ids."""
        return self._specs[dep_id]

    def register(self, spec: DependencySpec) -> None:
        """Let a plugin declare its own dependency."""
        with self._lock:
            self._specs[spec.id] = spec
            self._cache.pop(spec.id, None)

    # ------------------------------------------------------------------ configuration
    def _on_config_change(self, domain: str, keys: frozenset[str]) -> None:
        if domain == "advanced" and any(k.startswith(("paths.", "environment")) for k in keys):
            self.invalidate()

    def _custom_path(self, spec: DependencySpec) -> str:
        if self._config is None or not spec.config_key:
            return ""
        try:
            return str(self._config.get("advanced", f"paths.{spec.config_key}", "") or "")
        except KeyError:
            return ""

    def environment(self) -> dict[str, str]:
        """Environment for child processes: ours plus ``advanced.environment`` (validated by the schema)."""
        env = dict(os.environ)
        if self._config is not None:
            extra = self._config.get("advanced", "environment", {}) or {}
            env.update({str(k): str(v) for k, v in extra.items()})
        return env

    # ------------------------------------------------------------------ probing
    def invalidate(self, dep_id: str | None = None) -> None:
        """Forget cached results (after changing custom paths, installing something...)."""
        with self._lock:
            if dep_id is None:
                self._cache.clear()
            else:
                self._cache.pop(dep_id, None)

    def cached(self, dep_id: str) -> DependencyStatus | None:
        """Last known status without probing (never blocks)."""
        with self._lock:
            return self._cache.get(dep_id)

    def is_checked(self) -> bool:
        """True once every dependency has been probed at least once."""
        with self._lock:
            return all(dep_id in self._cache for dep_id in self._specs)

    def status(self, dep_id: str, *, refresh: bool = False) -> DependencyStatus:
        """Status of one dependency, probing it when it is not cached yet."""
        with self._lock:
            if not refresh and dep_id in self._cache:
                return self._cache[dep_id]
            spec = self._specs[dep_id]
        result = self._probe(spec)
        with self._lock:
            self._cache[dep_id] = result
        return result

    def check_all(self, *, refresh: bool = True) -> dict[str, DependencyStatus]:
        """Probe every dependency (call from a worker thread)."""
        return {dep_id: self.status(dep_id, refresh=refresh) for dep_id in list(self._specs)}

    def available(self, dep_id: str) -> bool:
        """True when ``dep_id`` is usable."""
        try:
            return self.status(dep_id).available
        except KeyError:
            return False

    def path(self, dep_id: str) -> str | None:
        """Absolute path of the executable (binary dependencies) or ``None``."""
        status = self.status(dep_id)
        return status.path if status.available and status.path else None

    def _probe(self, spec: DependencySpec) -> DependencyStatus:
        try:
            if spec.kind == "python":
                return self._probe_python(spec)
            return self._probe_binary(spec)
        except Exception as exc:  # detection must never crash the application
            log.exception("dependency probe for %s failed", spec.id)
            return DependencyStatus(spec.id, spec.name, False, warning=f"probe failed: {exc}")

    def _probe_python(self, spec: DependencySpec) -> DependencyStatus:
        try:
            found = importlib.util.find_spec(spec.module)
        except (ImportError, ValueError):
            found = None
        if found is None:
            return DependencyStatus(spec.id, spec.name, False)
        version = ""
        for dist in spec.distributions:
            try:
                version = importlib.metadata.version(dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        details: dict[str, Any] = {}
        if spec.id == "pillow":
            details["features"] = _pillow_features()
        return DependencyStatus(
            spec.id, spec.name, True, "python", str(found.origin or ""), version, details=details
        )

    def _probe_binary(self, spec: DependencySpec) -> DependencyStatus:
        warning = ""
        custom = self._custom_path(spec)
        env = self.environment()
        if custom:
            expanded = os.path.expanduser(custom)
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                version, error = self._run_version(spec, expanded, env)
                if error:
                    warning = f"custom path {custom} did not run: {error}"
                else:
                    return DependencyStatus(spec.id, spec.name, True, "custom", expanded, version)
            else:
                warning = f"custom path {custom} is not an executable file"
        search_path = env.get("PATH")
        for name in spec.binaries:
            located = self._which(name, search_path)
            if not located:
                continue
            version, error = self._run_version(spec, located, env)
            if error:
                warning = warning or f"{name} was found but did not run: {error}"
                continue
            note = f"{warning}; using the system {name}" if warning else ""
            return DependencyStatus(spec.id, spec.name, True, "system", located, version, warning=note)
        return DependencyStatus(spec.id, spec.name, False, warning=warning)

    def _run_version(self, spec: DependencySpec, executable: str, env: Mapping[str, str]) -> tuple[str, str]:
        """Run ``executable`` to read its version. Returns ``(version, error)``."""
        try:
            completed = subprocess.run(  # noqa: S603 - argument list, no shell
                [executable, *spec.version_args],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=PROBE_TIMEOUT,
                stdin=subprocess.DEVNULL,
                env=dict(env),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "", "timed out"
        except OSError as exc:
            return "", exc.strerror or str(exc)
        output = f"{completed.stdout}\n{completed.stderr}"
        match = re.search(spec.version_regex, output)
        if match is None and completed.returncode != 0:
            return "", f"exit code {completed.returncode}"
        return (match.group(1) if match else ""), ""

    # ------------------------------------------------------------------ requirements
    def missing(self, requires: Sequence[str]) -> list[str]:
        """Unsatisfied entries of ``requires`` (``"a|b"`` = any of; reported as written)."""
        return [req for req in requires if not any(self.available(alt) for alt in req.split("|"))]

    def satisfied(self, requires: Sequence[str]) -> bool:
        """True when every requirement is met."""
        return not self.missing(requires)

    # ------------------------------------------------------------------ install help
    def distro(self) -> Distro:
        """The detected distribution (cached)."""
        if self._distro is None:
            self._distro = detect_distro()
        return self._distro

    def install_hint(self, dep_id: str, family: Family | None = None) -> InstallHint | None:
        """Install command for ``dep_id`` on ``family`` (default: this system's distribution)."""
        spec = self._specs.get(dep_id)
        if spec is None:
            return None
        fam: Family = family or self.distro().family
        pip_command = f"python3 -m pip install {spec.pip}" if spec.pip else ""
        package = spec.packages.get(fam, "") if fam != "unknown" else ""
        command = f"{_INSTALL_PREFIX[fam]} {package}" if package else ""
        if not command and not pip_command:
            return None
        return InstallHint(dep_id, fam, command, pip_command, spec.note)

    def install_command_for(self, dep_ids: Iterable[str], family: Family | None = None) -> str:
        """One command that installs every given dependency's system package (empty when unknown)."""
        fam: Family = family or self.distro().family
        if fam == "unknown":
            return ""
        packages: list[str] = []
        for dep_id in dep_ids:
            spec = self._specs.get(dep_id)
            package = spec.packages.get(fam, "") if spec else ""
            if package and package not in packages:
                packages.append(package)
        return f"{_INSTALL_PREFIX[fam]} {' '.join(packages)}" if packages else ""

    def explain(self, requires: Sequence[str]) -> str:
        """Message for a tool that cannot run: what is missing and how to install it."""
        lines: list[str] = []
        for requirement in self.missing(requires):
            alternatives = [self._specs[a] for a in requirement.split("|") if a in self._specs]
            if not alternatives:
                lines.append(f"Missing: {requirement}")
                continue
            names = " or ".join(spec.name for spec in alternatives)
            lines.append(f"Missing: {names} - {alternatives[0].description}")
            hint = self.install_hint(alternatives[0].id)
            if hint:
                if hint.command:
                    lines.append(f"  Install: {hint.command}")
                if hint.pip_command:
                    lines.append(f"  or: {hint.pip_command}")
                if hint.note:
                    lines.append(f"  Note: {hint.note}")
        return "\n".join(lines)


def _pillow_features() -> dict[str, bool]:
    """Which optional codecs this Pillow build was compiled with."""
    try:
        from PIL import features
    except ImportError:  # pragma: no cover - Pillow is a hard requirement
        return {}
    result: dict[str, bool] = {}
    for name in ("webp", "avif", "jpg", "jpg_2000", "libtiff", "freetype2", "raqm", "littlecms2", "zlib"):
        try:
            result[name] = bool(features.check(name))
        except Exception:
            result[name] = False
    return result

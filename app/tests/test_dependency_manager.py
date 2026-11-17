"""Tests for the dependency checker and install hints."""

from __future__ import annotations

import shlex
import stat
from pathlib import Path

import pytest

from app.core.config_manager import ConfigManager
from app.core.dependency_manager import (
    DependencyManager,
    Distro,
    detect_distro,
)


def make_binary(directory: Path, name: str, output: str, *, code: int = 0, stream: str = "stdout") -> Path:
    """Create a fake executable that prints ``output`` and exits with ``code`` (shell builtins only)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    redirect = " >&2" if stream == "stderr" else ""
    lines = " ".join(shlex.quote(line) for line in output.splitlines())
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' {lines}{redirect}\nexit {code}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def bindir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated PATH that only contains our fake programs."""
    directory = tmp_path / "bin"
    directory.mkdir()
    monkeypatch.setenv("PATH", str(directory))
    return directory


class TestBinaries:
    def test_detects_version_and_path(self, bindir: Path) -> None:
        exe = make_binary(bindir, "ffmpeg", "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023")
        status = DependencyManager().status("ffmpeg")
        assert status.available and status.version == "6.1.1-3ubuntu5"
        assert status.path == str(exe) and status.source == "system"

    def test_missing_binary(self, bindir: Path) -> None:
        status = DependencyManager().status("ffprobe")
        assert not status.available and status.label == "not found" and status.path == ""

    def test_imagemagick_accepts_legacy_convert(self, bindir: Path) -> None:
        make_binary(bindir, "convert", "Version: ImageMagick 6.9.12-98 Q16 x86_64")
        status = DependencyManager().status("imagemagick")
        assert status.available and status.version == "6.9.12-98"

    def test_version_printed_on_stderr(self, bindir: Path) -> None:
        make_binary(bindir, "pdftoppm", "pdftoppm version 24.02.0\nCopyright 2005-2024", stream="stderr")
        assert DependencyManager().status("poppler").version == "24.02.0"

    def test_sevenzip_banner_variants(self, bindir: Path) -> None:
        make_binary(bindir, "7z", "7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov")
        assert DependencyManager().status("sevenzip").version == "16.02"
        make_binary(bindir, "7zz", "7-Zip (z) 23.01 (x64) : Copyright (c) 1999-2023")
        assert DependencyManager().status("sevenzip", refresh=True).version == "23.01"

    def test_broken_binary_is_not_available(self, bindir: Path) -> None:
        make_binary(bindir, "ffmpeg", "boom", code=127)
        status = DependencyManager().status("ffmpeg")
        assert not status.available and "did not run" in status.warning

    def test_custom_path_wins_and_invalid_custom_path_falls_back(self, tmp_path: Path, bindir: Path, user_dir: Path) -> None:
        system = make_binary(bindir, "ffmpeg", "ffmpeg version 1.0")
        custom = make_binary(tmp_path / "custom", "my-ffmpeg", "ffmpeg version 2.0")
        config = ConfigManager(user_dir)
        deps = DependencyManager(config)
        assert deps.status("ffmpeg").path == str(system)

        config.set("advanced", "paths.ffmpeg", str(custom))  # config change invalidates the cache
        status = deps.status("ffmpeg")
        assert status.path == str(custom) and status.version == "2.0" and status.source == "custom"

        config.set("advanced", "paths.ffmpeg", str(tmp_path / "nope"))
        status = deps.status("ffmpeg")
        assert status.path == str(system) and "not an executable file" in status.warning

    def test_environment_comes_from_config(self, user_dir: Path, bindir: Path) -> None:
        make_binary(bindir, "ffmpeg", "ffmpeg version 1.0")
        config = ConfigManager(user_dir)
        config.set("advanced", "environment", {"FOO": "bar"})
        assert DependencyManager(config).environment()["FOO"] == "bar"

    def test_cache_and_invalidate(self, bindir: Path) -> None:
        deps = DependencyManager()
        assert deps.cached("ffmpeg") is None and not deps.is_checked()
        assert not deps.available("ffmpeg")
        make_binary(bindir, "ffmpeg", "ffmpeg version 1.0")
        assert not deps.available("ffmpeg")  # cached
        deps.invalidate("ffmpeg")
        assert deps.available("ffmpeg")
        deps.check_all()
        assert deps.is_checked()


class TestPython:
    def test_pillow_is_detected_with_features(self) -> None:
        status = DependencyManager().status("pillow")
        assert status.available and status.source == "python" and status.version
        assert "webp" in status.details["features"]

    def test_missing_module(self) -> None:
        status = DependencyManager().status("cairosvg") if not _has("cairosvg") else None
        if status is not None:
            assert not status.available and status.label == "not found"


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


class TestRequirements:
    def test_any_of_requirement(self, bindir: Path) -> None:
        deps = DependencyManager()
        assert deps.missing(["ffmpeg", "imagemagick|rsvg_convert"]) == ["ffmpeg", "imagemagick|rsvg_convert"]
        make_binary(bindir, "rsvg-convert", "rsvg-convert version 2.58.0")
        deps.invalidate()
        assert deps.missing(["ffmpeg", "imagemagick|rsvg_convert"]) == ["ffmpeg"]
        assert not deps.satisfied(["ffmpeg"])
        assert deps.satisfied(["imagemagick|rsvg_convert", "pillow"])

    def test_unknown_dependency_is_unavailable_not_an_error(self, bindir: Path) -> None:
        assert DependencyManager().missing(["does-not-exist"]) == ["does-not-exist"]


class TestDistroHints:
    @pytest.mark.parametrize(
        ("os_release", "family"),
        [
            ('ID=ubuntu\nID_LIKE=debian\nPRETTY_NAME="Ubuntu 24.04 LTS"\n', "apt"),
            ("ID=linuxmint\nID_LIKE=\"ubuntu debian\"\n", "apt"),
            ("ID=fedora\nPRETTY_NAME=\"Fedora Linux 40\"\n", "dnf"),
            ("ID=arch\n", "pacman"),
            ("ID=cachyos\nID_LIKE=arch\n", "pacman"),
            ("ID=opensuse-tumbleweed\nID_LIKE=\"opensuse suse\"\n", "zypper"),
            ("ID=gentoo\n", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_detect_family(self, tmp_path: Path, os_release: str, family: str) -> None:
        path = tmp_path / "os-release"
        path.write_text(os_release)
        assert detect_distro(path).family == family

    def test_missing_os_release(self, tmp_path: Path) -> None:
        assert detect_distro(tmp_path / "nope").family == "unknown"

    def test_install_hints_per_distro_are_display_only(self) -> None:
        ubuntu = DependencyManager(distro=Distro("ubuntu", "Ubuntu", ("debian",), "apt"))
        assert ubuntu.install_hint("ffmpeg").command == "sudo apt install ffmpeg"
        assert ubuntu.install_hint("poppler").command == "sudo apt install poppler-utils"
        fedora = DependencyManager(distro=Distro("fedora", "Fedora", (), "dnf"))
        assert fedora.install_hint("exiftool").command == "sudo dnf install perl-Image-ExifTool"
        assert "RPM Fusion" in fedora.install_hint("ffmpeg").note
        arch = DependencyManager(distro=Distro("arch", "Arch", (), "pacman"))
        assert arch.install_hint("opencv").command == "sudo pacman -S python-opencv"
        assert arch.install_hint("opencv").pip_command == "python3 -m pip install opencv-python-headless"

    def test_unknown_distro_still_offers_pip(self) -> None:
        deps = DependencyManager(distro=Distro())
        hint = deps.install_hint("opencv")
        assert hint is not None and hint.command == "" and "pip install" in hint.pip_command
        assert deps.install_hint("ffmpeg") is None
        assert deps.install_hint("nonexistent") is None

    def test_combined_install_command_deduplicates_packages(self) -> None:
        deps = DependencyManager(distro=Distro("debian", "Debian", (), "apt"))
        assert deps.install_command_for(["ffmpeg", "ffprobe", "imagemagick"]) == "sudo apt install ffmpeg imagemagick"
        assert DependencyManager(distro=Distro()).install_command_for(["ffmpeg"]) == ""

    def test_explain_lists_missing_and_how_to_fix(self, bindir: Path) -> None:
        deps = DependencyManager(distro=Distro("ubuntu", "Ubuntu", (), "apt"))
        text = deps.explain(["ffmpeg", "pillow"])
        assert "FFmpeg" in text and "sudo apt install ffmpeg" in text and "Pillow" not in text
        assert deps.explain(["pillow"]) == ""

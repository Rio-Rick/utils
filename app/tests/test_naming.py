from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

import pytest

from app.core.errors import ToolError
from app.core.naming import OutputPlanner, render_template, sanitize_filename


def test_sanitize_removes_separators_and_dots():
    assert sanitize_filename("../../etc/passwd") == "etcpasswd"
    assert sanitize_filename("a/b\\c:d") == "abcd"
    assert sanitize_filename("  ..hidden.. ") == "hidden"
    assert sanitize_filename("") == "output"
    assert sanitize_filename("\x00\x01") == "output"
    assert sanitize_filename("con") == "_con"


def test_sanitize_limits_utf8_length():
    result = sanitize_filename("é" * 500)
    assert len(result.encode()) <= 200
    assert result  # still valid text


def test_render_template_basic_and_padding():
    fields = {"name": "photo", "index": 7, "width": 640}
    assert render_template("{name}_{width}", fields) == "photo_640"
    assert render_template("{name}-{index:03d}", fields) == "photo-007"
    assert render_template("{index:3d}", fields) == "  7"


@pytest.mark.parametrize(
    "template",
    ["{name.__class__}", "{name!r}", "{0}", "{name[0]}", "{missing}", "{", "{name:>10}", "{__import__}"],
)
def test_render_template_rejects_unsafe_or_unknown(template):
    with pytest.raises(ToolError):
        render_template(template, {"name": "x", "index": 1})


def test_plan_subfolder_and_rename(tmp_path: Path):
    src = tmp_path / "pic.png"
    src.write_bytes(b"x")
    planner = OutputPlanner(location="subfolder", subfolder_name="out", overwrite="rename")
    first = planner.plan(src, "webp")
    second = planner.plan(src, "webp")
    assert first.path == tmp_path / "out" / "pic.webp"
    assert second.path == tmp_path / "out" / "pic (1).webp"
    assert first.path.exists() and second.path.exists()  # reserved


def test_plan_never_targets_the_source(tmp_path: Path):
    src = tmp_path / "pic.png"
    src.write_bytes(b"x")
    for mode in ("rename", "overwrite", "skip"):
        planner = OutputPlanner(location="same", overwrite=mode)
        target = planner.plan(src, "png")
        assert target.path != src
        assert not target.skip
        target.discard()


def test_plan_overwrite_and_skip(tmp_path: Path):
    src = tmp_path / "a.png"
    src.write_bytes(b"x")
    existing = tmp_path / "out" / "a.jpg"
    existing.parent.mkdir()
    existing.write_bytes(b"old")
    over = OutputPlanner(location="subfolder", subfolder_name="out", overwrite="overwrite").plan(src, "jpg")
    assert over.path == existing and not over.skip and not over.reserved
    skip = OutputPlanner(location="subfolder", subfolder_name="out", overwrite="skip").plan(src, "jpg")
    assert skip.skip and skip.path == existing
    assert existing.read_bytes() == b"old"


def test_discard_removes_empty_placeholder_only(tmp_path: Path):
    src = tmp_path / "a.png"
    src.write_bytes(b"x")
    planner = OutputPlanner(location="same")
    with pytest.raises(RuntimeError):
        with planner.plan(src, "jpg") as target:
            path = target.path
            assert path.exists()
            raise RuntimeError("boom")
    assert not path.exists()

    target = planner.plan(src, "jpg")
    target.path.write_bytes(b"data")
    target.discard()
    assert target.path.exists()


def test_custom_directory_and_structure(tmp_path: Path):
    base = tmp_path / "in"
    (base / "sub").mkdir(parents=True)
    src = base / "sub" / "x.png"
    src.write_bytes(b"x")
    out = tmp_path / "result"
    planner = OutputPlanner(location="custom", directory=str(out), preserve_structure=True)
    assert planner.plan(src, "jpg", base_dir=base).path == out / "sub" / "x.jpg"
    flat = OutputPlanner(location="custom", directory=str(out), preserve_structure=False)
    assert flat.plan(src, "jpg", base_dir=base).path == out / "x.jpg"


def test_subfolder_is_rooted_at_the_dropped_folder(tmp_path: Path):
    base = tmp_path / "photos"
    (base / "2024").mkdir(parents=True)
    src = base / "2024" / "x.png"
    src.write_bytes(b"x")
    planner = OutputPlanner(location="subfolder", subfolder_name="converted", preserve_structure=True)
    assert planner.plan(src, "webp", base_dir=base).path == base / "converted" / "2024" / "x.webp"


def test_custom_without_directory_uses_fallback(tmp_path: Path):
    src = tmp_path / "x.png"
    src.write_bytes(b"x")
    fallback = tmp_path / "fb"
    planner = OutputPlanner(location="custom", directory="", fallback_directory=str(fallback))
    assert planner.plan(src, "jpg").path == fallback / "x.jpg"


def test_template_fields_and_traversal_is_neutralised(tmp_path: Path):
    src = tmp_path / "x.png"
    src.write_bytes(b"x")
    stamp = datetime(2024, 5, 6, 7, 8, 9)
    planner = OutputPlanner(location="same", template="{name}_{width}x{height}_{date}_{index:03d}")
    target = planner.plan(src, "jpg", {"width": 10, "height": 20}, index=4, now=stamp)
    assert target.path.name == "x_10x20_2024-05-06_004.jpg"

    evil = OutputPlanner(location="same", template="../../{name}")
    assert evil.plan(src, "jpg").path.parent == tmp_path


def test_parallel_reservations_never_collide(tmp_path: Path):
    src = tmp_path / "x.png"
    src.write_bytes(b"x")
    planner = OutputPlanner(location="same", overwrite="rename")
    paths: list[Path] = []
    lock = threading.Lock()

    def worker() -> None:
        target = planner.plan(src, "jpg")
        with lock:
            paths.append(target.path)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(paths)) == 16


def test_sibling_dir_is_unique(tmp_path: Path):
    src = tmp_path / "movie.mp4"
    src.write_bytes(b"x")
    planner = OutputPlanner(location="same", overwrite="rename")
    a = planner.sibling_dir(src, "movie_frames")
    b = planner.sibling_dir(src, "movie_frames")
    assert a != b and a.is_dir() and b.is_dir()
    assert b.name == "movie_frames (1)"


def test_unwritable_output_dir_is_a_tool_error(tmp_path: Path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    src = tmp_path / "x.png"
    src.write_bytes(b"x")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        planner = OutputPlanner(location="custom", directory=str(locked / "inner"))
        with pytest.raises(ToolError):
            planner.plan(src, "jpg")
    finally:
        locked.chmod(0o700)


def test_from_options_reads_prefixed_keys(tmp_path: Path):
    planner = OutputPlanner.from_options(
        {
            "output.location": "custom",
            "output.directory": str(tmp_path),
            "output.naming_template": "{name}_x",
            "output.overwrite": "skip",
            "output.subfolder_name": "a/b",
            "output.preserve_structure": False,
        }
    )
    assert planner.location == "custom"
    assert planner.template == "{name}_x"
    assert planner.overwrite == "skip"
    assert planner.subfolder_name == "ab"
    assert planner.preserve_structure is False

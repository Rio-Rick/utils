from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.inputs import collect_files, natural_key


def touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def names(result) -> list[str]:
    return [p.name for p in result.paths]


def test_natural_order():
    items = ["img10.png", "img2.png", "IMG1.png", "a.png"]
    assert sorted(items, key=natural_key) == ["a.png", "IMG1.png", "img2.png", "img10.png"]


def test_files_and_folders_are_expanded_filtered_and_sorted(tmp_path: Path):
    touch(tmp_path / "set" / "b10.png")
    touch(tmp_path / "set" / "b2.png")
    touch(tmp_path / "set" / "notes.txt")
    touch(tmp_path / "set" / ".hidden.png")
    touch(tmp_path / "set" / "sub" / "deep.PNG")
    touch(tmp_path / "set" / ".git" / "x.png")
    single = touch(tmp_path / "single.png")
    result = collect_files([tmp_path / "set", single], extensions={"png"})
    assert names(result) == ["b2.png", "b10.png", "deep.PNG", "single.png"]
    assert result.ignored == 1  # notes.txt inside the folder
    assert result.skipped == []
    bases = {p.name: base for p, base in result.files}
    assert bases["b2.png"] == tmp_path / "set" and bases["deep.PNG"] == tmp_path / "set" and bases["single.png"] is None


def test_explicit_wrong_type_missing_and_duplicates_are_reported(tmp_path: Path):
    text = touch(tmp_path / "a.txt")
    png = touch(tmp_path / "a.png")
    result = collect_files([text, tmp_path / "missing.png", png, png, tmp_path], extensions={"png"})
    assert names(result) == ["a.png"]
    reasons = dict(result.skipped)
    assert reasons[text] == "unsupported file type"
    assert reasons[tmp_path / "missing.png"] == "not found"


def test_non_recursive_and_skip_dirs(tmp_path: Path):
    touch(tmp_path / "top.png")
    touch(tmp_path / "sub" / "inner.png")
    touch(tmp_path / "converted" / "old.png")
    assert names(collect_files([tmp_path], extensions={"png"}, recursive=False)) == ["top.png"]
    found = names(collect_files([tmp_path], extensions={"png"}, skip_dirs=["converted"]))
    assert "old.png" not in found and set(found) == {"top.png", "inner.png"}


def test_limit_marks_truncated_only_when_more_files_exist(tmp_path: Path):
    for i in range(5):
        touch(tmp_path / f"f{i}.png")
    touch(tmp_path / "other.txt")
    exact = collect_files([tmp_path], extensions={"png"}, limit=5)
    assert len(exact.files) == 5 and not exact.truncated
    cut = collect_files([tmp_path], extensions={"png"}, limit=3)
    assert len(cut.files) == 3 and cut.truncated


def test_no_extension_filter_accepts_everything(tmp_path: Path):
    touch(tmp_path / "a.bin")
    touch(tmp_path / "noext")
    assert set(names(collect_files([tmp_path]))) == {"a.bin", "noext"}


def test_symlinks_are_not_followed(tmp_path: Path):
    real = tmp_path / "real"
    touch(real / "a.png")
    (real / "loop").symlink_to(tmp_path, target_is_directory=True)
    (real / "dangling.png").symlink_to(tmp_path / "nowhere.png")
    result = collect_files([real], extensions={"png"})
    assert names(result) == ["a.png"]


def test_cancel_stops_the_scan(tmp_path: Path):
    for i in range(10):
        touch(tmp_path / f"f{i}.png")
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    result = collect_files([tmp_path], extensions={"png"}, cancelled=cancelled)
    assert len(result.files) < 10


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unreadable_folder_is_reported(tmp_path: Path):
    locked = tmp_path / "locked"
    touch(locked / "a.png")
    locked.chmod(0)
    try:
        result = collect_files([tmp_path], extensions={"png"})
    finally:
        locked.chmod(0o755)
    assert any(path.name == "locked" for path, _ in result.skipped)

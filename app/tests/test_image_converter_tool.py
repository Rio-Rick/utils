from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from app.core.errors import Cancelled, ToolError
from app.tools.base import RunContext, WorkItem
from app.tools.image.converter import ImageConverter, settings_from_options


def noise(path: Path, size=(160, 120)) -> Path:
    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(path)
    return path


@pytest.fixture
def tool(services) -> ImageConverter:
    found = services.plugins.get("image.converter")
    assert isinstance(found, ImageConverter)
    return found


def run(tool: ImageConverter, services, source: Path, base: Path | None = None, **overrides):
    options = tool.resolve_options(services, overrides=overrides)
    ctx = RunContext(services)
    return tool.run_item(WorkItem(source, base, extra={"index": 1}), options, ctx)


def test_defaults_are_valid_and_cover_every_documented_group(tool):
    options = tool.validate_options({})
    assert options["format"] == "webp" and options["quality"] == 85
    groups = {s.group for s in tool.options}
    assert {"Format", "Resize", "Crop", "Transform", "Metadata", "JPEG", "PNG", "WebP", "AVIF", "TIFF", "GIF", "ICO", "Output"} <= groups
    assert tool.supported_extensions() >= {"png", "jpg", "webp", "svg", "gif"}
    assert tool.available_choices("format")[0] == "same" and "png" in tool.available_choices("format")
    assert tool.available_choices("quality") is None


def test_settings_are_translated_for_the_engine(tool):
    options = tool.validate_options({"format": "jpeg", "quality": 70, "jpeg.quality": 55, "resize.mode": "fit", "resize.width": 300,
                                     "ico.sizes": ["64", "16", "64"], "metadata.preserve": True})
    s = settings_from_options(options, max_pixels=123)
    assert (s.format, s.quality, s.resize_mode, s.width, s.ico_sizes) == ("jpeg", 55, "fit", 300, (16, 64))
    assert s.preserve_metadata and s.max_pixels == 123
    assert settings_from_options(options, "png").quality == 70  # no png specific override


def test_invalid_combinations_are_rejected_before_any_work(tool):
    with pytest.raises(ToolError, match="width or a height"):
        tool.validate_options({"resize.mode": "fit"})
    with pytest.raises(ToolError, match="both"):
        tool.validate_options({"resize.mode": "fill", "resize.width": 100})
    with pytest.raises(ToolError):
        tool.validate_options({"crop.mode": "aspect", "crop.aspect": "wide"})
    with pytest.raises(ToolError):
        tool.validate_options({"quality": 0})


def test_conversion_writes_next_to_the_source_in_a_subfolder(tool, services, tmp_path):
    src = noise(tmp_path / "photo.png")
    result = run(tool, services, src, **{"format": "jpeg", "quality": 60})
    out = tmp_path / "converted" / "photo.jpg"
    assert result.outputs == [out] and out.exists()
    assert result.meta["output_format"] == "JPEG" and result.meta["input_format"] == "PNG"
    assert result.meta["input_size"] == src.stat().st_size and result.meta["output_size"] == out.stat().st_size
    assert result.meta["dimensions"] == "160x120"
    assert "\u2192" in result.message
    assert src.exists()  # the source is never touched


def test_naming_template_can_use_dimensions_and_format(tool, services, tmp_path):
    src = noise(tmp_path / "pic.png")
    result = run(tool, services, src, **{"format": "webp", "resize.mode": "fit", "resize.width": 80,
                                         "output.naming_template": "{name}_{width}x{height}_{format}_q{quality}"})
    assert result.outputs[0].name == "pic_80x60_webp_q85.webp"


def test_same_format_follows_the_source_and_svg_falls_back_to_png(tool, services, tmp_path, qapp):
    jpg = tmp_path / "a.jpg"
    Image.new("RGB", (20, 20), "red").save(jpg)
    out = run(tool, services, jpg, format="same", **{"output.location": "same"}).outputs[0]
    assert out.name == "a (1).jpg"  # same folder + same name: renamed, never overwriting the source
    assert Image.open(out).format == "JPEG"
    svg = tmp_path / "v.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>')
    assert run(tool, services, svg, format="same").outputs[0].suffix == ".png"


def test_overwrite_policies(tool, services, tmp_path):
    src = noise(tmp_path / "x.png")
    first = run(tool, services, src, format="png").outputs[0]
    second = run(tool, services, src, format="png").outputs[0]
    assert second.name == "x (1).png"
    skipped = run(tool, services, src, format="png", **{"output.overwrite": "skip"})
    assert skipped.skipped and skipped.outputs == [first]
    before = first.stat().st_mtime_ns
    replaced = run(tool, services, src, format="png", **{"output.overwrite": "overwrite"})
    assert replaced.outputs == [first] and not replaced.skipped and first.stat().st_mtime_ns >= before


def test_skip_does_not_decode_the_source(tool, services, tmp_path):
    src = tmp_path / "broken.png"
    src.write_bytes(b"definitely not a png")
    (tmp_path / "converted").mkdir()
    (tmp_path / "converted" / "broken.webp").write_bytes(b"old")
    assert run(tool, services, src, **{"output.overwrite": "skip"}).skipped


def test_failures_leave_no_placeholder_files(tool, services, tmp_path):
    src = tmp_path / "broken.png"
    src.write_bytes(b"definitely not a png")
    with pytest.raises(ToolError):
        run(tool, services, src)
    assert list((tmp_path / "converted").glob("*")) == []

    good = noise(tmp_path / "good.png")
    ctx = RunContext(services)
    ctx.control.cancel()
    options = tool.resolve_options(services)
    with pytest.raises(Cancelled):
        tool.run_item(WorkItem(good, None), options, ctx)
    assert list((tmp_path / "converted").glob("*")) == []


def test_folder_structure_is_mirrored_under_the_output_folder(tool, services, tmp_path):
    root = tmp_path / "album"
    (root / "trip").mkdir(parents=True)
    nested = noise(root / "trip" / "beach.png")
    scan = tool.expand_inputs([root], tool.resolve_options(services))
    items = tool.make_items(scan, tool.resolve_options(services))
    assert [i.source for i in items] == [nested] and items[0].base_dir == root
    result = run(tool, services, items[0].source, items[0].base_dir)
    assert result.outputs[0] == root / "converted" / "trip" / "beach.webp"
    # a second scan does not pick the results up again
    again = tool.expand_inputs([root], tool.resolve_options(services))
    assert again.paths == [nested]


def test_custom_output_directory_and_application_fallback(tool, services, tmp_path):
    src = noise(tmp_path / "x.png")
    target = tmp_path / "elsewhere"
    result = run(tool, services, src, **{"output.location": "custom", "output.directory": str(target)})
    assert result.outputs[0].parent == target

    services.config.set("app", "directories.output", str(tmp_path / "default-out"))
    result = run(tool, services, src, **{"output.location": "custom"})
    assert result.outputs[0].parent == tmp_path / "default-out"


def test_max_pixels_limit_comes_from_the_configuration(tool, services, tmp_path):
    services.config.set("advanced", "limits.max_image_pixels", 1_000_000)
    big = tmp_path / "big.png"
    Image.new("RGB", (1200, 1000), "blue").save(big)
    with pytest.raises(ToolError, match="too large"):
        run(tool, services, big)
    assert run(tool, services, noise(tmp_path / "small.png")).outputs


def test_target_size_and_progress_reporting(tool, services, tmp_path):
    src = noise(tmp_path / "n.png", (300, 300))
    seen: list[tuple[float, str]] = []
    options = tool.resolve_options(services, overrides={"format": "jpeg", "quality": 95, "compress.target_kb": 20})
    result = tool.run_item(WorkItem(src, None), options, RunContext(services, on_progress=lambda f, m: seen.append((f, m))))
    assert result.meta["output_size"] <= 20 * 1024
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and {m for _, m in seen} >= {"Reading", "Encoding"}


def test_preview_runs_the_same_pipeline(tool, services, tmp_path):
    src = noise(tmp_path / "n.png", (200, 100))
    options = tool.resolve_options(services, overrides={"format": "jpeg", "quality": 30, "resize.mode": "fit", "resize.width": 100})
    from app.core.control import JobControl

    preview = tool.make_preview(src, options, services, JobControl())
    assert preview.dimensions == (100, 50) and preview.after_size < preview.before_size
    assert not list(tmp_path.glob("converted"))  # previews never write files

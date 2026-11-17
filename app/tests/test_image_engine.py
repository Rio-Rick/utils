from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.core.control import JobControl
from app.core.errors import Cancelled, ToolError
from app.engines import image_engine as eng
from app.engines.image_engine import ConvertSettings, convert_image, prepare, probe_image, render_preview, target_size


def make_rgba(path: Path, size=(40, 20), *, left=(255, 0, 0, 255), right=(0, 0, 255, 0)) -> Path:
    im = Image.new("RGBA", size, right)
    for x in range(size[0] // 2):
        for y in range(size[1]):
            im.putpixel((x, y), left)
    im.save(path)
    return path


def noise_image(path: Path, size=(256, 256), fmt="PNG") -> Path:
    import os

    im = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    im.save(path, fmt)
    return path


def out(tmp: Path, name: str) -> Path:
    return tmp / "out" / name


# ----------------------------------------------------------------------------- sizing
@pytest.mark.parametrize(
    "mode,w,h,src,allow_up,expected",
    [
        ("original", 100, 100, (400, 200), False, (400, 200)),
        ("fit", 200, 200, (400, 200), False, (200, 100)),
        ("fit", 800, 800, (400, 200), False, (400, 200)),
        ("fit", 800, 800, (400, 200), True, (800, 400)),
        ("fit", 100, 0, (400, 200), False, (100, 50)),
        ("fit", 0, 50, (400, 200), False, (100, 50)),
        ("fill", 100, 100, (400, 200), False, (100, 100)),
        ("fill", 800, 800, (400, 200), False, (200, 200)),
        ("stretch", 100, 100, (400, 200), False, (100, 100)),
        ("stretch", 100, 0, (400, 200), False, (100, 50)),
        ("stretch", 1000, 1000, (400, 200), False, (1000, 1000)),
    ],
)
def test_target_size(mode, w, h, src, allow_up, expected):
    s = ConvertSettings(resize_mode=mode, width=w, height=h, allow_upscale=allow_up)
    assert target_size(*src, s) == expected


def test_stretch_without_keep_aspect_keeps_other_side():
    s = ConvertSettings(resize_mode="stretch", width=100, keep_aspect=False)
    assert target_size(400, 200, s) == (100, 200)


def test_validate_rejects_bad_settings():
    with pytest.raises(ToolError):
        ConvertSettings(resize_mode="fit").validate()
    with pytest.raises(ToolError):
        ConvertSettings(resize_mode="fill", width=10).validate()
    with pytest.raises(ToolError):
        ConvertSettings(format="xyz").validate()
    with pytest.raises(ToolError):
        ConvertSettings(crop_mode="aspect", crop_aspect="abc").validate()
    with pytest.raises(ToolError):
        ConvertSettings(format="ico", ico_sizes=(512,)).validate()


# ----------------------------------------------------------------------------- pipeline
@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp", "avif", "gif", "bmp", "tiff", "ico"])
def test_every_output_format_roundtrips(tmp_path, fmt):
    if not eng.format_available(fmt):
        pytest.skip(f"{fmt} not supported by this Pillow")
    src = make_rgba(tmp_path / "a.png", (48, 32))
    dest = out(tmp_path, f"a.{eng.OUTPUT_FORMATS[fmt].extension}")
    result = convert_image(src, dest, ConvertSettings(format=fmt, quality=80))
    assert dest.exists() and result.output_size == dest.stat().st_size > 0
    with Image.open(dest) as im:
        assert im.format == eng.OUTPUT_FORMATS[fmt].pil
        if fmt != "ico":
            assert im.size == (48, 32)
    assert not list(dest.parent.glob(".*.part*"))


def test_alpha_is_flattened_onto_background_for_jpeg(tmp_path):
    src = make_rgba(tmp_path / "a.png", (20, 10))
    dest = out(tmp_path, "a.jpg")
    convert_image(src, dest, ConvertSettings(format="jpeg", quality=95, background="#00ff00"))
    with Image.open(dest) as im:
        assert im.mode == "RGB"
        r, g, b = im.getpixel((18, 5))  # transparent half
        assert g > 200 and r < 60 and b < 60


def test_transparency_is_kept_for_png_and_webp(tmp_path):
    src = make_rgba(tmp_path / "a.png", (20, 10))
    for fmt, name in (("png", "a.png"), ("webp", "a.webp")):
        dest = out(tmp_path, name)
        convert_image(src, dest, ConvertSettings(format=fmt, webp_lossless=True))
        with Image.open(dest) as im:
            assert im.convert("RGBA").getpixel((18, 5))[3] == 0


def test_resize_fill_crops_to_exact_size(tmp_path):
    src = make_rgba(tmp_path / "a.png", (400, 200))
    dest = out(tmp_path, "a.png")
    result = convert_image(src, dest, ConvertSettings(format="png", resize_mode="fill", width=100, height=100))
    assert result.output_dimensions == (100, 100)
    with Image.open(dest) as im:
        assert im.size == (100, 100)


def test_rotate_90_is_clockwise(tmp_path):
    src = make_rgba(tmp_path / "a.png", (40, 20))  # left half red
    dest = out(tmp_path, "r.png")
    convert_image(src, dest, ConvertSettings(format="png", rotate=90))
    with Image.open(dest) as im:
        assert im.size == (20, 40)
        assert im.getpixel((10, 5))[:3] == (255, 0, 0)    # old left half is now the top
        assert im.getpixel((10, 35))[3] == 0              # old right half is the bottom


def test_arbitrary_rotation_expands_canvas(tmp_path):
    src = make_rgba(tmp_path / "a.png", (40, 20))
    result = convert_image(src, out(tmp_path, "r.png"), ConvertSettings(format="png", rotate=45))
    assert result.output_dimensions[0] > 40 and result.output_dimensions[1] > 20


def test_flip(tmp_path):
    src = make_rgba(tmp_path / "a.png", (40, 20))
    dest = out(tmp_path, "f.png")
    convert_image(src, dest, ConvertSettings(format="png", flip_h=True))
    with Image.open(dest) as im:
        assert im.getpixel((35, 5))[:3] == (255, 0, 0)


def test_crop_rect_and_aspect(tmp_path):
    src = make_rgba(tmp_path / "a.png", (40, 20))
    r = convert_image(src, out(tmp_path, "c1.png"), ConvertSettings(format="png", crop_mode="rect", crop_x=5, crop_y=2, crop_width=10, crop_height=8))
    assert r.output_dimensions == (10, 8)
    r = convert_image(src, out(tmp_path, "c2.png"), ConvertSettings(format="png", crop_mode="aspect", crop_aspect="1:1"))
    assert r.output_dimensions == (20, 20)
    r = convert_image(src, out(tmp_path, "c3.png"), ConvertSettings(format="png", crop_mode="aspect", crop_aspect="1:1", crop_anchor="left"))
    with Image.open(r.output) as im:
        assert im.getpixel((19, 10))[:3] == (255, 0, 0)  # left-anchored square is all red
    with pytest.raises(ToolError):
        convert_image(src, out(tmp_path, "c4.png"), ConvertSettings(format="png", crop_mode="rect", crop_x=100, crop_y=100))


def test_exif_orientation_is_applied_and_metadata_policy(tmp_path):
    im = Image.new("RGB", (40, 20), (10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW
    exif[0x010F] = "ACME"
    exif[0x8825] = {1: "N"}
    src = tmp_path / "o.jpg"
    im.save(src, exif=exif)
    plain = convert_image(src, out(tmp_path, "plain.jpg"), ConvertSettings(format="jpeg"))
    assert plain.output_dimensions == (20, 40)
    with Image.open(plain.output) as o:
        assert not o.getexif().get(0x010F)

    kept = convert_image(src, out(tmp_path, "kept.jpg"), ConvertSettings(format="jpeg", preserve_metadata=True))
    with Image.open(kept.output) as o:
        ex = o.getexif()
        assert ex.get(0x010F) == "ACME"
        assert ex.get(0x0112) in (None, 1)
        assert EXIF_GPS not in ex

    gps = convert_image(src, out(tmp_path, "gps.jpg"), ConvertSettings(format="jpeg", preserve_metadata=True, strip_gps=False))
    with Image.open(gps.output) as o:
        assert EXIF_GPS in o.getexif()

    raw = convert_image(src, out(tmp_path, "raw.jpg"), ConvertSettings(format="jpeg", auto_orient=False))
    assert raw.output_dimensions == (40, 20)


EXIF_GPS = 0x8825


def test_icc_profile_is_kept_by_default_and_dropped_on_request(tmp_path):
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    src = tmp_path / "p.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(src, icc_profile=icc)
    kept = convert_image(src, out(tmp_path, "k.png"), ConvertSettings(format="png"))
    dropped = convert_image(src, out(tmp_path, "d.png"), ConvertSettings(format="png", keep_icc=False))
    with Image.open(kept.output) as a, Image.open(dropped.output) as b:
        assert a.info.get("icc_profile") == icc
        assert "icc_profile" not in b.info


def test_animated_gif_is_preserved_or_reduced(tmp_path):
    frames = [Image.new("RGB", (30, 20), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    src = tmp_path / "anim.gif"
    frames[0].save(src, save_all=True, append_images=frames[1:], duration=[50, 80, 120], loop=0)
    gif = convert_image(src, out(tmp_path, "a.gif"), ConvertSettings(format="gif", resize_mode="fit", width=15))
    assert gif.frames == 3 and gif.output_dimensions == (15, 10)
    webp = convert_image(src, out(tmp_path, "a.webp"), ConvertSettings(format="webp"))
    with Image.open(webp.output) as im:
        assert getattr(im, "n_frames", 1) == 3
    png = convert_image(src, out(tmp_path, "a.png"), ConvertSettings(format="png"))
    with Image.open(png.output) as im:
        assert getattr(im, "n_frames", 1) == 3
    jpg = convert_image(src, out(tmp_path, "a.jpg"), ConvertSettings(format="jpeg"))
    assert jpg.frames == 1 and any("first frame" in n for n in jpg.notes)
    still = convert_image(src, out(tmp_path, "s.webp"), ConvertSettings(format="webp", animated=False))
    assert still.frames == 1


def test_palette_images_are_resampled_properly(tmp_path):
    src = tmp_path / "pal.gif"
    gradient = Image.linear_gradient("L").convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
    gradient.save(src)
    result = convert_image(src, out(tmp_path, "pal.png"), ConvertSettings(format="png", resize_mode="stretch", width=100, height=100))
    with Image.open(result.output) as im:
        assert im.size == (100, 100)
        assert im.mode in {"RGB", "RGBA"}  # converted before resampling


def test_palette_png_stays_small_without_resize(tmp_path):
    src = tmp_path / "pal.png"
    Image.linear_gradient("L").convert("P").save(src)
    result = convert_image(src, out(tmp_path, "pal2.png"), ConvertSettings(format="png"))
    with Image.open(result.output) as im:
        assert im.mode == "P"


def test_png_palette_quantization_option(tmp_path):
    src = noise_image(tmp_path / "n.png", (64, 64))
    plain = convert_image(src, out(tmp_path, "plain.png"), ConvertSettings(format="png"))
    quant = convert_image(src, out(tmp_path, "quant.png"), ConvertSettings(format="png", png_palette=True, png_colors=16))
    assert quant.output_size < plain.output_size
    with Image.open(quant.output) as im:
        assert im.mode == "P"


def test_sixteen_bit_and_cmyk_sources(tmp_path):
    gray16 = tmp_path / "g16.png"
    Image.new("I;16", (8, 8), 40000).save(gray16)
    assert convert_image(gray16, out(tmp_path, "g.jpg"), ConvertSettings(format="jpeg")).output.exists()
    cmyk = tmp_path / "c.jpg"
    Image.new("CMYK", (8, 8), (0, 255, 255, 0)).save(cmyk)
    result = convert_image(cmyk, out(tmp_path, "c.png"), ConvertSettings(format="png"))
    with Image.open(result.output) as im:
        assert im.mode == "RGB"
        assert im.getpixel((1, 1))[0] > 200  # red-ish


def test_ico_sizes_and_square_padding(tmp_path):
    src = make_rgba(tmp_path / "w.png", (64, 32))
    result = convert_image(src, out(tmp_path, "w.ico"), ConvertSettings(format="ico", ico_sizes=(16, 32, 64)))
    with Image.open(result.output) as im:
        assert im.info["sizes"] == {(16, 16), (32, 32), (64, 64)}


def test_target_size_search_respects_limit(tmp_path):
    src = noise_image(tmp_path / "n.png", (300, 300))
    big = convert_image(src, out(tmp_path, "big.jpg"), ConvertSettings(format="jpeg", quality=95))
    limit_kb = max(5, big.output_size // 1024 // 3)
    small = convert_image(src, out(tmp_path, "small.jpg"), ConvertSettings(format="jpeg", quality=95, target_kb=limit_kb))
    assert small.output_size <= limit_kb * 1024
    assert small.output_size < big.output_size


def test_target_size_note_when_unreachable_or_lossless(tmp_path):
    src = noise_image(tmp_path / "n.png", (200, 200))
    r = convert_image(src, out(tmp_path, "u.jpg"), ConvertSettings(format="jpeg", target_kb=1))
    assert any("could not reach" in n for n in r.notes)
    r = convert_image(src, out(tmp_path, "l.png"), ConvertSettings(format="png", target_kb=1))
    assert any("lossy" in n for n in r.notes)


def test_jpeg_options_change_output(tmp_path):
    src = noise_image(tmp_path / "n.png", (128, 128))
    a = convert_image(src, out(tmp_path, "q10.jpg"), ConvertSettings(format="jpeg", quality=10))
    b = convert_image(src, out(tmp_path, "q95.jpg"), ConvertSettings(format="jpeg", quality=95))
    assert a.output_size < b.output_size
    c = convert_image(src, out(tmp_path, "s444.jpg"), ConvertSettings(format="jpeg", quality=90, jpeg_subsampling="4:4:4", jpeg_progressive=True))
    from PIL import JpegImagePlugin

    with Image.open(c.output) as im:
        assert JpegImagePlugin.get_sampling(im) == 0
        assert im.info.get("progressive") or im.info.get("progression")


def test_big_jpeg_uses_draft_and_still_hits_exact_size(tmp_path):
    src = tmp_path / "big.jpg"
    Image.new("RGB", (1600, 1200), (50, 100, 150)).save(src, quality=90)
    result = convert_image(src, out(tmp_path, "small.webp"), ConvertSettings(format="webp", resize_mode="fit", width=200, height=200))
    assert result.output_dimensions == (200, 150)
    assert result.input_dimensions == (1600, 1200)


def test_same_format_keeps_source_format(tmp_path):
    src = make_rgba(tmp_path / "a.png")
    prepared = prepare(src, ConvertSettings(format="same"))
    assert prepared.fmt.key == "png"


def test_max_pixels_limit(tmp_path):
    src = make_rgba(tmp_path / "a.png", (100, 100))
    with pytest.raises(ToolError) as info:
        convert_image(src, out(tmp_path, "a.jpg"), ConvertSettings(format="jpeg", max_pixels=1000))
    assert "too large" in str(info.value)


def test_corrupt_and_missing_files_raise_tool_errors(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    dest = out(tmp_path, "bad.jpg")
    with pytest.raises(ToolError):
        convert_image(bad, dest, ConvertSettings(format="jpeg"))
    assert not dest.exists()
    assert not dest.parent.exists() or not list(dest.parent.iterdir())
    with pytest.raises(ToolError):
        convert_image(tmp_path / "missing.png", dest, ConvertSettings(format="jpeg"))


def test_truncated_image_leaves_nothing_behind(tmp_path):
    good = noise_image(tmp_path / "n.png", (64, 64))
    data = good.read_bytes()
    broken = tmp_path / "t.png"
    broken.write_bytes(data[: len(data) // 2])
    dest = out(tmp_path, "t.jpg")
    with pytest.raises(ToolError):
        convert_image(broken, dest, ConvertSettings(format="jpeg"))
    assert not dest.exists()


def test_cancel_before_start(tmp_path):
    src = make_rgba(tmp_path / "a.png")
    control = JobControl()
    control.cancel()
    dest = out(tmp_path, "a.jpg")
    with pytest.raises(Cancelled):
        convert_image(src, dest, ConvertSettings(format="jpeg"), control=control)
    assert not dest.exists()


def test_cancel_during_encoding_leaves_no_file(tmp_path):
    src = make_rgba(tmp_path / "a.png")
    control = JobControl()
    dest = out(tmp_path, "a.jpg")

    def progress(fraction: float, message: str) -> None:
        if message == "Encoding":
            control.cancel()

    # cancellation is checked after the encoder returns: the temporary file must disappear
    with pytest.raises(Cancelled):
        convert_image(src, dest, ConvertSettings(format="jpeg"), control=control, progress=progress)
    assert not dest.exists()
    assert not list(dest.parent.glob(".*.part*"))


def test_progress_is_reported_in_order(tmp_path):
    src = make_rgba(tmp_path / "a.png")
    seen: list[float] = []
    convert_image(src, out(tmp_path, "a.jpg"), ConvertSettings(format="jpeg"), progress=lambda f, m: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_preview_runs_real_pipeline(tmp_path):
    src = noise_image(tmp_path / "n.png", (200, 100))
    preview = render_preview(src, ConvertSettings(format="jpeg", quality=20, resize_mode="fit", width=100), max_side=150)
    assert preview.after_size > 0 and preview.before_size == src.stat().st_size
    assert preview.dimensions == (100, 50)
    assert max(preview.before.size) <= 150 and max(preview.after.size) <= 150


def test_probe_image(tmp_path):
    src = make_rgba(tmp_path / "a.png", (12, 7))
    info = probe_image(src)
    assert info and (info.width, info.height, info.format, info.has_alpha) == (12, 7, "PNG", True)
    assert probe_image(tmp_path / "nothing.txt") is None


def test_available_formats_and_extensions():
    formats = eng.available_output_formats()
    assert {"png", "jpeg", "gif", "bmp", "tiff", "ico"} <= set(formats)
    ext = eng.input_extensions()
    assert {"png", "jpg", "jpeg", "svg", "gif", "webp"} <= ext
    assert eng.format_for_extension(".JPG") == "jpeg" and eng.format_for_extension("xyz") is None


def test_svg_rendering_with_qt(tmp_path, qapp):
    svg = tmp_path / "s.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"><rect width="40" height="20" fill="#ff0000"/></svg>')
    natural = convert_image(svg, out(tmp_path, "natural.png"), ConvertSettings(format="png"))
    assert natural.output_dimensions == (40, 20)
    big = convert_image(svg, out(tmp_path, "big.png"), ConvertSettings(format="png", resize_mode="fit", width=400))
    assert big.output_dimensions == (400, 200)
    flat = convert_image(svg, out(tmp_path, "flat.jpg"), ConvertSettings(format="jpeg", background="#ffffff"))
    with Image.open(flat.output) as im:
        assert im.getpixel((5, 5))[0] > 200


def test_svg_without_any_renderer_gives_actionable_error(tmp_path, monkeypatch):
    from app.core.errors import DependencyMissing
    from app.engines import svg as svg_mod

    def missing(*_a, **_k):
        raise DependencyMissing("qt")

    monkeypatch.setattr(svg_mod, "_qt", missing)
    svg = tmp_path / "s.svg"
    svg.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='4' height='4'/>")
    with pytest.raises(ToolError) as info:
        convert_image(svg, out(tmp_path, "s.png"), ConvertSettings(format="png"))
    assert "SVG" in str(info.value) and info.value.hint


def test_render_bytes_decodes_back(tmp_path):
    src = noise_image(tmp_path / "n.png", (32, 32))
    prepared = prepare(src, ConvertSettings(format="webp", quality=50))
    data = eng.render_bytes(prepared, ConvertSettings(format="webp", quality=50))
    assert Image.open(io.BytesIO(data)).size == (32, 32)

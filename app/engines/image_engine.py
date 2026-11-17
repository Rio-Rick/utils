"""Image conversion engine (Pillow).

One pipeline serves every caller - the batch converter, the before/after preview, the
"compress to a target size" search and the SVG/PDF helpers - so behaviour never diverges:

    open -> orient (EXIF) -> rotate/flip -> crop -> resize -> mode/alpha handling -> encode

Large JPEGs are decoded at reduced size (``draft``) when they are going to be shrunk
anyway, files are written to a temporary sibling and renamed on success, and animated
GIF/WebP/PNG keep their animation when the target format supports it.
"""

from __future__ import annotations

import io
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageCms, ImageColor, ImageOps, JpegImagePlugin, UnidentifiedImageError, features

from app.core.control import JobControl
from app.core.errors import Cancelled, ToolError
from app.core.fsutil import part_file
from app.engines.process import path_arg, run_process
from app.engines.svg import SVG_EXTENSIONS, Locator, render_svg

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # we enforce our own (configurable) limit after reading the header
EXIF_GPS_IFD = 0x8825
MIN_LOSSY_QUALITY = 5
MAX_ANIMATION_PIXELS = 400_000_000
RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
}
_ANIMATED_SOURCES = frozenset({"GIF", "WEBP", "PNG", "APNG"})
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})
_TIFF_COMPRESSION = {"none": "raw", "lzw": "tiff_lzw", "deflate": "tiff_adobe_deflate", "packbits": "packbits", "jpeg": "jpeg"}
_ICC_SPACES = {"RGB ": {"RGB", "RGBA", "P", "PA"}, "GRAY": {"L", "LA", "1"}, "CMYK": {"CMYK"}}


@dataclass(frozen=True, slots=True)
class FormatInfo:
    """Description of an output format."""

    key: str
    label: str
    pil: str
    extension: str
    extensions: tuple[str, ...]
    mime: str
    alpha: bool
    lossy: bool = False
    animated: bool = False


OUTPUT_FORMATS: dict[str, FormatInfo] = {
    "png": FormatInfo("png", "PNG", "PNG", "png", ("png", "apng"), "image/png", alpha=True, animated=True),
    "jpeg": FormatInfo("jpeg", "JPEG", "JPEG", "jpg", ("jpg", "jpeg", "jpe", "jfif"), "image/jpeg", alpha=False, lossy=True),
    "webp": FormatInfo("webp", "WebP", "WEBP", "webp", ("webp",), "image/webp", alpha=True, lossy=True, animated=True),
    "avif": FormatInfo("avif", "AVIF", "AVIF", "avif", ("avif",), "image/avif", alpha=True, lossy=True),
    "gif": FormatInfo("gif", "GIF", "GIF", "gif", ("gif",), "image/gif", alpha=True, animated=True),
    "bmp": FormatInfo("bmp", "BMP", "BMP", "bmp", ("bmp",), "image/bmp", alpha=False),
    "tiff": FormatInfo("tiff", "TIFF", "TIFF", "tiff", ("tif", "tiff"), "image/tiff", alpha=True),
    "ico": FormatInfo("ico", "ICO", "ICO", "ico", ("ico",), "image/vnd.microsoft.icon", alpha=True),
}
_PIL_TO_KEY = {info.pil: key for key, info in OUTPUT_FORMATS.items()} | {"MPO": "jpeg", "APNG": "png"}
_MODES_ALLOWED: dict[str, frozenset[str]] = {
    "png": frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "I;16", "I"}),
    "webp": frozenset({"RGB", "RGBA"}),
    "avif": frozenset({"RGB", "RGBA"}),
    "gif": frozenset({"P", "L", "RGB", "RGBA"}),
    "tiff": frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "I;16", "I", "F"}),
    "ico": frozenset({"RGBA"}),
}
_plugins_ready = False


def ensure_plugins() -> None:
    """Register optional Pillow plugins (HEIC/AVIF via pillow-heif) once."""
    global _plugins_ready
    if _plugins_ready:
        return
    _plugins_ready = True
    try:
        import pillow_heif  # type: ignore[import-not-found]

        pillow_heif.register_heif_opener()
        if not features.check("avif"):
            pillow_heif.register_avif_opener()
    except Exception:  # optional dependency
        log.debug("pillow-heif not available")


def format_available(key: str) -> bool:
    """Can this build of Pillow (plus plugins) write ``key``?"""
    ensure_plugins()
    if key == "webp":
        return bool(features.check("webp"))
    if key == "avif":
        return bool(features.check("avif")) or Image.registered_extensions().get(".avif") == "AVIF"
    return key in OUTPUT_FORMATS


def available_output_formats() -> list[str]:
    """Output format keys that work on this system, in display order."""
    return [key for key in OUTPUT_FORMATS if format_available(key)]


def input_extensions() -> frozenset[str]:
    """File extensions (no dot) the converter can read on this system."""
    ensure_plugins()
    found = {ext for key, info in OUTPUT_FORMATS.items() if key != "avif" or format_available("avif") for ext in info.extensions}
    found |= {"jp2", "tga", "ppm", "pgm", "pbm"} | SVG_EXTENSIONS
    registered = Image.registered_extensions()
    found |= {ext for ext in ("heic", "heif") if f".{ext}" in registered}
    return frozenset(found)


def format_for_extension(extension: str) -> str | None:
    """Output format key for a file extension, if it is one we can write."""
    ext = extension.lower().lstrip(".")
    for key, info in OUTPUT_FORMATS.items():
        if ext in info.extensions:
            return key
    return None


@dataclass(frozen=True, slots=True)
class ConvertSettings:
    """Everything the pipeline needs; independent of how it is configured."""

    format: str = "webp"
    quality: int = 85
    resize_mode: str = "original"
    width: int = 0
    height: int = 0
    keep_aspect: bool = True
    allow_upscale: bool = False
    resample: str = "lanczos"
    crop_mode: str = "none"
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0
    crop_aspect: str = "1:1"
    crop_anchor: str = "center"
    rotate: int = 0
    flip_h: bool = False
    flip_v: bool = False
    auto_orient: bool = True
    background: str = "#ffffff"
    jpeg_subsampling: str = "4:2:0"
    jpeg_progressive: bool = False
    jpeg_optimize: bool = True
    png_compress_level: int = 6
    png_optimize: bool = False
    png_palette: bool = False
    png_colors: int = 256
    webp_lossless: bool = False
    webp_method: int = 4
    avif_speed: int = 6
    tiff_compression: str = "lzw"
    gif_colors: int = 256
    ico_sizes: tuple[int, ...] = (16, 32, 48, 64, 128, 256)
    preserve_metadata: bool = False
    keep_icc: bool = True
    strip_gps: bool = True
    target_kb: int = 0
    animated: bool = True
    max_pixels: int = 0

    def validate(self) -> None:
        """Raise :class:`ToolError` for combinations the pipeline cannot honour."""
        if self.format != "same" and self.format not in OUTPUT_FORMATS:
            raise ToolError(f"Unsupported output format: {self.format}")
        if self.resize_mode not in {"original", "fit", "fill", "stretch"}:
            raise ToolError(f"Unknown resize mode: {self.resize_mode}")
        if self.resize_mode != "original" and not (self.width or self.height):
            raise ToolError("Resize needs a width or a height", hint="Set a width and/or height, or choose 'original'.")
        if self.resize_mode == "fill" and not (self.width and self.height):
            raise ToolError("'Fill' needs both a width and a height")
        if self.crop_mode not in {"none", "rect", "aspect"}:
            raise ToolError(f"Unknown crop mode: {self.crop_mode}")
        if self.crop_mode == "aspect":
            parse_aspect(self.crop_aspect)
        if min(self.crop_x, self.crop_y, self.crop_width, self.crop_height) < 0:
            raise ToolError("Crop values cannot be negative")
        if self.format == "ico" and (not self.ico_sizes or not all(8 <= n <= 256 for n in self.ico_sizes)):
            raise ToolError("Icon sizes must be between 8 and 256 pixels")


@dataclass(frozen=True, slots=True)
class ConvertResult:
    """What :func:`convert_image` produced."""

    output: Path
    input_size: int
    output_size: int
    input_format: str
    output_format: str
    input_dimensions: tuple[int, int]
    output_dimensions: tuple[int, int]
    frames: int = 1
    notes: tuple[str, ...] = ()

    @property
    def reduction_pct(self) -> float:
        """Size reduction in percent (negative when the file grew)."""
        if self.input_size <= 0:
            return 0.0
        return (1 - self.output_size / self.input_size) * 100.0


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Header information read without decoding the pixels."""

    path: Path
    format: str
    width: int
    height: int
    mode: str
    frames: int
    size_bytes: int
    has_alpha: bool
    dpi: tuple[float, float] | None = None
    icc_profile: bool = False
    exif: bool = False


def parse_aspect(text: str) -> tuple[float, float]:
    """``"16:9"`` / ``"4x3"`` / ``"1.5"`` -> ``(w, h)``."""
    cleaned = text.strip().lower().replace("x", ":").replace("/", ":")
    try:
        if ":" in cleaned:
            left, right = cleaned.split(":", 1)
            w, h = float(left), float(right)
        else:
            w, h = float(cleaned), 1.0
    except ValueError:
        raise ToolError(f"Invalid aspect ratio '{text}'", hint="Use a form like 16:9, 4:3 or 1:1.") from None
    if w <= 0 or h <= 0 or not math.isfinite(w) or not math.isfinite(h):
        raise ToolError(f"Invalid aspect ratio '{text}'")
    return w, h


def parse_color(text: str) -> tuple[int, int, int]:
    """``#rgb`` / ``#rrggbb`` / ``#rrggbbaa`` -> RGB tuple."""
    try:
        return ImageColor.getrgb(text)[:3]
    except ValueError:
        raise ToolError(f"Invalid colour '{text}'", hint="Use #rrggbb.") from None


def probe_image(path: Path) -> ImageInfo | None:
    """Read header information; ``None`` when the file is not an image Pillow can open."""
    ensure_plugins()
    try:
        size_bytes = path.stat().st_size
        if path.suffix[1:].lower() in SVG_EXTENSIONS:
            return ImageInfo(path, "SVG", 0, 0, "RGBA", 1, size_bytes, True)
        with Image.open(path) as im:
            try:
                exif = bool(im.getexif())
            except Exception:
                exif = False
            dpi = im.info.get("dpi")
            return ImageInfo(
                path,
                im.format or "?",
                im.width,
                im.height,
                im.mode,
                int(getattr(im, "n_frames", 1) or 1),
                size_bytes,
                im.mode in _ALPHA_MODES or "transparency" in im.info,
                (float(dpi[0]), float(dpi[1])) if isinstance(dpi, tuple) and len(dpi) == 2 else None,
                bool(im.info.get("icc_profile")),
                exif,
            )
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return None


# --------------------------------------------------------------------------- opening
@dataclass(slots=True)
class _Source:
    image: Image.Image
    format: str
    size: tuple[int, int]
    frames: int
    jpeg_sampling: int | None = None

    def close(self) -> None:
        self.image.close()


def _open_source(
    path: Path,
    s: ConvertSettings,
    locate: Locator | None,
    control: JobControl,
    env: dict[str, str] | None,
) -> _Source:
    ensure_plugins()
    if path.suffix[1:].lower() in SVG_EXTENSIONS:
        width = height = None
        if s.resize_mode in {"fit", "fill", "stretch"}:
            both = s.width and s.height
            width = (max(s.width, s.height) if s.resize_mode != "fit" and both else s.width) or None
            height = None if s.resize_mode != "fit" and both else (s.height or None)
        image = render_svg(path, width=width, height=height, locate=locate, control=control, env=env)
        _check_pixels(image.size, 1, s.max_pixels, path)
        return _Source(image, "SVG", image.size, 1)
    try:
        image = Image.open(path)
    except (UnidentifiedImageError, Image.DecompressionBombError, ValueError) as exc:
        decoded = _decode_with_imagemagick(path, locate, control, env)
        if decoded is None:
            raise ToolError(
                f"{path.name} is not a supported or valid image",
                hint="Check that the file is not corrupted; install ImageMagick to read more formats.",
                detail=str(exc),
            ) from exc
        return _Source(decoded, "?", decoded.size, 1)
    except OSError as exc:
        raise ToolError(f"Cannot read {path.name}: {exc.strerror or exc}") from exc
    size = (image.width, image.height)
    frames = int(getattr(image, "n_frames", 1) or 1)
    animated_source = (image.format or "") in _ANIMATED_SOURCES
    _check_pixels(size, frames if animated_source and s.animated else 1, s.max_pixels, path)
    sampling = None
    if image.format == "JPEG":
        try:
            sampling = JpegImagePlugin.get_sampling(image)
        except Exception:
            sampling = None
        wanted = _draft_request(s, size)
        if wanted:
            image.draft("L" if image.mode == "L" else "RGB", wanted)
    return _Source(image, image.format or "?", size, frames if animated_source else 1, sampling)


def _decode_with_imagemagick(path: Path, locate: Locator | None, control: JobControl, env: dict[str, str] | None) -> Image.Image | None:
    exe = locate("imagemagick") if locate else None
    if not exe:
        return None
    try:
        result = run_process([exe, path_arg(path) + "[0]", "png:-"], control=control, env=env, capture_stdout=True, stdout_limit=1 << 30)
        return Image.open(io.BytesIO(result.stdout)).copy()
    except Cancelled:
        raise
    except Exception as exc:
        log.debug("ImageMagick could not decode %s: %s", path, exc)
        return None


def _check_pixels(size: tuple[int, int], frames: int, max_pixels: int, path: Path) -> None:
    if max_pixels and size[0] * size[1] > max_pixels:
        raise ToolError(
            f"{path.name} is too large ({size[0]}x{size[1]} px)",
            hint="Raise 'advanced.limits.max_image_pixels' in Settings to allow bigger images.",
        )
    if frames > 1 and size[0] * size[1] * frames > MAX_ANIMATION_PIXELS:
        raise ToolError(f"{path.name} is a very large animation", hint="Turn off 'keep animation' to convert only the first frame.")


def _draft_request(s: ConvertSettings, size: tuple[int, int]) -> tuple[int, int] | None:
    """Size to ask the JPEG decoder for (it then decodes at 1/2, 1/4 or 1/8 scale), or ``None``."""
    if s.crop_mode != "none" or s.resize_mode in {"original", "fill"} or s.rotate % 90 != 0:
        return None
    if s.resize_mode == "stretch" and not (s.width and s.height):
        return None
    swapped = s.rotate % 180 == 90
    width, height = (size[1], size[0]) if swapped else size
    out_w, out_h = target_size(width, height, s)
    if out_w >= width or out_h >= height:
        return None
    return (out_h, out_w) if swapped else (out_w, out_h)


# --------------------------------------------------------------------------- geometry
def target_size(width: int, height: int, s: ConvertSettings) -> tuple[int, int]:
    """Output size for a ``width x height`` input under the resize settings."""
    mode, tw, th = s.resize_mode, s.width, s.height
    if mode == "original" or not (tw or th):
        return width, height
    if mode == "stretch":
        if tw and th:
            return tw, th
        if s.keep_aspect:
            return (tw, max(1, round(height * tw / width))) if tw else (max(1, round(width * th / height)), th)
        return tw or width, th or height
    if mode == "fill" and tw and th:
        scale = max(tw / width, th / height)
        if scale > 1 and not s.allow_upscale:
            k = min(width / tw, height / th)
            return max(1, round(tw * k)), max(1, round(th * k))
        return tw, th
    scale = min(tw / width if tw else math.inf, th / height if th else math.inf)
    if not s.allow_upscale:
        scale = min(scale, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _has_alpha(im: Image.Image) -> bool:
    return im.mode in _ALPHA_MODES or "transparency" in im.info


def _rotate(im: Image.Image, angle: int) -> Image.Image:
    angle %= 360
    if angle == 0:
        return im
    if angle == 90:
        return im.transpose(Image.Transpose.ROTATE_270)  # Pillow rotates counter-clockwise
    if angle == 180:
        return im.transpose(Image.Transpose.ROTATE_180)
    if angle == 270:
        return im.transpose(Image.Transpose.ROTATE_90)
    if im.mode not in {"RGBA", "LA"}:
        im = im.convert("RGBA")
    return im.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)


def _crop(im: Image.Image, s: ConvertSettings) -> Image.Image:
    width, height = im.size
    if s.crop_mode == "rect":
        x, y = max(s.crop_x, 0), max(s.crop_y, 0)
        if x >= width or y >= height:
            raise ToolError("The crop area is outside the image", hint=f"The image (after rotation) is {width}x{height} px.")
        cw = min(s.crop_width or width - x, width - x)
        ch = min(s.crop_height or height - y, height - y)
        return im.crop((x, y, x + cw, y + ch))
    if s.crop_mode == "aspect":
        rw, rh = parse_aspect(s.crop_aspect)
        ratio = rw / rh
        if width / height > ratio:
            new_w, new_h = max(1, round(height * ratio)), height
        else:
            new_w, new_h = width, max(1, round(width / ratio))
        x = {"left": 0, "right": width - new_w}.get(s.crop_anchor, (width - new_w) // 2)
        y = {"top": 0, "bottom": height - new_h}.get(s.crop_anchor, (height - new_h) // 2)
        return im.crop((x, y, x + new_w, y + new_h))
    return im


def _resize(im: Image.Image, s: ConvertSettings) -> Image.Image:
    width, height = im.size
    size = target_size(width, height, s)
    box = None
    if s.resize_mode == "fill" and s.width and s.height:
        ratio = size[0] / size[1]
        if width / height > ratio:
            crop_w, crop_h = height * ratio, float(height)
        else:
            crop_w, crop_h = float(width), width / ratio
        box = ((width - crop_w) / 2, (height - crop_h) / 2, (width + crop_w) / 2, (height + crop_h) / 2)
        if size == (width, height) and box == (0.0, 0.0, float(width), float(height)):
            return im
    elif size == (width, height):
        return im
    if im.mode in {"P", "PA", "1"}:  # palette images can only be resized with NEAREST
        im = im.convert("RGBA" if _has_alpha(im) else ("L" if im.mode == "1" else "RGB"))
    resample = RESAMPLE.get(s.resample, Image.Resampling.LANCZOS)
    gap = 3.0 if resample in {Image.Resampling.LANCZOS, Image.Resampling.BICUBIC} else None
    return im.resize(size, resample, box=box, reducing_gap=gap)


def _transform(im: Image.Image, s: ConvertSettings) -> Image.Image:
    im = _rotate(im, s.rotate)
    if s.flip_h:
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if s.flip_v:
        im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return _resize(_crop(im, s), s)


# --------------------------------------------------------------------------- colour / mode
def _cmyk_to_rgb(im: Image.Image) -> Image.Image:
    profile = im.info.get("icc_profile")
    if profile:
        try:
            source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
            return ImageCms.profileToProfile(im, source, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            log.debug("ICC conversion failed; using the plain CMYK -> RGB conversion")
    return im.convert("RGB")


def _to_8bit(im: Image.Image) -> Image.Image:
    try:
        if im.mode.startswith("I;16") or im.mode == "I":
            return im.point(lambda v: v * (1 / 256)).convert("L")
        if im.mode == "F":
            return im.point(lambda v: v * 255).convert("L")
    except Exception:
        log.debug("high bit depth reduction failed", exc_info=True)
    return im.convert("L")


def _flatten(im: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """Composite transparency onto ``background`` and return an RGB (or grey) image."""
    gray = im.mode in {"L", "LA", "1"} and background[0] == background[1] == background[2]
    if _has_alpha(im):
        rgba = im.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, background)
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas.convert("L") if gray else canvas
    if im.mode in {"L", "1"}:
        return im
    return im.convert("RGB")


def _prepare_mode(im: Image.Image, fmt: FormatInfo, s: ConvertSettings, background: tuple[int, int, int]) -> Image.Image:
    """Bring ``im`` into a mode ``fmt`` can store (flattening alpha for formats without it)."""
    if im.mode == "CMYK" and fmt.key != "tiff":
        im = _cmyk_to_rgb(im)
    elif im.mode in {"YCbCr", "LAB", "HSV"}:
        im = im.convert("RGB")
    high_depth = im.mode in {"I", "F"} or im.mode.startswith("I;16")
    if high_depth and (fmt.key not in {"png", "tiff"} or (im.mode == "F" and fmt.key == "png")):
        im = _to_8bit(im)
    if not fmt.alpha:
        im = _flatten(im, background)
        return im.convert("L") if im.mode == "1" and fmt.key == "jpeg" else im
    palette_ok = fmt.key in {"png", "tiff", "gif"} and im.mode == "P" and not _has_alpha(im)
    if not palette_ok and (im.mode not in _MODES_ALLOWED[fmt.key] or (im.mode == "P" and fmt.key != "gif")):
        im = im.convert("RGBA" if _has_alpha(im) or fmt.key == "ico" else "RGB")
    if fmt.key == "png" and s.png_palette and im.mode in {"RGB", "RGBA", "L"}:
        method = Image.Quantize.FASTOCTREE if im.mode == "RGBA" else Image.Quantize.MEDIANCUT
        source = im.convert("RGB") if im.mode == "L" else im
        im = source.quantize(colors=max(2, min(256, s.png_colors)), method=method)
    return im


# --------------------------------------------------------------------------- metadata
def _collect_metadata(im: Image.Image, s: ConvertSettings) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if (s.preserve_metadata or s.keep_icc) and (icc := im.info.get("icc_profile")):
        meta["icc_profile"] = icc
    if not s.preserve_metadata:
        return meta
    try:
        exif = im.getexif()
        if s.strip_gps and EXIF_GPS_IFD in exif:
            del exif[EXIF_GPS_IFD]
        if len(exif):
            meta["exif"] = exif.tobytes()
    except Exception:
        log.debug("EXIF could not be copied", exc_info=True)
    if xmp := im.info.get("xmp"):
        meta["xmp"] = xmp
    if dpi := im.info.get("dpi"):
        meta["dpi"] = dpi
    return meta


def _icc_fits(profile: bytes, mode: str) -> bool:
    modes = _ICC_SPACES.get(profile[16:20].decode("latin-1", "replace"))
    return modes is None or mode in modes


# --------------------------------------------------------------------------- pipeline
@dataclass(slots=True)
class Prepared:
    """A fully transformed image (or animation) ready to be encoded."""

    fmt: FormatInfo
    image: Image.Image
    extra_frames: list[Image.Image] = field(default_factory=list)
    durations: list[int] = field(default_factory=list)
    loop: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    source_format: str = ""
    source_size: tuple[int, int] = (0, 0)
    jpeg_sampling: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return 1 + len(self.extra_frames)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


def resolve_format_key(requested: str, source_format: str) -> str:
    """Output format key for the ``format`` setting (``"same"`` follows the source; unwritable sources become PNG)."""
    return _PIL_TO_KEY.get(source_format.upper(), "png") if requested == "same" else requested


def _resolve_format(s: ConvertSettings, source_format: str) -> FormatInfo:
    key = resolve_format_key(s.format, source_format)
    if not format_available(key):
        raise ToolError(
            f"{OUTPUT_FORMATS[key].label} output is not supported by this installation of Pillow",
            hint="Update Pillow (11.3+ includes AVIF) or install pillow-heif.",
        )
    return OUTPUT_FORMATS[key]


def prepare(
    source: Path,
    s: ConvertSettings,
    *,
    control: JobControl | None = None,
    locate: Locator | None = None,
    env: dict[str, str] | None = None,
) -> Prepared:
    """Open ``source`` and apply every transformation of ``s`` (no encoding yet)."""
    s.validate()
    ctl = control or JobControl()
    ctl.check()
    src = _open_source(Path(source), s, locate, ctl, env)
    try:
        return _prepare_opened(src, s, ctl)
    except (OSError, ValueError, SyntaxError, EOFError) as exc:  # decoding is lazy: truncated files fail here
        raise ToolError(
            f"{Path(source).name} could not be decoded: {exc}",
            hint="The file looks truncated or corrupted.",
        ) from exc
    finally:
        src.close()


def _prepare_opened(src: _Source, s: ConvertSettings, ctl: JobControl) -> Prepared:
    fmt = _resolve_format(s, src.format)
    background = parse_color(s.background)
    animated = s.animated and fmt.animated and src.frames > 1
    notes: list[str] = []
    if src.frames > 1 and not animated:
        notes.append("animation reduced to the first frame")
    if animated:
        return _prepare_animation(src, s, fmt, background, ctl, notes)
    image = ImageOps.exif_transpose(src.image) if s.auto_orient else src.image.copy()
    image.load()
    meta = _collect_metadata(image, s)
    original_mode = image.mode
    ctl.check()
    image = _transform(image, s)
    ctl.check()
    image = _prepare_mode(image, fmt, s, background)
    if "icc_profile" in meta and (original_mode == "CMYK" and fmt.key != "tiff" or not _icc_fits(meta["icc_profile"], image.mode)):
        del meta["icc_profile"]
    image.info = {}  # nothing is written that was not asked for explicitly
    return Prepared(fmt, image, meta=meta, source_format=src.format, source_size=src.size, jpeg_sampling=src.jpeg_sampling, notes=notes)


def _prepare_animation(src: _Source, s: ConvertSettings, fmt: FormatInfo, background: tuple[int, int, int], ctl: JobControl, notes: list[str]) -> Prepared:
    frames: list[Image.Image] = []
    durations: list[int] = []
    meta = _collect_metadata(src.image, s)
    loop = int(src.image.info.get("loop", 0) or 0)
    for index in range(src.frames):
        ctl.check()
        src.image.seek(index)
        durations.append(int(src.image.info.get("duration", 100) or 100))
        frame = _prepare_mode(_transform(src.image.convert("RGBA"), s), fmt, s, background)
        frame.info = {}
        frames.append(frame)
    return Prepared(fmt, frames[0], frames[1:], durations, loop, meta, src.format, src.size, notes=notes)


# --------------------------------------------------------------------------- encoding
def _quality_for(s: ConvertSettings, fmt: FormatInfo) -> int:
    return max(1, min(100, s.quality)) if fmt.lossy else 100


def _save_params(prepared: Prepared, s: ConvertSettings, quality: int, *, with_metadata: bool = True) -> dict[str, Any]:
    fmt = prepared.fmt
    params: dict[str, Any] = {}
    if fmt.key == "jpeg":
        params.update(quality=quality, optimize=s.jpeg_optimize, progressive=s.jpeg_progressive)
        if prepared.image.mode == "RGB":
            keep = prepared.jpeg_sampling if prepared.jpeg_sampling is not None else "4:2:0"
            params["subsampling"] = keep if s.jpeg_subsampling == "keep" else s.jpeg_subsampling
    elif fmt.key == "png":
        params.update(compress_level=max(0, min(9, s.png_compress_level)), optimize=s.png_optimize)
    elif fmt.key == "webp":
        params.update(quality=100 if s.webp_lossless else quality, lossless=s.webp_lossless, method=max(0, min(6, s.webp_method)))
    elif fmt.key == "avif":
        params.update(quality=quality, speed=max(0, min(10, s.avif_speed)))
    elif fmt.key == "gif":
        params["optimize"] = True
    elif fmt.key == "tiff":
        compression = _TIFF_COMPRESSION.get(s.tiff_compression, "tiff_lzw")
        if compression == "jpeg" and prepared.image.mode not in {"RGB", "L"}:
            compression = "tiff_adobe_deflate"
        params["compression"] = compression
    elif fmt.key == "ico":
        params["sizes"] = [(n, n) for n in s.ico_sizes]
    if with_metadata:
        params.update(prepared.meta)
    if prepared.frames > 1:
        params.update(save_all=True, append_images=prepared.extra_frames, duration=prepared.durations, loop=prepared.loop)
        if fmt.key == "gif":
            params["disposal"] = 2
    return params


def _ico_ready(prepared: Prepared, s: ConvertSettings) -> Image.Image:
    """ICO wants a square canvas at least as large as the biggest icon size."""
    image = prepared.image
    side = max(s.ico_sizes)
    longest = max(image.size)
    if image.size[0] != image.size[1]:
        canvas = Image.new("RGBA", (longest, longest), (0, 0, 0, 0))
        canvas.paste(image, ((longest - image.size[0]) // 2, (longest - image.size[1]) // 2))
        image = canvas
    if image.size[0] < side:
        image = image.resize((side, side), Image.Resampling.LANCZOS)
    return image


def encode(prepared: Prepared, s: ConvertSettings, target: Any, *, quality: int | None = None) -> None:
    """Write ``prepared`` to ``target`` (a path or a binary file object)."""
    fmt = prepared.fmt
    image = _ico_ready(prepared, s) if fmt.key == "ico" else prepared.image
    if fmt.key == "gif" and s.gif_colors < 256 and image.mode != "P":
        image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=max(2, s.gif_colors))
    q = quality if quality is not None else _quality_for(s, fmt)
    try:
        image.save(target, format=fmt.pil, **_save_params(prepared, s, q))
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        if not prepared.meta:
            raise ToolError(f"Cannot encode as {fmt.label}: {exc}") from exc
        if hasattr(target, "seek"):
            target.seek(0)
            target.truncate()
        try:
            image.save(target, format=fmt.pil, **_save_params(prepared, s, q, with_metadata=False))
        except (OSError, ValueError, TypeError, KeyError, SyntaxError) as retry_exc:
            raise ToolError(f"Cannot encode as {fmt.label}: {retry_exc}") from retry_exc
        prepared.notes.append("metadata could not be written")


def _search_quality(prepared: Prepared, s: ConvertSettings, limit_bytes: int) -> tuple[bytes, int]:
    """Highest quality whose encoded size fits ``limit_bytes`` (binary search)."""
    low, high = MIN_LOSSY_QUALITY, max(MIN_LOSSY_QUALITY, min(100, s.quality))
    best: tuple[bytes, int] | None = None
    while low <= high:
        mid = (low + high) // 2
        buffer = io.BytesIO()
        encode(prepared, s, buffer, quality=mid)
        data = buffer.getvalue()
        if len(data) <= limit_bytes:
            best, low = (data, mid), mid + 1
        else:
            high = mid - 1
    if best is None:
        buffer = io.BytesIO()
        encode(prepared, s, buffer, quality=MIN_LOSSY_QUALITY)
        return buffer.getvalue(), MIN_LOSSY_QUALITY
    return best


def _is_lossy(prepared: Prepared, s: ConvertSettings) -> bool:
    return prepared.fmt.lossy and not (prepared.fmt.key == "webp" and s.webp_lossless)


def render_bytes(prepared: Prepared, s: ConvertSettings) -> bytes:
    """Encode to memory, honouring the target size."""
    if s.target_kb > 0 and prepared.frames == 1:
        if _is_lossy(prepared, s):
            data, _quality = _search_quality(prepared, s, s.target_kb * 1024)
            if len(data) > s.target_kb * 1024:
                prepared.notes.append(f"could not reach {s.target_kb} KB (smallest result: {len(data) // 1024} KB)")
            return data
        prepared.notes.append("target size only applies to lossy formats")
    buffer = io.BytesIO()
    encode(prepared, s, buffer)
    return buffer.getvalue()


def write_prepared(
    prepared: Prepared,
    destination: Path,
    settings: ConvertSettings,
    *,
    input_size: int = 0,
    control: JobControl | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> ConvertResult:
    """Encode an already prepared image to ``destination`` atomically.

    A failed or cancelled write leaves nothing behind (the data goes to a temporary
    sibling that is renamed on success).
    """
    ctl = control or JobControl()
    report = progress or (lambda _fraction, _message: None)
    ctl.check()
    report(0.55, "Encoding")
    with part_file(destination) as tmp:
        if settings.target_kb > 0:
            tmp.write_bytes(render_bytes(prepared, settings))
        else:
            encode(prepared, settings, tmp)
        ctl.check()
    report(1.0, "Done")
    return ConvertResult(
        destination,
        input_size,
        destination.stat().st_size,
        prepared.source_format,
        prepared.fmt.label,
        prepared.source_size,
        prepared.size,
        prepared.frames,
        tuple(prepared.notes),
    )


def convert_image(
    source: Path,
    destination: Path,
    settings: ConvertSettings,
    *,
    control: JobControl | None = None,
    progress: Callable[[float, str], None] | None = None,
    locate: Locator | None = None,
    env: dict[str, str] | None = None,
) -> ConvertResult:
    """Convert ``source`` to ``destination`` (the extension must match the chosen format)."""
    ctl = control or JobControl()
    report = progress or (lambda _fraction, _message: None)
    source = Path(source)
    try:
        input_size = source.stat().st_size
    except OSError as exc:
        raise ToolError(f"Cannot read {source.name}: {exc.strerror or exc}") from exc
    report(0.05, "Reading")
    prepared = prepare(source, settings, control=ctl, locate=locate, env=env)
    return write_prepared(prepared, destination, settings, input_size=input_size, control=ctl, progress=report)


@dataclass(frozen=True, slots=True)
class Preview:
    """Result of :func:`render_preview`."""

    before: Image.Image
    after: Image.Image
    before_size: int
    after_size: int
    fmt: str
    dimensions: tuple[int, int]
    notes: tuple[str, ...]

    @property
    def reduction_pct(self) -> float:
        return 0.0 if self.before_size <= 0 else (1 - self.after_size / self.before_size) * 100.0


def load_for_display(
    source: Path,
    max_side: int = 1400,
    *,
    locate: Locator | None = None,
    control: JobControl | None = None,
    env: dict[str, str] | None = None,
) -> Image.Image:
    """A display-sized, EXIF-oriented RGB(A) copy of ``source`` (cheap for huge JPEGs thanks to ``draft``)."""
    ensure_plugins()
    ctl = control or JobControl()
    if source.suffix[1:].lower() in SVG_EXTENSIONS:
        return render_svg(source, width=max_side, locate=locate, control=ctl, env=env)
    try:
        with Image.open(source) as image:
            if image.format == "JPEG":
                image.draft("RGB", (max_side, max_side))
            oriented = ImageOps.exif_transpose(image)
            oriented.load()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ToolError(f"Cannot preview {source.name}: {exc}") from exc
    if max(oriented.size) > max_side:
        oriented.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if oriented.mode in {"P", "PA"}:
        oriented = oriented.convert("RGBA")
    return oriented


def render_preview(
    source: Path,
    settings: ConvertSettings,
    *,
    max_side: int = 1400,
    control: JobControl | None = None,
    locate: Locator | None = None,
    env: dict[str, str] | None = None,
) -> Preview:
    """Run the real pipeline in memory and decode the result again (shows true compression artefacts)."""
    ctl = control or JobControl()
    settings = replace(settings, animated=False)
    prepared = prepare(source, settings, control=ctl, locate=locate, env=env)
    data = render_bytes(prepared, settings)
    ctl.check()
    after = Image.open(io.BytesIO(data))
    after.load()
    if max(after.size) > max_side:
        after.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    before = load_for_display(source, max_side, locate=locate, control=ctl, env=env)
    return Preview(before, after, source.stat().st_size, len(data), prepared.fmt.label, prepared.size, tuple(prepared.notes))

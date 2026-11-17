"""Image Converter: format conversion, resize / crop / rotate / flip and compression."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config_schema import Kind, Setting
from app.core.control import JobControl
from app.core.errors import ToolError
from app.core.formatting import human_size, reduction_text
from app.core.naming import OutputPlanner
from app.engines import image_engine as engine
from app.engines.image_engine import ConvertSettings, Prepared
from app.tools.base import BatchTool, Category, ItemResult, RunContext, WorkItem, output_settings, register_tool

if TYPE_CHECKING:
    from app.core.services import Services

_FORMAT_LABELS = {"same": "Same as source", **{key: info.label for key, info in engine.OUTPUT_FORMATS.items()}}
_FORMATS = tuple(_FORMAT_LABELS)
_ANY_LOSSY = ("jpeg", "webp", "avif", "same")
_RESIZING = ("fit", "fill", "stretch")
_ICO_SIZES = ("16", "24", "32", "48", "64", "72", "96", "128", "256")


def _aspect_problem(value: str) -> str | None:
    try:
        engine.parse_aspect(value)
    except ToolError as exc:
        return exc.message
    return None


OPTIONS: tuple[Setting, ...] = (
    Setting("format", Kind.CHOICE, "webp", "Output format",
            "Format of the converted files. 'Same as source' keeps each file's own format "
            "(formats that cannot be written, such as SVG, become PNG).",
            choices=_FORMATS, choice_labels=_FORMAT_LABELS, group="Format"),
    Setting("quality", Kind.INT, 85, "Quality",
            "Quality of JPEG, lossy WebP and AVIF files. Higher means larger, better looking files. "
            "Format specific values in the advanced section override it.",
            minimum=1, maximum=100, unit="%", group="Format", visible_if={"format": _ANY_LOSSY}),
    Setting("background", Kind.COLOR, "#ffffff", "Background colour",
            "Fills transparent areas when the output format has no transparency (JPEG, BMP).",
            group="Format", visible_if={"format": ("jpeg", "bmp", "same")}),
    Setting("compress.target_kb", Kind.INT, 0, "Target file size",
            "Search the highest quality that still fits this size (JPEG, lossy WebP, AVIF). "
            "0 = off. Quality never goes below 5.",
            minimum=0, maximum=10_000_000, unit="KB", group="Format", visible_if={"format": _ANY_LOSSY}),
    Setting("webp.lossless", Kind.BOOL, False, "Lossless WebP",
            "Store WebP without any quality loss (larger files, exact pixels).",
            group="Format", visible_if={"format": ("webp", "same")}),
    Setting("png.compress_level", Kind.INT, 6, "PNG compression level",
            "0 = fastest / largest, 9 = slowest / smallest. PNG is lossless either way.",
            minimum=0, maximum=9, group="Format", visible_if={"format": ("png", "same")}),

    Setting("resize.mode", Kind.CHOICE, "original", "Resize",
            "original: keep the size. fit: scale to fit inside the box. fill: scale and crop to "
            "exactly fill the box. stretch: force the exact size (ignores the aspect ratio).",
            choices=("original", "fit", "fill", "stretch"),
            choice_labels={"original": "Keep original size", "fit": "Fit inside", "fill": "Fill and crop",
                           "stretch": "Stretch"}, group="Resize"),
    Setting("resize.width", Kind.INT, 0, "Width", "0 = automatic (from the height and the aspect ratio).",
            minimum=0, maximum=32768, unit="px", group="Resize", visible_if={"resize.mode": _RESIZING}),
    Setting("resize.height", Kind.INT, 0, "Height", "0 = automatic (from the width and the aspect ratio).",
            minimum=0, maximum=32768, unit="px", group="Resize", visible_if={"resize.mode": _RESIZING}),
    Setting("resize.keep_aspect", Kind.BOOL, True, "Preserve aspect ratio",
            "Stretch with only one dimension set: derive the other one from the aspect ratio.",
            group="Resize", visible_if={"resize.mode": ("stretch",)}),
    Setting("resize.allow_upscale", Kind.BOOL, False, "Allow enlarging",
            "Let 'fit' and 'fill' make small images bigger. Off = never enlarge.",
            group="Resize", visible_if={"resize.mode": ("fit", "fill")}),
    Setting("resize.resample", Kind.CHOICE, "lanczos", "Resampling filter",
            "lanczos is sharpest, bilinear smoothest, nearest keeps hard pixel edges.",
            choices=("lanczos", "bicubic", "bilinear", "box", "nearest"), advanced=True, group="Resize",
            visible_if={"resize.mode": _RESIZING}),

    Setting("crop.mode", Kind.CHOICE, "none", "Crop",
            "rect: cut out a rectangle. aspect: cut the largest area with a given aspect ratio.",
            choices=("none", "rect", "aspect"),
            choice_labels={"none": "No crop", "rect": "Rectangle", "aspect": "Aspect ratio"}, group="Crop"),
    Setting("crop.x", Kind.INT, 0, "Left", "Distance of the crop area from the left edge.",
            minimum=0, maximum=1_000_000, unit="px", group="Crop", visible_if={"crop.mode": ("rect",)}),
    Setting("crop.y", Kind.INT, 0, "Top", "Distance of the crop area from the top edge.",
            minimum=0, maximum=1_000_000, unit="px", group="Crop", visible_if={"crop.mode": ("rect",)}),
    Setting("crop.width", Kind.INT, 0, "Crop width", "0 = up to the right edge.",
            minimum=0, maximum=1_000_000, unit="px", group="Crop", visible_if={"crop.mode": ("rect",)}),
    Setting("crop.height", Kind.INT, 0, "Crop height", "0 = down to the bottom edge.",
            minimum=0, maximum=1_000_000, unit="px", group="Crop", visible_if={"crop.mode": ("rect",)}),
    Setting("crop.aspect", Kind.STR, "1:1", "Aspect ratio", "For example 1:1, 4:3, 16:9 or 3:2.",
            group="Crop", visible_if={"crop.mode": ("aspect",)}, validator=_aspect_problem),
    Setting("crop.anchor", Kind.CHOICE, "center", "Keep area at",
            "Which part of the image the aspect ratio crop keeps.",
            choices=("center", "top", "bottom", "left", "right"), group="Crop",
            visible_if={"crop.mode": ("aspect",)}),

    Setting("rotate", Kind.INT, 0, "Rotate", "Clockwise. Angles other than 90/180/270 add transparent corners.",
            minimum=0, maximum=359, unit="\u00b0", group="Transform"),
    Setting("flip_h", Kind.BOOL, False, "Flip horizontally", "Mirror left and right.", group="Transform"),
    Setting("flip_v", Kind.BOOL, False, "Flip vertically", "Mirror top and bottom.", group="Transform"),
    Setting("auto_orient", Kind.BOOL, True, "Apply EXIF orientation",
            "Rotate photos upright according to their EXIF orientation tag before anything else.",
            advanced=True, group="Transform"),
    Setting("animated", Kind.BOOL, True, "Keep animation",
            "Convert every frame of animated GIF / WebP / PNG when the output format supports "
            "animation. Off = first frame only.", advanced=True, group="Transform"),

    Setting("metadata.preserve", Kind.BOOL, False, "Keep metadata",
            "Copy EXIF, XMP and the resolution into the result. Off = strip them.",
            advanced=True, group="Metadata"),
    Setting("metadata.strip_gps", Kind.BOOL, True, "Remove GPS location",
            "Drop the GPS position even when metadata is kept (privacy).",
            advanced=True, group="Metadata", visible_if={"metadata.preserve": (True,)}),
    Setting("metadata.keep_icc", Kind.BOOL, True, "Keep colour profile",
            "Keep the embedded ICC profile so colours look the same everywhere.",
            advanced=True, group="Metadata"),

    Setting("jpeg.quality", Kind.INT, 0, "JPEG quality", "Overrides the general quality for JPEG. 0 = use the general quality.",
            minimum=0, maximum=100, unit="%", advanced=True, group="JPEG", visible_if={"format": ("jpeg", "same")}),
    Setting("jpeg.subsampling", Kind.CHOICE, "4:2:0", "Chroma subsampling",
            "4:4:4 keeps colour detail (larger), 4:2:0 is the usual compromise, 'keep' reuses the "
            "source file's setting.", choices=("4:2:0", "4:2:2", "4:4:4", "keep"), advanced=True,
            group="JPEG", visible_if={"format": ("jpeg", "same")}),
    Setting("jpeg.progressive", Kind.BOOL, False, "Progressive JPEG",
            "Load in several passes in browsers; slightly smaller files.", advanced=True, group="JPEG",
            visible_if={"format": ("jpeg", "same")}),
    Setting("jpeg.optimize", Kind.BOOL, True, "Optimise Huffman tables",
            "Smaller files at no quality cost; a little slower.", advanced=True, group="JPEG",
            visible_if={"format": ("jpeg", "same")}),

    Setting("png.optimize", Kind.BOOL, False, "Extra PNG optimisation",
            "Try harder to shrink the file (slower).", advanced=True, group="PNG",
            visible_if={"format": ("png", "same")}),
    Setting("png.palette", Kind.BOOL, False, "Reduce to a palette",
            "Quantise to a limited number of colours: much smaller files, slight colour loss.",
            advanced=True, group="PNG", visible_if={"format": ("png", "same")}),
    Setting("png.colors", Kind.INT, 256, "Palette colours", "", minimum=2, maximum=256, advanced=True,
            group="PNG", visible_if={"png.palette": (True,)}),

    Setting("webp.quality", Kind.INT, 0, "WebP quality", "Overrides the general quality for lossy WebP. 0 = use the general quality.",
            minimum=0, maximum=100, unit="%", advanced=True, group="WebP", visible_if={"format": ("webp", "same")}),
    Setting("webp.method", Kind.INT, 4, "WebP effort",
            "0 = fastest, 6 = smallest file (slowest).", minimum=0, maximum=6, advanced=True, group="WebP",
            visible_if={"format": ("webp", "same")}),

    Setting("avif.quality", Kind.INT, 0, "AVIF quality", "Overrides the general quality for AVIF. 0 = use the general quality.",
            minimum=0, maximum=100, unit="%", advanced=True, group="AVIF", visible_if={"format": ("avif", "same")}),
    Setting("avif.speed", Kind.INT, 6, "AVIF speed", "0 = slowest / best compression, 10 = fastest.",
            minimum=0, maximum=10, advanced=True, group="AVIF", visible_if={"format": ("avif", "same")}),

    Setting("tiff.compression", Kind.CHOICE, "lzw", "TIFF compression",
            "Lossless methods (lzw, deflate, packbits) or 'jpeg' (lossy, RGB / grey only).",
            choices=("none", "lzw", "deflate", "packbits", "jpeg"), advanced=True, group="TIFF",
            visible_if={"format": ("tiff", "same")}),
    Setting("gif.colors", Kind.INT, 256, "GIF colours", "Number of palette colours (2-256).", minimum=2, maximum=256,
            advanced=True, group="GIF", visible_if={"format": ("gif", "same")}),
    Setting("ico.sizes", Kind.LIST, ["16", "32", "48", "64", "128", "256"], "Icon sizes",
            "Sizes (pixels, square) stored in the .ico file.", choices=_ICO_SIZES, advanced=True,
            group="ICO", visible_if={"format": ("ico", "same")}),
    *output_settings(),
)


def settings_from_options(options: Mapping[str, Any], format_key: str | None = None, *, max_pixels: int = 0) -> ConvertSettings:
    """Resolved tool options -> engine settings (``format_key`` replaces ``"same"`` once the source is known)."""
    fmt = format_key or str(options["format"])
    quality = int(options["quality"])
    specific = int(options.get(f"{fmt}.quality", 0) or 0)
    return ConvertSettings(
        format=fmt,
        quality=specific or quality,
        resize_mode=options["resize.mode"],
        width=int(options["resize.width"]),
        height=int(options["resize.height"]),
        keep_aspect=bool(options["resize.keep_aspect"]),
        allow_upscale=bool(options["resize.allow_upscale"]),
        resample=options["resize.resample"],
        crop_mode=options["crop.mode"],
        crop_x=int(options["crop.x"]),
        crop_y=int(options["crop.y"]),
        crop_width=int(options["crop.width"]),
        crop_height=int(options["crop.height"]),
        crop_aspect=options["crop.aspect"],
        crop_anchor=options["crop.anchor"],
        rotate=int(options["rotate"]),
        flip_h=bool(options["flip_h"]),
        flip_v=bool(options["flip_v"]),
        auto_orient=bool(options["auto_orient"]),
        background=options["background"],
        jpeg_subsampling=options["jpeg.subsampling"],
        jpeg_progressive=bool(options["jpeg.progressive"]),
        jpeg_optimize=bool(options["jpeg.optimize"]),
        png_compress_level=int(options["png.compress_level"]),
        png_optimize=bool(options["png.optimize"]),
        png_palette=bool(options["png.palette"]),
        png_colors=int(options["png.colors"]),
        webp_lossless=bool(options["webp.lossless"]),
        webp_method=int(options["webp.method"]),
        avif_speed=int(options["avif.speed"]),
        tiff_compression=options["tiff.compression"],
        gif_colors=int(options["gif.colors"]),
        ico_sizes=tuple(sorted({int(n) for n in options["ico.sizes"]})),
        preserve_metadata=bool(options["metadata.preserve"]),
        keep_icc=bool(options["metadata.keep_icc"]),
        strip_gps=bool(options["metadata.strip_gps"]),
        target_kb=int(options["compress.target_kb"]),
        animated=bool(options["animated"]),
        max_pixels=max_pixels,
    )


def engine_helpers(services: Services) -> tuple[engine.Locator, dict[str, str] | None]:
    """The executable locator and extra environment the engine passes to external converters."""
    return services.deps.path, (services.deps.environment() or None)


@register_tool
class ImageConverter(BatchTool):
    """Convert images between formats, with resizing, cropping, rotation and compression."""

    id = "image.converter"
    name = "Image Converter"
    description = "Convert between PNG, JPG, WebP, AVIF, GIF, BMP, TIFF and ICO. Resize, crop, rotate and compress in one go."
    category = Category.IMAGE
    icon = "image"
    keywords = ("convert", "png", "jpg", "jpeg", "webp", "avif", "gif", "bmp", "tiff", "ico", "svg", "resize", "crop", "compress", "heic")
    config_domain = "image"
    config_section = "converter"
    config_title = "Image tools"
    requires = ("pillow",)
    options = OPTIONS
    preview = "compare"
    order = 10

    def supported_extensions(self) -> frozenset[str]:
        return engine.input_extensions()

    @classmethod
    def available_choices(cls, key: str) -> tuple[str, ...] | None:
        if key == "format":
            return ("same", *engine.available_output_formats())
        return None

    def check_options(self, options: Mapping[str, Any]) -> None:
        settings_from_options(options).validate()

    # ------------------------------------------------------------------ helpers
    def _format_key(self, options: Mapping[str, Any], source: Path) -> str:
        requested = str(options["format"])
        if requested != "same":
            return requested
        info = engine.probe_image(source)
        return engine.resolve_format_key("same", info.format if info else "")

    def _settings(self, options: Mapping[str, Any], services: Services, source: Path) -> ConvertSettings:
        key = self._format_key(options, source)
        max_pixels = int(services.config.get("advanced", "limits.max_image_pixels", 0) or 0)
        return replace(settings_from_options(options, key, max_pixels=max_pixels), format=key)

    # ------------------------------------------------------------------ processing
    def run_item(self, item: WorkItem, options: Mapping[str, Any], ctx: RunContext) -> ItemResult:
        source = item.source
        if source is None:
            raise ToolError("No input file")
        services = ctx.services
        locate, env = engine_helpers(services)
        settings = self._settings(options, services, source)
        try:
            input_size = source.stat().st_size
        except OSError as exc:
            raise ToolError(f"Cannot read {source.name}: {exc.strerror or exc}") from exc

        cache: dict[str, Prepared] = {}

        def prepared() -> Prepared:
            if "image" not in cache:
                ctx.progress(0.05, "Reading")
                cache["image"] = engine.prepare(source, settings, control=ctx.control, locate=locate, env=env)
            return cache["image"]

        planner = OutputPlanner.from_options(options, fallback_directory=str(services.config.get("app", "directories.output", "") or ""))
        fields = {
            "format": settings.format,
            "quality": settings.quality,
            "width": lambda: prepared().size[0],
            "height": lambda: prepared().size[1],
        }
        extension = engine.OUTPUT_FORMATS[settings.format].extension
        with planner.plan(source, extension, fields, base_dir=item.base_dir, index=int(item.extra.get("index", 1))) as target:
            if target.skip:
                return ItemResult(outputs=[target.path], skipped=True, message="Skipped: the output file already exists")
            result = engine.write_prepared(prepared(), target.path, settings, input_size=input_size, control=ctx.control, progress=ctx.progress)

        message = f"{human_size(result.input_size)} \u2192 {human_size(result.output_size)} ({reduction_text(result.input_size, result.output_size)})"
        if result.notes:
            message += " \u00b7 " + "; ".join(result.notes)
        ctx.log(f"{source.name}: {message}")
        return ItemResult(
            outputs=[result.output],
            message=message,
            meta={
                "input_size": result.input_size,
                "output_size": result.output_size,
                "reduction_pct": round(result.reduction_pct, 2),
                "input_format": result.input_format,
                "output_format": result.output_format,
                "dimensions": f"{result.output_dimensions[0]}x{result.output_dimensions[1]}",
                "frames": result.frames,
            },
        )

    def make_preview(self, source: Path, options: Mapping[str, Any], services: Services, control: JobControl) -> engine.Preview:
        locate, env = engine_helpers(services)
        return engine.render_preview(source, self._settings(options, services, source), control=control, locate=locate, env=env)

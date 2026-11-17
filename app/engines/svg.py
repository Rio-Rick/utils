"""Rasterising SVG files.

Pillow cannot read SVG, so the first available backend is used:

1. ``cairosvg`` (pip, optional)
2. ``rsvg-convert`` (librsvg)
3. ImageMagick
4. QtSvg (always installed with the application; fonts need a running Qt application)

All backends run with argument lists (no shell) and honour the cancel control.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageColor

from app.core.control import JobControl
from app.core.errors import Cancelled, DependencyMissing, ToolError
from app.engines.process import path_arg, run_process

log = logging.getLogger(__name__)

SVG_EXTENSIONS = frozenset({"svg", "svgz"})
MAX_SIDE = 16384
Locator = Callable[[str], "str | None"]


def _clamp_size(width: int | None, height: int | None) -> tuple[int | None, int | None]:
    return (
        min(int(width), MAX_SIDE) if width else None,
        min(int(height), MAX_SIDE) if height else None,
    )


def render_svg(
    path: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    background: str | None = None,
    locate: Locator | None = None,
    control: JobControl | None = None,
    env: dict[str, str] | None = None,
) -> Image.Image:
    """Rasterise ``path``. ``width``/``height`` request a size (the aspect ratio is kept; one may be ``None``).

    ``locate("rsvg_convert" | "imagemagick")`` returns an executable path or ``None``.
    ``background`` (``#rrggbb``) fills transparent areas; ``None`` keeps transparency.
    """
    width, height = _clamp_size(width, height)
    errors: list[str] = []
    backends: list[tuple[str, Callable[[], Image.Image]]] = [
        ("cairosvg", lambda: _cairosvg(path, width, height)),
        ("rsvg-convert", lambda: _rsvg(path, width, height, locate, control, env)),
        ("ImageMagick", lambda: _magick(path, width, height, locate, control, env)),
        ("QtSvg", lambda: _qt(path, width, height)),
    ]
    for name, backend in backends:
        try:
            image = backend()
        except Cancelled:
            raise
        except DependencyMissing:
            continue
        except Exception as exc:  # a backend failing must not hide the next one
            log.debug("SVG backend %s failed for %s: %s", name, path, exc)
            errors.append(f"{name}: {exc}")
            continue
        image = image.convert("RGBA")
        if background:
            canvas = Image.new("RGBA", image.size, ImageColor.getrgb(background)[:3] + (255,))
            canvas.alpha_composite(image)
            return canvas.convert("RGB")
        return image
    detail = "; ".join(errors)
    raise ToolError(
        f"Cannot render {path.name}: no working SVG renderer",
        hint="Install cairosvg (pip install cairosvg) or librsvg (rsvg-convert).",
        detail=detail,
    )


def _cairosvg(path: Path, width: int | None, height: int | None) -> Image.Image:
    try:
        import cairosvg  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyMissing("cairosvg") from exc
    data = cairosvg.svg2png(url=str(path), output_width=width, output_height=height)
    return Image.open(io.BytesIO(data)).copy()


def _rsvg(path: Path, width: int | None, height: int | None, locate: Locator | None, control: JobControl | None, env: dict[str, str] | None) -> Image.Image:
    exe = locate("rsvg_convert") if locate else None
    if not exe:
        raise DependencyMissing("rsvg_convert")
    argv = [exe, "--format=png", "--keep-aspect-ratio"]
    if width:
        argv.append(f"--width={width}")
    if height:
        argv.append(f"--height={height}")
    argv.append(path_arg(path))
    result = run_process(argv, control=control, env=env, capture_stdout=True, stdout_limit=512 * 1024 * 1024)
    return Image.open(io.BytesIO(result.stdout)).copy()


def _magick(path: Path, width: int | None, height: int | None, locate: Locator | None, control: JobControl | None, env: dict[str, str] | None) -> Image.Image:
    exe = locate("imagemagick") if locate else None
    if not exe:
        raise DependencyMissing("imagemagick")
    argv = [exe, "-background", "none", "-density", "192", f"svg:{path_arg(path)}"]
    if width or height:
        argv += ["-resize", f"{width or ''}x{height or ''}"]
    argv.append("png:-")
    result = run_process(argv, control=control, env=env, capture_stdout=True, stdout_limit=512 * 1024 * 1024)
    return Image.open(io.BytesIO(result.stdout)).copy()


def _qt(path: Path, width: int | None, height: int | None) -> Image.Image:
    try:
        from PySide6.QtCore import QByteArray, Qt
        from PySide6.QtGui import QGuiApplication, QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer
    except ImportError as exc:
        raise DependencyMissing("PySide6 QtSvg") from exc
    if QGuiApplication.instance() is None:
        raise DependencyMissing("Qt application (needed for SVG text rendering)")
    data = path.read_bytes()
    if path.suffix.lower() == ".svgz" or data[:2] == b"\x1f\x8b":
        import gzip

        data = gzip.decompress(data)
    renderer = QSvgRenderer(QByteArray(data))
    if not renderer.isValid():
        raise ToolError(f"{path.name} is not a valid SVG file")
    natural = renderer.defaultSize()
    nw, nh = max(natural.width(), 1), max(natural.height(), 1)
    if width and height:
        scale = min(width / nw, height / nh)
    elif width:
        scale = width / nw
    elif height:
        scale = height / nh
    else:
        scale = 1.0
    out_w, out_h = max(1, round(nw * scale)), max(1, round(nh * scale))
    if out_w > MAX_SIDE or out_h > MAX_SIDE:
        raise ToolError("SVG render size is too large")
    image = QImage(out_w, out_h, QImage.Format.Format_RGBA8888_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    raw = bytes(image.constBits())
    return Image.frombuffer("RGBA", (out_w, out_h), raw, "raw", "RGBA", image.bytesPerLine(), 1).copy()


__all__ = ["SVG_EXTENSIONS", "render_svg"]

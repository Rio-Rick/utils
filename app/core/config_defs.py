"""Definitions of the core configuration domains.

One domain corresponds to one YAML file (``app.yaml``, ``ui.yaml`` ...). Tool
specific domains (``image``, ``contact_sheet``, ...) are contributed by the tools
themselves. Edit values in ``config/user/*.yaml`` or through *Settings*; the
shipped copies in ``config/defaults`` are generated from these definitions by
``scripts/gen_config.py``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from app.core.config_schema import Kind, Setting

ROUTES = (
    "dashboard", "image", "video", "audio", "file", "other", "history", "presets", "settings",
)
DASHBOARD_SECTIONS = (
    "active_jobs", "quick_actions", "favorites", "recent_tools", "recent_files", "dependencies",
)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BLOCKED_ENV = frozenset({"PATH", "IFS", "ENV", "BASH_ENV", "SHELLOPTS", "HOME", "USER", "LOGNAME"})
_BLOCKED_ENV_PREFIXES = ("LD_", "DYLD_", "PYTHON")


def validate_environment(value: Any) -> str | None:
    """Reject variables that could be used to hijack child processes (loader / interpreter hooks)."""
    for name, text in value.items():
        if not _ENV_NAME_RE.match(name):
            return f"invalid variable name {name!r}"
        if name in _BLOCKED_ENV or name.startswith(_BLOCKED_ENV_PREFIXES):
            return f"variable {name} may not be set from the configuration"
        if not isinstance(text, (str, int, float)) or isinstance(text, bool) or "\x00" in str(text):
            return f"value of {name} must be plain text"
    return None


def validate_tool_overrides(value: Any) -> str | None:
    """``{tool_id: {option: value}}`` - options are validated against the tool when it is used."""
    for tool_id, options in value.items():
        if not isinstance(options, Mapping) or not all(isinstance(k, str) for k in options):
            return f"overrides of {tool_id!r} must be a mapping of option -> value"
    return None


APP_SETTINGS: tuple[Setting, ...] = (
    Setting("name", Kind.STR, "Linux Media & Utility Toolkit", "Application name",
            "Shown in the window title and on the dashboard.", group="General"),
    Setting("language", Kind.CHOICE, "system", "Interface language",
            "Language of the user interface. 'system' follows your locale.",
            choices=("system", "en", "id"),
            choice_labels={"system": "System", "en": "English", "id": "Bahasa Indonesia"},
            restart=True, group="General"),
    Setting("directories.input", Kind.DIR, "", "Default input directory",
            "Folder the file pickers start in. Empty = your home folder.", group="Directories"),
    Setting("directories.output", Kind.DIR, "", "Default output directory",
            "Used when the output location is 'custom' and no folder was chosen. "
            "Empty = next to the source file.", group="Directories"),
    Setting("directories.temp", Kind.DIR, "", "Temporary directory",
            "Scratch space for intermediate files. Empty = the system temporary directory.",
            advanced=True, group="Directories"),
    Setting("max_concurrent_jobs", Kind.INT, 2, "Maximum concurrent jobs",
            "How many files are processed at the same time across the whole queue.",
            minimum=1, maximum=32, group="Performance"),
    Setting("priority", Kind.CHOICE, "normal", "Default processing priority",
            "CPU priority of workers and external processes. 'high' may need extra privileges "
            "and silently falls back to normal.",
            choices=("low", "normal", "high"), group="Performance"),
    Setting("startup.page", Kind.CHOICE, "dashboard", "Start page",
            "Page shown when the application starts.",
            choices=("dashboard", "last"),
            choice_labels={"dashboard": "Dashboard", "last": "Last visited page"}, group="Startup"),
    Setting("startup.check_dependencies", Kind.BOOL, True, "Check dependencies at startup",
            "Detect FFmpeg, ImageMagick and other optional programs in the background when the "
            "application starts.", group="Startup"),
    Setting("logging.level", Kind.CHOICE, "INFO", "Logging level",
            "Verbosity of the application log.",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"), group="Logging"),
    Setting("logging.to_file", Kind.BOOL, True, "Write log file",
            "Keep a rotating log file in the state directory (see docs).", group="Logging"),
    Setting("logging.max_size_mb", Kind.INT, 5, "Maximum log file size",
            "Size at which the log file is rotated.", minimum=1, maximum=100, unit="MB",
            advanced=True, group="Logging"),
    Setting("logging.backup_count", Kind.INT, 3, "Rotated log files to keep",
            "", minimum=0, maximum=20, advanced=True, group="Logging"),
    Setting("logging.redact_paths", Kind.CHOICE, "home", "Hide personal paths in logs",
            "'home' replaces your home directory by ~, 'basename' logs file names only, "
            "'none' logs full paths.",
            choices=("none", "home", "basename"), group="Logging"),
    Setting("history.enabled", Kind.BOOL, True, "Record history",
            "Remember processed files, results and errors.", group="History"),
    Setting("history.max_entries", Kind.INT, 5000, "Maximum history entries",
            "Oldest entries are removed beyond this number. 0 = unlimited.",
            minimum=0, maximum=1_000_000, group="History"),
    Setting("history.retention_days", Kind.INT, 0, "History retention",
            "Delete entries older than this many days. 0 = keep forever.",
            minimum=0, maximum=3650, unit="days", group="History"),
    Setting("history.store_options", Kind.BOOL, True, "Store job options in history",
            "Needed to retry a history entry with the same settings.", advanced=True, group="History"),
)

UI_SETTINGS: tuple[Setting, ...] = (
    Setting("theme", Kind.CHOICE, "system", "Theme", "Colour scheme of the application.",
            choices=("system", "dark", "light"),
            choice_labels={"system": "System", "dark": "Dark", "light": "Light"}, group="Appearance"),
    Setting("accent_color", Kind.COLOR, "#6c8cff", "Accent colour",
            "Used for selections, buttons and progress bars.", group="Appearance"),
    Setting("font.family", Kind.STR, "", "Font family",
            "Empty = the system font.", group="Appearance"),
    Setting("font.size", Kind.INT, 10, "Font size", "Base font size.",
            minimum=7, maximum=24, unit="pt", group="Appearance"),
    Setting("layout", Kind.CHOICE, "comfortable", "Layout density",
            "Compact reduces spacing and control height.",
            choices=("comfortable", "compact"),
            choice_labels={"comfortable": "Comfortable", "compact": "Compact"}, group="Appearance"),
    Setting("animations", Kind.BOOL, True, "Animations",
            "Fade and slide animations (toasts, transitions).", group="Appearance"),
    Setting("show_advanced", Kind.BOOL, False, "Show advanced settings",
            "Expand the 'Advanced' sections of every tool and settings page.", group="Appearance"),
    Setting("preview_visible", Kind.BOOL, True, "Show preview panel",
            "Show the preview panel next to the options of a tool.", group="Appearance"),
    Setting("sidebar.width", Kind.INT, 232, "Sidebar width", "",
            minimum=160, maximum=400, unit="px", group="Sidebar"),
    Setting("sidebar.collapsed", Kind.BOOL, False, "Collapse sidebar to icons", "", group="Sidebar"),
    Setting("sidebar.items", Kind.LIST, list(ROUTES), "Sidebar entries and order",
            "Entries shown in the sidebar, top to bottom. Settings is always available.",
            choices=ROUTES, group="Sidebar"),
    Setting("sidebar.show_pinned", Kind.BOOL, True, "Show pinned tools in the sidebar",
            "Favourite tools get a shortcut below the main entries.", group="Sidebar"),
    Setting("window.width", Kind.INT, 1280, "Window width", "", minimum=640, maximum=16000,
            unit="px", group="Window"),
    Setting("window.height", Kind.INT, 800, "Window height", "", minimum=480, maximum=16000,
            unit="px", group="Window"),
    Setting("window.x", Kind.INT, -1, "Window x position",
            "-1 lets the window manager decide.", minimum=-1, maximum=100000, unit="px",
            advanced=True, group="Window"),
    Setting("window.y", Kind.INT, -1, "Window y position",
            "-1 lets the window manager decide.", minimum=-1, maximum=100000, unit="px",
            advanced=True, group="Window"),
    Setting("window.maximized", Kind.BOOL, False, "Start maximised", "", group="Window"),
    Setting("window.remember_geometry", Kind.BOOL, True, "Remember window size and position",
            "Saved automatically when the window is closed.", group="Window"),
    Setting("dashboard.sections", Kind.LIST, list(DASHBOARD_SECTIONS), "Dashboard sections and order",
            "Sections shown on the dashboard, top to bottom.", choices=DASHBOARD_SECTIONS,
            group="Dashboard"),
    Setting("dashboard.recent_limit", Kind.INT, 8, "Items per dashboard section", "",
            minimum=1, maximum=50, group="Dashboard"),
    Setting("toasts.enabled", Kind.BOOL, True, "Toast notifications",
            "Small messages in the corner when a job finishes or fails.", group="Notifications"),
    Setting("toasts.duration_ms", Kind.INT, 4000, "Toast duration", "",
            minimum=1000, maximum=30000, unit="ms", group="Notifications"),
)

PROCESSING_SETTINGS: tuple[Setting, ...] = (
    Setting("output.location", Kind.CHOICE, "subfolder", "Output location",
            "Where results are written: next to the source, in a sub-folder next to the source, "
            "or in a custom folder. Tools inherit this unless they override it.",
            choices=("subfolder", "same", "custom"),
            choice_labels={"subfolder": "Sub-folder next to source", "same": "Same folder as source",
                           "custom": "Custom folder"}, group="Output"),
    Setting("output.subfolder_name", Kind.STR, "converted", "Sub-folder name",
            "Used when the location is 'sub-folder'.", group="Output",
            visible_if={"output.location": ("subfolder",)}),
    Setting("output.directory", Kind.DIR, "", "Output directory",
            "Used when the location is 'custom'. Empty = default output directory of the "
            "application settings.", group="Output", visible_if={"output.location": ("custom",)}),
    Setting("output.naming_template", Kind.STR, "{name}", "Output naming template",
            "File name without extension. Fields: {name} {ext} {index} {date} {time} {datetime} "
            "{parent}, plus tool specific ones such as {format} {width} {height}. "
            "Use {index:03d} for zero padding. Example: {name}_{width}x{height}.",
            group="Output"),
    Setting("output.overwrite", Kind.CHOICE, "rename", "If the output file exists",
            "'rename' adds a number, 'overwrite' replaces the file, 'skip' leaves it untouched.",
            choices=("rename", "overwrite", "skip"), group="Output"),
    Setting("output.preserve_structure", Kind.BOOL, True, "Keep folder structure",
            "When a folder is added, mirror its sub-folders in the output.", group="Output"),
    Setting("threads", Kind.INT, 0, "Threads per external process",
            "Thread count passed to FFmpeg and ImageMagick. 0 = automatic.",
            minimum=0, maximum=64, group="Execution"),
    Setting("cleanup_temp", Kind.BOOL, True, "Remove temporary files",
            "Delete intermediate files when an item finishes. Disable to inspect them.",
            group="Execution"),
    Setting("on_error", Kind.CHOICE, "continue", "When a file fails",
            "'continue' processes the remaining files, 'stop' cancels the rest of the batch.",
            choices=("continue", "stop"), group="Execution"),
    Setting("retry_count", Kind.INT, 0, "Automatic retries",
            "How often a failed file is retried automatically.", minimum=0, maximum=5,
            group="Execution"),
    Setting("batch_limit", Kind.INT, 2000, "Batch size limit",
            "Maximum number of files accepted in one batch. 0 = unlimited.",
            minimum=0, maximum=1_000_000, group="Queue"),
    Setting("queue.mode", Kind.CHOICE, "fifo", "Queue order",
            "Order in which queued files are started.",
            choices=("fifo", "smallest_first", "largest_first"),
            choice_labels={"fifo": "First in, first out", "smallest_first": "Smallest files first",
                           "largest_first": "Largest files first"}, group="Queue"),
    Setting("queue.auto_start", Kind.BOOL, True, "Start jobs immediately",
            "When disabled, new jobs wait in the queue until you press 'Start queue'.",
            group="Queue"),
)

ADVANCED_SETTINGS: tuple[Setting, ...] = (
    Setting("paths.ffmpeg", Kind.FILE, "", "FFmpeg executable",
            "Full path of a custom ffmpeg binary. Empty = search PATH.",
            advanced=True, sensitive=True, group="Executables"),
    Setting("paths.ffprobe", Kind.FILE, "", "FFprobe executable",
            "Full path of a custom ffprobe binary. Empty = search PATH.",
            advanced=True, sensitive=True, group="Executables"),
    Setting("paths.imagemagick", Kind.FILE, "", "ImageMagick executable",
            "Full path of 'magick' (or 'convert'). Empty = search PATH.",
            advanced=True, sensitive=True, group="Executables"),
    Setting("paths.rsvg_convert", Kind.FILE, "", "rsvg-convert executable",
            "Full path of rsvg-convert (SVG rendering). Empty = search PATH.",
            advanced=True, sensitive=True, group="Executables"),
    Setting("paths.exiftool", Kind.FILE, "", "ExifTool executable",
            "Full path of exiftool. Empty = search PATH.",
            advanced=True, sensitive=True, group="Executables"),
    Setting("paths.poppler", Kind.FILE, "", "Poppler (pdftoppm) executable",
            "Full path of pdftoppm. Empty = search PATH.",
            advanced=True, sensitive=True, group="Executables"),
    Setting("paths.sevenzip", Kind.FILE, "", "7-Zip executable",
            "Full path of 7zz / 7z / 7za. Empty = search PATH.",
            advanced=True, sensitive=True, group="Executables"),
    Setting("ffmpeg.global_args", Kind.ARGS, "", "FFmpeg global arguments",
            "Extra arguments placed before everything else, e.g. -hide_banner. No shell is used; "
            "quote values like in a terminal.", advanced=True, sensitive=True, group="FFmpeg"),
    Setting("ffmpeg.input_args", Kind.ARGS, "", "FFmpeg input arguments",
            "Extra arguments placed before each input (-i), e.g. -hwaccel auto.",
            advanced=True, sensitive=True, group="FFmpeg"),
    Setting("ffmpeg.output_args", Kind.ARGS, "", "FFmpeg output arguments",
            "Extra arguments placed before the output of each FFmpeg command.",
            advanced=True, sensitive=True, group="FFmpeg"),
    Setting("ffmpeg.loglevel", Kind.CHOICE, "error", "FFmpeg log level",
            "Verbosity of FFmpeg's own messages (shown in job logs).",
            choices=("quiet", "fatal", "error", "warning", "info"), advanced=True, group="FFmpeg"),
    Setting("environment", Kind.MAP, {}, "Environment variables for child processes",
            "Extra variables passed to FFmpeg & co, as NAME: value. Loader and interpreter "
            "variables (LD_*, PYTHON*, PATH ...) are refused.",
            advanced=True, sensitive=True, validator=validate_environment, group="Environment"),
    Setting("tool_overrides", Kind.MAP, {}, "Per-tool defaults",
            "Override the defaults of a single tool, e.g. image.converter: {quality: 90}. "
            "Option names are listed in docs/configuration.md.",
            advanced=True, validator=validate_tool_overrides, group="Per-tool"),
    Setting("limits.max_image_pixels", Kind.INT, 200_000_000, "Maximum image size",
            "Images with more pixels are refused (protects against decompression bombs and "
            "out-of-memory).", minimum=1_000_000, maximum=2_000_000_000, unit="pixels",
            advanced=True, group="Limits"),
    Setting("limits.process_timeout", Kind.INT, 0, "External process timeout",
            "Stop FFmpeg & co if one call runs longer than this. 0 = no limit.",
            minimum=0, maximum=86400, unit="s", advanced=True, group="Limits"),
)

SHORTCUT_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("navigate.dashboard", "Go to Dashboard", "Ctrl+1"),
    ("navigate.image", "Go to Image Tools", "Ctrl+2"),
    ("navigate.video", "Go to Video Tools", "Ctrl+3"),
    ("navigate.audio", "Go to Audio Tools", "Ctrl+4"),
    ("navigate.file", "Go to File Tools", "Ctrl+5"),
    ("navigate.other", "Go to Other Utilities", "Ctrl+6"),
    ("navigate.history", "Go to History", "Ctrl+7"),
    ("navigate.presets", "Go to Presets", "Ctrl+8"),
    ("navigate.settings", "Open Settings", "Ctrl+,"),
    ("app.command_palette", "Quick open (command palette)", "Ctrl+K"),
    ("app.toggle_sidebar", "Collapse / expand sidebar", "Ctrl+B"),
    ("app.toggle_advanced", "Show / hide advanced settings", "Ctrl+Shift+A"),
    ("app.jobs", "Show jobs and queue", "Ctrl+J"),
    ("app.quit", "Quit", "Ctrl+Q"),
    ("tool.add_files", "Add files", "Ctrl+O"),
    ("tool.add_folder", "Add folder", "Ctrl+Shift+O"),
    ("tool.start", "Start processing", "Ctrl+Return"),
    ("tool.cancel", "Cancel processing", "Ctrl+."),
    ("tool.pause", "Pause / resume processing", "Ctrl+Shift+P"),
    ("tool.select_all", "Select all files", "Ctrl+A"),
    ("tool.remove_selected", "Remove selected files", "Delete"),
    ("tool.clear", "Clear the file list", "Ctrl+Shift+Delete"),
)

SHORTCUT_SETTINGS: tuple[Setting, ...] = tuple(
    Setting(action, Kind.SHORTCUT, default, label, group=action.split(".")[0].title())
    for action, label, default in SHORTCUT_ACTIONS
)

PLUGIN_SETTINGS: tuple[Setting, ...] = (
    Setting("disabled", Kind.LIST, [], "Disabled tools",
            "Tool ids hidden everywhere (e.g. image.converter).", group="Tools"),
    Setting("order", Kind.LIST, [], "Custom tool order",
            "Tool ids in the order they should appear; tools not listed follow in default order.",
            group="Tools"),
    Setting("favorites", Kind.LIST, [], "Favourite (pinned) tools",
            "Tool ids pinned to the sidebar and the dashboard.", group="Tools"),
    Setting("user_plugins.enabled", Kind.BOOL, True, "Load user plugins",
            "Load tool plugins (*.py) from the user plugin directory. Plugins run with your "
            "permissions - only install plugins you trust.", sensitive=True, advanced=True,
            group="Plugins"),
    Setting("user_plugins.directory", Kind.DIR, "", "User plugin directory",
            "Empty = <data dir>/plugins.", sensitive=True, advanced=True, group="Plugins"),
    Setting("entry_points.enabled", Kind.BOOL, True, "Load pip-installed plugins",
            "Load tools registered by installed Python packages (entry-point group "
            "'linux_media_toolkit.tools').", sensitive=True, advanced=True, group="Plugins"),
)

CORE_SETTINGS: dict[str, tuple[Setting, ...]] = {
    "app": APP_SETTINGS,
    "ui": UI_SETTINGS,
    "processing": PROCESSING_SETTINGS,
    "advanced": ADVANCED_SETTINGS,
    "shortcuts": SHORTCUT_SETTINGS,
    "plugins": PLUGIN_SETTINGS,
}

DOMAIN_TITLES: dict[str, str] = {
    "app": "Application",
    "ui": "User interface",
    "processing": "Processing",
    "advanced": "Advanced",
    "shortcuts": "Keyboard shortcuts",
    "plugins": "Tools & plugins",
}

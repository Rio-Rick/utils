"""Exceptions shared by the processing layers (engines, tools, job manager)."""

from __future__ import annotations


class Cancelled(Exception):
    """The user cancelled the running item. Not an error: the job manager reports it as 'cancelled'."""


class ToolError(Exception):
    """A processing step failed in a way the user should hear about.

    ``message`` is a one-line summary; ``hint`` says what to try (shown below it in the UI);
    ``detail`` carries technical output (for example the last lines of FFmpeg's stderr)
    that goes to the job log rather than the toast.
    """

    def __init__(self, message: str, hint: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.detail = detail

    def __str__(self) -> str:
        return self.message


class DependencyMissing(ToolError):
    """A required external program or Python package is not available."""


class ProcessError(ToolError):
    """An external program exited with an error, timed out or produced too much output."""

    def __init__(self, message: str, *, returncode: int | None = None, hint: str = "", detail: str = "") -> None:
        super().__init__(message, hint, detail)
        self.returncode = returncode

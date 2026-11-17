"""Running external programs safely.

Every external program (FFmpeg, ImageMagick, ...) is started through :func:`run_process`:

* argument *lists* only - there is no shell, so file names can never be interpreted as commands;
* the child gets its own session/process group so cancel, pause and timeouts reach its children;
* stdout/stderr are drained by reader threads (no pipe deadlocks) with bounded memory;
* only the tail of stderr is kept, which is what the user needs to see when something fails.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import IO

from app.core.control import KILL_GRACE_SECONDS, JobControl
from app.core.errors import Cancelled, ProcessError

log = logging.getLogger(__name__)

_LINE_SPLIT = re.compile(rb"[\r\n]+")
_MAX_LINE_BYTES = 64 * 1024
_MAX_LOGGED_LINE = 2000
DEFAULT_STDOUT_LIMIT = 256 * 1024 * 1024
_NICE = {"low": 10, "normal": 0, "high": -5}


@dataclass(slots=True)
class ProcessResult:
    """Outcome of a finished external program."""

    argv0: str
    returncode: int
    stdout: bytes
    stderr_tail: list[str]
    duration: float

    @property
    def stderr_text(self) -> str:
        """The last stderr lines joined for display."""
        return "\n".join(self.stderr_tail)


def path_arg(path: str | os.PathLike[str]) -> str:
    """Absolute path string for an argument list.

    Starting with ``/`` it can neither be mistaken for an option (``-foo.png``) nor for a
    URL/protocol (``http:x``) by FFmpeg and friends.
    """
    text = os.path.abspath(os.fspath(path))
    if "\x00" in text:
        raise ValueError("path contains a NUL character")
    return text


def nice_for(priority: str) -> int:
    """``nice`` increment for a configured priority name."""
    return _NICE.get(priority, 0)


def apply_thread_priority(priority: str) -> None:
    """Best-effort CPU priority of the calling worker thread (Linux scheduling is per thread)."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice_for(priority))
    except (OSError, AttributeError):
        pass  # raising priority needs privileges; staying at the current level is fine


def _apply_process_priority(proc: subprocess.Popen[bytes], priority: str) -> None:
    nice = nice_for(priority)
    if nice == 0:
        return
    try:
        os.setpriority(os.PRIO_PGRP, proc.pid, nice)
    except OSError:
        log.debug("could not set priority %s for process %s", priority, proc.pid)


def _read_lines(stream: IO[bytes], callback: Callable[[str], None] | None, tail: deque[str] | None) -> None:
    """Split a byte stream at ``\\n`` and ``\\r`` (FFmpeg updates its status line with ``\\r``)."""
    buffer = b""
    while True:
        chunk = stream.read1(65536)  # type: ignore[attr-defined]
        if not chunk:
            break
        buffer += chunk
        *lines, buffer = _LINE_SPLIT.split(buffer)
        if len(buffer) > _MAX_LINE_BYTES:
            buffer = buffer[-_MAX_LINE_BYTES:]
        for raw in lines:
            _deliver(raw, callback, tail)
    _deliver(buffer, callback, tail)


def _deliver(raw: bytes, callback: Callable[[str], None] | None, tail: deque[str] | None) -> None:
    if not raw:
        return
    line = raw.decode("utf-8", "replace")
    if tail is not None:
        tail.append(line[:_MAX_LOGGED_LINE])
    if callback is not None:
        try:
            callback(line)
        except Exception:  # a faulty callback must not stall the pipe
            log.exception("output callback failed")


def _read_bytes(stream: IO[bytes], sink: bytearray, limit: int, overflow: threading.Event) -> None:
    while True:
        chunk = stream.read1(1 << 20)  # type: ignore[attr-defined]
        if not chunk:
            break
        sink += chunk
        if len(sink) > limit:
            overflow.set()
            break


def _kill_group(proc: subprocess.Popen[bytes], sig: int, *, force: bool = False) -> None:
    """Signal the child's process group. ``force`` also signals leftovers after the leader exited."""
    if not force and proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM, then SIGKILL after the grace period."""
    _kill_group(proc, signal.SIGCONT)
    _kill_group(proc, signal.SIGTERM)
    try:
        proc.wait(KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_group(proc, signal.SIGKILL)


def run_process(
    argv: Sequence[str | os.PathLike[str]],
    *,
    control: JobControl | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    timeout: float | None = None,
    capture_stdout: bool = False,
    stdout_limit: int = DEFAULT_STDOUT_LIMIT,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
    stderr_lines: int = 80,
    priority: str = "normal",
    check: bool = True,
) -> ProcessResult:
    """Run ``argv`` and wait for it.

    ``capture_stdout`` returns stdout as bytes (for example one PNG frame from FFmpeg);
    ``on_stdout_line`` instead streams stdout as text lines (FFmpeg ``-progress pipe:1``).
    Raises :class:`Cancelled` when ``control`` was cancelled and :class:`ProcessError` on
    timeout, oversized output, a program that cannot be started, or (with ``check``) a
    non-zero exit status.
    """
    args = [os.fspath(a) for a in argv]
    if not args or not args[0]:
        raise ValueError("empty command")
    if any("\x00" in a for a in args):
        raise ValueError("arguments must not contain NUL characters")
    name = os.path.basename(args[0])
    if control is not None:
        control.check()

    try:
        proc = subprocess.Popen(  # noqa: S603 - argument list, never a shell
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if (capture_stdout or on_stdout_line) else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=os.fspath(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            start_new_session=True,
            close_fds=True,
        )
    except FileNotFoundError as exc:
        raise ProcessError(f"{name} was not found", hint="Check the executable path in Settings → Dependencies.") from exc
    except PermissionError as exc:
        raise ProcessError(f"{name} cannot be executed (permission denied)") from exc
    except OSError as exc:
        raise ProcessError(f"{name} could not be started: {exc.strerror or exc}") from exc

    started = time.monotonic()
    if control is not None:
        control.attach(proc)
    _apply_process_priority(proc, priority)

    tail: deque[str] = deque(maxlen=max(1, stderr_lines))
    sink = bytearray()
    overflow = threading.Event()
    threads: list[threading.Thread] = []
    assert proc.stderr is not None
    threads.append(threading.Thread(target=_read_lines, args=(proc.stderr, on_stderr_line, tail), daemon=True))
    if proc.stdout is not None:
        if capture_stdout:
            threads.append(threading.Thread(target=_read_bytes, args=(proc.stdout, sink, stdout_limit, overflow), daemon=True))
        else:
            threads.append(threading.Thread(target=_read_lines, args=(proc.stdout, on_stdout_line, None), daemon=True))
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        while True:
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if overflow.is_set():
                _stop(proc)
                break
            if timeout is not None:
                paused = control.paused_seconds() if control is not None else 0.0
                if time.monotonic() - started - paused > timeout:
                    timed_out = True
                    _stop(proc)
                    break
    except BaseException:
        _stop(proc)
        raise
    finally:
        # Grandchildren that inherited the pipes would keep the readers blocked for their whole
        # lifetime: give the readers a moment to see EOF, then kill whatever is left in the group.
        deadline = time.monotonic() + 1.0
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(t.is_alive() for t in threads):
            _kill_group(proc, signal.SIGKILL, force=True)
            for thread in threads:
                thread.join(2.0)
        if not any(t.is_alive() for t in threads):  # closing a stream a reader is blocked on would hang
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        proc.wait()
        if control is not None:
            control.detach(proc)

    duration = time.monotonic() - started
    if control is not None and control.cancelled:
        raise Cancelled
    result = ProcessResult(name, proc.returncode, bytes(sink), list(tail), duration)
    if overflow.is_set():
        raise ProcessError(f"{name} produced more than {stdout_limit // (1024 * 1024)} MB of output", detail=result.stderr_text)
    if timed_out:
        raise ProcessError(f"{name} did not finish within {timeout:g} s and was stopped", detail=result.stderr_text)
    if check and proc.returncode != 0:
        last = result.stderr_tail[-1] if result.stderr_tail else ""
        message = f"{name} failed (exit code {proc.returncode})" + (f": {last}" if last else "")
        raise ProcessError(message, returncode=proc.returncode, detail=result.stderr_text)
    return result

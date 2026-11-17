"""Cooperative cancel / pause switch for one running work item.

Python code polls :meth:`JobControl.check` between steps; external programs started
through :func:`app.engines.process.run_process` are attached so cancelling terminates
their whole process group and pausing stops/continues it (``SIGSTOP``/``SIGCONT``).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time

from app.core.errors import Cancelled

log = logging.getLogger(__name__)

KILL_GRACE_SECONDS = 2.0


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    """Send ``sig`` to the process group of ``proc`` (children included); ignore a vanished process."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass
    except OSError:
        log.debug("could not signal process %s", proc.pid, exc_info=True)


class JobControl:
    """Thread-safe cancel/pause state shared by a worker thread and the UI thread."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[bytes]] = set()
        self._paused_total = 0.0
        self._paused_since: float | None = None

    # ------------------------------------------------------------------ state
    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` was called."""
        return self._cancelled.is_set()

    @property
    def paused(self) -> bool:
        """True while paused (and not cancelled)."""
        return not self._running.is_set() and not self._cancelled.is_set()

    def paused_seconds(self) -> float:
        """Total time spent paused so far (timeouts do not count it)."""
        with self._lock:
            extra = time.monotonic() - self._paused_since if self._paused_since is not None else 0.0
            return self._paused_total + extra

    # ------------------------------------------------------------------ commands
    def cancel(self) -> None:
        """Request cancellation; attached processes get SIGTERM now and SIGKILL after a short grace period."""
        with self._lock:
            if self._cancelled.is_set():
                return
            self._cancelled.set()
            self._end_pause_locked()
            processes = list(self._processes)
        self._running.set()  # wake up threads blocked in check()
        for proc in processes:
            _signal_group(proc, signal.SIGCONT)  # a stopped process cannot handle SIGTERM
            _signal_group(proc, signal.SIGTERM)
        if processes:
            timer = threading.Timer(KILL_GRACE_SECONDS, self._kill_survivors, args=(processes,))
            timer.daemon = True
            timer.start()

    @staticmethod
    def _kill_survivors(processes: list[subprocess.Popen[bytes]]) -> None:
        for proc in processes:
            _signal_group(proc, signal.SIGKILL)

    def pause(self) -> None:
        """Pause: Python code blocks in :meth:`check`, external processes are stopped."""
        with self._lock:
            if self._cancelled.is_set() or not self._running.is_set():
                return
            self._running.clear()
            self._paused_since = time.monotonic()
            processes = list(self._processes)
        for proc in processes:
            _signal_group(proc, signal.SIGSTOP)

    def resume(self) -> None:
        """Undo :meth:`pause`."""
        with self._lock:
            if self._running.is_set():
                return
            self._end_pause_locked()
            processes = list(self._processes)
        self._running.set()
        for proc in processes:
            _signal_group(proc, signal.SIGCONT)

    def _end_pause_locked(self) -> None:
        if self._paused_since is not None:
            self._paused_total += time.monotonic() - self._paused_since
            self._paused_since = None

    # ------------------------------------------------------------------ worker side
    def check(self) -> None:
        """Raise :class:`Cancelled` after a cancel request; block while paused."""
        while True:
            if self._cancelled.is_set():
                raise Cancelled
            if self._running.wait(0.25):
                if self._cancelled.is_set():
                    raise Cancelled
                return

    def attach(self, proc: subprocess.Popen[bytes]) -> None:
        """Register a started process so cancel/pause reach it (applies the current state at once)."""
        with self._lock:
            self._processes.add(proc)
            cancelled = self._cancelled.is_set()
            paused = not self._running.is_set()
        if cancelled:
            _signal_group(proc, signal.SIGTERM)
            timer = threading.Timer(KILL_GRACE_SECONDS, self._kill_survivors, args=([proc],))
            timer.daemon = True
            timer.start()
        elif paused:
            _signal_group(proc, signal.SIGSTOP)

    def detach(self, proc: subprocess.Popen[bytes]) -> None:
        """Forget ``proc`` (after it exited)."""
        with self._lock:
            self._processes.discard(proc)

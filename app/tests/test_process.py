"""Tests for the subprocess runner and the cancel/pause control."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from app.core.control import JobControl
from app.core.errors import Cancelled, ProcessError
from app.engines.process import nice_for, path_arg, run_process

PY = sys.executable


def py(code: str, *args: str) -> list[str]:
    return [PY, "-c", code, *args]


def proc_state(pid: int) -> str:
    """Single-letter state from /proc (R, S, T = stopped, Z ...); '' when the process is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return ""
    return stat.rsplit(")", 1)[1].split()[0]


def alive(pid: int) -> bool:
    return proc_state(pid) not in ("", "Z")


def wait_for(predicate, timeout: float = 5.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class TestRunProcess:
    def test_captures_stdout_bytes_and_stderr_tail(self) -> None:
        code = "import sys; sys.stdout.buffer.write(b'\\x89PNG\\x00data'); print('warn', file=sys.stderr)"
        result = run_process(py(code), capture_stdout=True)
        assert result.stdout == b"\x89PNG\x00data"
        assert result.stderr_tail == ["warn"]
        assert result.returncode == 0 and result.argv0 == os.path.basename(PY)

    def test_failure_raises_with_last_stderr_line(self) -> None:
        code = "import sys; print('first', file=sys.stderr); print('boom: bad input', file=sys.stderr); sys.exit(3)"
        with pytest.raises(ProcessError) as info:
            run_process(py(code))
        assert info.value.returncode == 3
        assert "boom: bad input" in info.value.message
        assert "first" in info.value.detail

    def test_check_false_returns_the_result(self) -> None:
        assert run_process(py("import sys; sys.exit(7)"), check=False).returncode == 7

    def test_missing_executable_is_a_process_error(self) -> None:
        with pytest.raises(ProcessError, match="was not found"):
            run_process(["/nonexistent/ffmpeg-xyz", "-version"])

    def test_no_shell_interpretation_of_arguments(self, tmp_path: Path) -> None:
        marker = tmp_path / "pwned"
        nasty = f"; touch {marker}; echo $(id) `id` | cat"
        result = run_process(py("import sys; sys.stdout.write(sys.argv[1])", nasty), capture_stdout=True)
        assert result.stdout.decode() == nasty
        assert not marker.exists()

    def test_nul_and_empty_commands_rejected(self) -> None:
        with pytest.raises(ValueError):
            run_process([])
        with pytest.raises(ValueError):
            run_process([PY, "a\x00b"])

    def test_stdin_is_closed(self) -> None:
        result = run_process(py("import sys; sys.stdout.write(repr(sys.stdin.read()))"), capture_stdout=True, timeout=10)
        assert result.stdout == b"''"

    def test_timeout(self) -> None:
        started = time.monotonic()
        with pytest.raises(ProcessError, match="did not finish"):
            run_process(py("import time; time.sleep(30)"), timeout=0.4)
        assert time.monotonic() - started < 6

    def test_stdout_limit_stops_runaway_output(self) -> None:
        code = "import sys\nwhile True: sys.stdout.buffer.write(b'x' * 65536); sys.stdout.flush()"
        with pytest.raises(ProcessError, match="more than"):
            run_process(py(code), capture_stdout=True, stdout_limit=1024 * 1024, timeout=20)

    def test_status_lines_split_on_carriage_return(self) -> None:
        lines: list[str] = []
        code = "import sys; sys.stderr.write('frame=1\\rframe=2\\rframe=3\\ndone\\n')"
        run_process(py(code), on_stderr_line=lines.append)
        assert lines == ["frame=1", "frame=2", "frame=3", "done"]

    def test_stdout_lines_stream_to_callback(self) -> None:
        lines: list[str] = []
        run_process(py("print('out_time_us=10'); print('progress=end')"), on_stdout_line=lines.append)
        assert lines == ["out_time_us=10", "progress=end"]

    def test_stderr_tail_is_bounded(self) -> None:
        result = run_process(py("import sys\nfor i in range(500): print(i, file=sys.stderr)"), stderr_lines=5)
        assert result.stderr_tail == ["495", "496", "497", "498", "499"]

    def test_faulty_callback_does_not_stall_the_process(self) -> None:
        def boom(line: str) -> None:
            raise RuntimeError("callback bug")

        assert run_process(py("print('a'); print('b')"), on_stdout_line=boom).returncode == 0

    def test_environment_is_passed(self) -> None:
        result = run_process(py("import os; print(os.environ['LMT_X'])"), env={"LMT_X": "42", "PATH": os.environ["PATH"]}, capture_stdout=True)
        assert result.stdout.strip() == b"42"

    def test_working_directory(self, tmp_path: Path) -> None:
        result = run_process(py("import os; print(os.getcwd())"), cwd=tmp_path, capture_stdout=True)
        assert Path(result.stdout.decode().strip()).resolve() == tmp_path.resolve()

    def test_grandchild_holding_the_pipe_does_not_hang(self) -> None:
        code = "import subprocess, sys; subprocess.Popen(['sleep', '30']); sys.exit(0)"
        started = time.monotonic()
        run_process(py(code), timeout=20)
        assert time.monotonic() - started < 12


class TestCancelAndPause:
    def test_cancel_before_start(self) -> None:
        control = JobControl()
        control.cancel()
        with pytest.raises(Cancelled):
            run_process(py("print(1)"), control=control)

    def test_cancel_kills_the_whole_process_group(self) -> None:
        control = JobControl()
        pids: list[int] = []
        # a shell that starts a background child; both must die
        script = "sleep 60 & echo $!; wait"
        outcome: list[BaseException] = []

        def worker() -> None:
            try:
                run_process(["sh", "-c", script], control=control, on_stdout_line=lambda line: pids.append(int(line)))
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        assert wait_for(lambda: bool(pids))
        child = pids[0]
        assert alive(child)
        started = time.monotonic()
        control.cancel()
        thread.join(10)
        assert not thread.is_alive()
        assert time.monotonic() - started < 6
        assert isinstance(outcome[0], Cancelled)
        assert wait_for(lambda: not alive(child)), "background child survived the cancel"

    def test_cancel_escalates_to_sigkill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.core.control.KILL_GRACE_SECONDS", 0.3)
        monkeypatch.setattr("app.engines.process.KILL_GRACE_SECONDS", 0.3)
        control = JobControl()
        pids: list[int] = []
        # ignores SIGTERM, so only SIGKILL can stop it
        code = "import os, signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint(os.getpid(), flush=True)\ntime.sleep(60)"
        outcome: list[BaseException] = []

        def worker() -> None:
            try:
                run_process(py(code), control=control, on_stdout_line=lambda line: pids.append(int(line)))
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        assert wait_for(lambda: bool(pids))
        control.cancel()
        thread.join(10)
        assert not thread.is_alive() and isinstance(outcome[0], Cancelled)
        assert wait_for(lambda: not alive(pids[0]))

    def test_pause_stops_the_process_and_resume_continues(self) -> None:
        control = JobControl()
        pids: list[int] = []
        code = "import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(1.2)\nprint('done', flush=True)"
        lines: list[str] = []

        def on_line(line: str) -> None:
            lines.append(line)
            if line.isdigit():
                pids.append(int(line))

        thread = threading.Thread(target=lambda: run_process(py(code), control=control, on_stdout_line=on_line, timeout=30))
        thread.start()
        assert wait_for(lambda: bool(pids))
        control.pause()
        assert control.paused
        assert wait_for(lambda: proc_state(pids[0]) == "T"), "process was not stopped"
        time.sleep(1.5)  # longer than the child's own sleep: it must not finish while paused
        assert "done" not in lines
        control.resume()
        thread.join(10)
        assert not thread.is_alive() and "done" in lines
        assert control.paused_seconds() >= 1.4

    def test_timeout_does_not_count_paused_time(self) -> None:
        control = JobControl()
        pids: list[int] = []
        code = "import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(0.6)"
        outcome: list[BaseException] = []

        def worker() -> None:
            try:
                run_process(py(code), control=control, on_stdout_line=lambda l: pids.append(int(l)), timeout=3)
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        assert wait_for(lambda: bool(pids))
        control.pause()
        time.sleep(3.6)  # more than the timeout
        control.resume()
        thread.join(10)
        assert not outcome, outcome

    def test_check_blocks_while_paused_and_raises_on_cancel(self) -> None:
        control = JobControl()
        control.pause()
        reached = threading.Event()

        def worker() -> None:
            control.check()
            reached.set()

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.4)
        assert not reached.is_set()
        control.resume()
        assert reached.wait(2)
        thread.join()

        control.pause()
        errors: list[BaseException] = []

        def blocked() -> None:
            try:
                control.check()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=blocked)
        thread.start()
        time.sleep(0.2)
        control.cancel()
        thread.join(2)
        assert errors and isinstance(errors[0], Cancelled)
        assert not control.paused

    def test_pause_after_cancel_is_ignored(self) -> None:
        control = JobControl()
        control.cancel()
        control.pause()
        assert not control.paused and control.cancelled


class TestHelpers:
    def test_path_arg_is_absolute_and_never_an_option(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert path_arg("-rf") == str(tmp_path / "-rf")
        assert path_arg("a/../b.png") == str(tmp_path / "b.png")
        with pytest.raises(ValueError):
            path_arg("bad\x00name")

    def test_nice_mapping(self) -> None:
        assert (nice_for("low"), nice_for("normal"), nice_for("high"), nice_for("???")) == (10, 0, -5, 0)

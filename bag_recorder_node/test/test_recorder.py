"""Unit tests for pipeline/bag_recorder_node/bag_recorder_node/recorder.py.

Pure-Python tests with subprocess.Popen + shutil.disk_usage mocked so
the suite runs anywhere — no live DDS, no ros2 CLI required.
"""
from __future__ import annotations

import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Make the package importable without installing it.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
)

from bag_recorder_node import recorder as rc  # noqa: E402


# ----- compose_bag_name --------------------------------------------------

def test_compose_bag_name_shape():
    when = datetime(2026, 5, 14, 15, 30, 22, tzinfo=timezone.utc)
    out = rc.compose_bag_name("trackdrive", "TrainingMap", now=when)
    assert out == "trackdrive_TrainingMap_20260514_153022"


def test_compose_bag_name_sanitises():
    when = datetime(2026, 5, 14, 15, 30, 22, tzinfo=timezone.utc)
    out = rc.compose_bag_name("track drive!", "../etc/passwd", now=when)
    for bad in (" ", "/", "..", "!"):
        assert bad not in out


# ----- check_free_disk ---------------------------------------------------

def _disk_usage_mock(free_gib: int):
    def _fn(path):
        return SimpleNamespace(
            total=500 * 1024 ** 3,
            used=(500 - free_gib) * 1024 ** 3,
            free=free_gib * 1024 ** 3,
        )
    return _fn


def test_check_free_disk_above_floor(tmp_path: Path):
    ok, free = rc.check_free_disk(
        tmp_path, min_gib=10, _disk_usage=_disk_usage_mock(free_gib=42),
    )
    assert ok is True
    assert free == 42


def test_check_free_disk_below_floor(tmp_path: Path):
    ok, free = rc.check_free_disk(
        tmp_path, min_gib=10, _disk_usage=_disk_usage_mock(free_gib=3),
    )
    assert ok is False
    assert free == 3


def test_check_free_disk_falls_back_to_parent_when_missing(tmp_path: Path):
    missing = tmp_path / "not-yet-created"
    captured = {}

    def _fn(path):
        captured["path"] = path
        return SimpleNamespace(total=0, used=0, free=20 * 1024 ** 3)

    ok, free = rc.check_free_disk(missing, min_gib=10, _disk_usage=_fn)
    assert ok is True
    assert free == 20
    assert captured["path"] == str(tmp_path)


# ----- start_recording ---------------------------------------------------

def _popen_mock(pid: int = 12345, *, poll_returns=None):
    poll_iter = iter(poll_returns if poll_returns is not None else [None])

    def _fn(cmd, **kwargs):
        proc = MagicMock()
        proc.pid = pid
        proc.poll = MagicMock(side_effect=lambda: next(poll_iter, None))
        proc.terminate = MagicMock()
        _fn.last_cmd = cmd
        _fn.last_kwargs = kwargs
        return proc
    _fn.last_cmd = None
    _fn.last_kwargs = None
    return _fn


def test_start_recording_refuses_when_disk_full(tmp_path: Path):
    with pytest.raises(rc.DiskFullError):
        rc.start_recording(
            "trackdrive_X_20260514_153022",
            bags_dir=tmp_path,
            staging_dir=tmp_path / "staging",
            min_free_gib=10,
            _popen=_popen_mock(),
            _disk_usage=_disk_usage_mock(free_gib=3),
        )


def test_start_recording_returns_state_dict(tmp_path: Path):
    popen = _popen_mock(pid=4242)
    staging = tmp_path / "staging"
    state = rc.start_recording(
        "trackdrive_X_20260514_153022",
        bags_dir=tmp_path,
        staging_dir=staging,
        min_free_gib=10,
        _popen=popen,
        _disk_usage=_disk_usage_mock(free_gib=42),
    )
    assert state["name"] == "trackdrive_X_20260514_153022"
    assert state["bag_path"] == str(tmp_path / "trackdrive_X_20260514_153022")
    assert state["staging_path"] == str(staging / "trackdrive_X_20260514_153022")
    assert state["pid"] == 4242
    assert state["state"] == "recording"
    # start_new_session is what makes SIGINT-to-pgid land on the recorder.
    assert popen.last_kwargs.get("start_new_session") is True


def test_start_recording_sources_both_ros_and_ws(tmp_path: Path):
    """fs_msgs/dv_msgs are only discoverable after sourcing the
    dv_pipeline_stack workspace setup; if we forget to source it the
    recorder spams "unknown type" warnings for every custom topic."""
    popen = _popen_mock(pid=1)
    rc.start_recording(
        "x_y_20260514_153022",
        bags_dir=tmp_path,
        staging_dir=tmp_path / "staging",
        min_free_gib=10,
        _popen=popen,
        _disk_usage=_disk_usage_mock(free_gib=42),
    )
    shell_cmd = popen.last_cmd[-1]
    assert "source /opt/ros/humble/setup.bash" in shell_cmd
    assert "source /dv_pipeline_stack_ws/install/setup.bash" in shell_cmd
    assert "ros2 bag record -s mcap -a" in shell_cmd


def test_start_recording_raises_on_immediate_exit(tmp_path: Path):
    popen = _popen_mock(pid=1, poll_returns=[127])
    with pytest.raises(rc.RecorderSpawnError):
        rc.start_recording(
            "x_y_20260514_153022",
            bags_dir=tmp_path,
            staging_dir=tmp_path / "staging",
            min_free_gib=10,
            _popen=popen,
            _disk_usage=_disk_usage_mock(free_gib=42),
        )


# ----- stop_recording ----------------------------------------------------

def test_stop_recording_skips_terminal_state():
    state = {"state": "stopped", "name": "x"}
    out = rc.stop_recording(state, _sleep=lambda s: None)
    assert out["state"] == "stopped"


def test_stop_recording_sigints_and_moves(tmp_path: Path, monkeypatch):
    staging = tmp_path / "staging" / "x_y_20260514_153022"
    staging.mkdir(parents=True)
    (staging / "metadata.yaml").write_text("dummy")
    (staging / "x_y_20260514_153022_0.mcap").write_bytes(b"\x00" * 1024)
    final_path = tmp_path / "bags" / "x_y_20260514_153022"

    poll_iter = iter([None, 0])
    proc = MagicMock()
    proc.poll = MagicMock(side_effect=lambda: next(poll_iter, 0))
    proc.terminate = MagicMock()

    state = {
        "name": "x_y_20260514_153022",
        "bag_path": str(final_path),
        "staging_path": str(staging),
        "log_path": str(tmp_path / "staging" / "x_y_20260514_153022.log"),
        "pid": 12345,
        "state": "recording",
        "_proc": proc,
        "_log_fh": MagicMock(),
    }

    killpg_calls = []
    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)),
    )
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    out = rc.stop_recording(state, _sleep=lambda s: None)

    assert out["state"] == "stopped"
    assert killpg_calls == [(12345, signal.SIGINT)]
    proc.terminate.assert_not_called()
    assert not staging.exists()
    assert final_path.exists()
    assert (final_path / "metadata.yaml").exists()


def test_stop_recording_escalates_to_sigterm_on_timeout(
    tmp_path: Path, monkeypatch,
):
    staging = tmp_path / "staging" / "x_y_20260514_153022"
    staging.mkdir(parents=True)
    final_path = tmp_path / "bags" / "x_y_20260514_153022"

    proc = MagicMock()
    proc.poll = MagicMock(return_value=None)  # Never exits cleanly.
    proc.terminate = MagicMock()

    state = {
        "name": "x_y_20260514_153022",
        "bag_path": str(final_path),
        "staging_path": str(staging),
        "pid": 12345,
        "state": "recording",
        "_proc": proc,
        "_log_fh": MagicMock(),
    }

    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    out = rc.stop_recording(state, timeout_s=0.0, _sleep=lambda s: None)
    proc.terminate.assert_called_once()
    assert out["state"] == "stopped"
    assert final_path.exists()


def test_stop_recording_marks_failed_when_staging_empty(monkeypatch):
    proc = MagicMock()
    proc.poll = MagicMock(return_value=0)

    state = {
        "name": "x_y_20260514_153022",
        "bag_path": "/tmp/nope/x_y_20260514_153022",
        "staging_path": "/tmp/also-nope/x_y_20260514_153022",
        "pid": 99999,
        "state": "recording",
        "_proc": proc,
        "_log_fh": MagicMock(),
    }
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    out = rc.stop_recording(state, _sleep=lambda s: None)
    assert out["state"] == "failed"
    assert "error" in out

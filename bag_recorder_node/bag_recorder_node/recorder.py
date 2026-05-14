"""Pure-helper subprocess management for the bag_recorder ROS node (#465).

Spawns / signals / finalises a single `ros2 bag record` process inside
the dv_pipeline_stack container. No rclpy imports here — keeps the
test suite cheap and gives the ROS-node layer (`node.py`) a clean
seam for unit testing without a live DDS graph.

## Wire-format

  start_recording(staging_dir, bags_dir, bag_name) → state dict:
      name         — bag dir name
      bag_path     — absolute path on host where the bag WILL be
                     (post-stop). Bind-mounted into the container as
                     `bags_dir / bag_name`; the operator finds it
                     here once stop_recording moves the staged dir.
      staging_path — where the recorder is actively writing
                     (`staging_dir / bag_name`, fast in-container fs).
      pid          — ros2 bag record's local PID.
      started_at   — unix timestamp at spawn.
      state        — "recording"
      _proc        — Popen handle (private; not for JSON).
      _log_fh      — captured-log file handle (private).

  stop_recording(state) → same dict, mutated:
      state       — "stopped" or "failed"
      stopped_at  — unix timestamp at clean exit
      error       — diagnostic string (only on failure)

## Why staging + move

Streaming small rosbag2 writes through the macOS virtiofs bind mount
caps at ~1 MB/s sustained on Docker Desktop, well below our active
2 MB/s topic mix. Writing to `/tmp/ifssim_bag_active/...` (real ext4
inside the container overlay) keeps the writer fast; a single bulk
`shutil.move` to `/bags/...` on stop crosses virtiofs sequentially,
which is its best-case workload. Same pattern as `tools/record_bag.sh`.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

_LOG = logging.getLogger(__name__)


# Minimum free space (GiB) on the bags-mount filesystem before we
# agree to start. ~800 MB/min for the full topic set in mcap; 30 min
# = ~24 GB. 10 GiB floor catches the "operator will run out of disk
# mid-lap" failure mode.
DEFAULT_MIN_FREE_GIB = 10

# Where rosbag2 writes during recording (real ext4 inside the
# container). Tunable for tests.
DEFAULT_STAGING_DIR = Path(os.environ.get(
    "IFSSIM_BAG_STAGING_DIR", "/tmp/ifssim_bag_active",
))

# Where the bag ends up after a clean stop. Bind-mounted to the
# host's repo-root `bags/` from `docker-compose.yml`.
DEFAULT_BAGS_DIR = Path(os.environ.get("IFSSIM_BAGS_DIR", "/bags"))


class BagRecorderError(Exception):
    """Base for any failure that should bubble back to the service."""


class DiskFullError(BagRecorderError):
    """Free space below the configured floor."""


class RecorderSpawnError(BagRecorderError):
    """`ros2 bag record` failed to launch (binary missing, env bad, etc.)."""


# Sanitiser for the track segment of the bag name.
_NAME_BAD = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize(s: str) -> str:
    s = Path(s).stem
    return _NAME_BAD.sub("_", s).strip("_")[:48]


def compose_bag_name(
    event_type: str,
    track: Optional[str],
    now: Optional[datetime] = None,
) -> str:
    """Return a directory name suitable for `ros2 bag record -o`.

    Shape: `<event>_<track>_<YYYYMMDD_HHMMSS>`. Either segment may be
    absent (we fall back to "unknown") but the timestamp is always
    present so two consecutive recordings can never collide.
    """
    when = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    event = _sanitize(event_type) or "unknown"
    trk = _sanitize(track) if track else "no-track"
    return f"{event}_{trk}_{when}"


def check_free_disk(
    path: Path,
    min_gib: int = DEFAULT_MIN_FREE_GIB,
    *,
    _disk_usage: Callable = shutil.disk_usage,
) -> tuple[bool, int]:
    """Return (ok, free_gib) for the filesystem at `path`.

    `_disk_usage` is injectable so the test suite can drive synthetic
    numbers without touching the real disk.
    """
    p = Path(path)
    if not p.exists():
        p = p.parent
        if not p.exists():
            raise BagRecorderError(f"no existing parent for {path}")
    usage = _disk_usage(str(p))
    free_gib = usage.free // (1024 ** 3)
    return free_gib >= min_gib, free_gib


def start_recording(
    bag_name: str,
    *,
    bags_dir: Path = DEFAULT_BAGS_DIR,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    min_free_gib: int = DEFAULT_MIN_FREE_GIB,
    ros2_bin: str = "ros2",
    ros_setup: str = "/opt/ros/humble/setup.bash",
    ws_setup: str = "/dv_pipeline_stack_ws/install/setup.bash",
    _popen: Callable = subprocess.Popen,
    _disk_usage: Callable = shutil.disk_usage,
) -> dict:
    """Spawn `ros2 bag record` in-process, return state dict.

    Pre-checks free disk on the bind-mounted bags dir (NOT staging —
    staging is small + ephemeral, but bags_dir is where the final
    output lives and gets free-checked).

    The recorder runs in its own process group (start_new_session=True)
    so the SIGINT we send on stop reaches `ros2 bag record` itself,
    not just the bash wrapper that sources the ROS env.
    """
    bags_dir = Path(bags_dir)
    staging_dir = Path(staging_dir)

    ok, free_gib = check_free_disk(
        bags_dir, min_gib=min_free_gib, _disk_usage=_disk_usage,
    )
    if not ok:
        raise DiskFullError(
            f"only {free_gib} GiB free on {bags_dir} "
            f"(need ≥{min_free_gib} GiB) — refusing to start recording"
        )

    bags_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / bag_name
    final_path = bags_dir / bag_name
    log_path = staging_dir / f"{bag_name}.log"

    # `ros2 bag record -o <name>` refuses to overwrite an existing
    # output folder. If a prior call to this function crashed AFTER
    # spawning the recorder (logger bug, OOM, …) the staging dir can
    # linger. Remove it here so the operator can retry cleanly without
    # having to shell in. In normal operation each bag_name carries a
    # UTC timestamp so this branch is a no-op.
    if staging_path.exists():
        _LOG.warning(
            "bag_recorder: stale staging dir %s — removing before respawn",
            staging_path,
        )
        try:
            shutil.rmtree(staging_path)
        except Exception as ex:
            raise RecorderSpawnError(
                f"could not remove stale staging dir {staging_path}: {ex}"
            )

    # Source ROS humble + the dv_pipeline_stack workspace so the
    # recorder sees fs_msgs/dv_msgs message definitions (otherwise
    # rosbag2 logs hundreds of "unknown type" warnings per second
    # for our custom topics and skips them).
    #
    # `exec` replaces the shell with the ros2 process so the captured
    # pid is the recorder's own pid, not bash's — important for
    # signal.kill(pid, SIGINT).
    cmd_str = (
        f"source {ros_setup} && "
        f"source {ws_setup} && "
        f"cd {staging_dir} && "
        f"exec {ros2_bin} bag record -s mcap -a -o {bag_name}"
    )

    log_fh = open(log_path, "wb")
    proc = _popen(
        ["bash", "-lc", cmd_str],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # Settle window — fail-fast on bad env / missing mcap plugin /
    # binary not on PATH. 0.5 s is empirically enough: real failures
    # happen in <100 ms, real successes survive the window.
    time.sleep(0.5)
    rc = proc.poll() if hasattr(proc, "poll") else None
    if rc is not None and rc != 0:
        log_fh.close()
        tail = ""
        try:
            tail = log_path.read_text(errors="replace")[-500:]
        except Exception:
            pass
        raise RecorderSpawnError(
            f"ros2 bag record exited rc={rc} immediately. log tail: {tail!r}"
        )

    return {
        "name": bag_name,
        "bag_path": str(final_path),
        "staging_path": str(staging_path),
        "log_path": str(log_path),
        "pid": proc.pid,
        "started_at": time.time(),
        "state": "recording",
        "_proc": proc,
        "_log_fh": log_fh,
    }


def stop_recording(
    state: dict,
    *,
    timeout_s: float = 10.0,
    _sleep: Callable = time.sleep,
) -> dict:
    """SIGINT the recorder, wait, move staging→final.

    Mutates + returns `state`. Idempotent on a terminal-state dict
    (skip-and-return for "stopped"/"failed"/"none").
    """
    if state.get("state") in ("stopped", "failed", "none"):
        return state

    proc = state.get("_proc")
    pid = state.get("pid")

    if proc is not None and proc.poll() is None:
        try:
            # SIGINT to the process group — recorder ran with
            # start_new_session=True so pgid == pid. mcap close path:
            # writes the chunk index, closes the file cleanly.
            os.killpg(os.getpgid(pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError) as ex:
            _LOG.warning("bag_recorder: killpg failed: %s", ex)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            _sleep(0.2)
        else:
            _LOG.warning(
                "bag_recorder: %s didn't exit on SIGINT within %ss — "
                "escalating to SIGTERM (bag's final chunk may be truncated)",
                state.get("name"), timeout_s,
            )
            try:
                proc.terminate()
            except Exception:
                pass

    log_fh = state.get("_log_fh")
    if log_fh is not None:
        try:
            log_fh.close()
        except Exception:
            pass

    staging_path = Path(state.get("staging_path") or state.get("bag_path", ""))
    if not staging_path.exists():
        state["state"] = "failed"
        state["error"] = "recorder exited without writing a bag dir"
        return state

    final_path = Path(state.get("bag_path", ""))
    if staging_path != final_path:
        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging_path), str(final_path))
            log_path = Path(state.get("log_path", ""))
            if log_path.exists():
                try:
                    shutil.move(
                        str(log_path),
                        str(final_path.parent / log_path.name),
                    )
                except Exception as ex:
                    _LOG.warning("bag log move failed: %s", ex)
        except Exception as ex:
            state["state"] = "failed"
            state["error"] = (
                f"recorder exited cleanly but staging→final move "
                f"failed: {ex} (bag still in {staging_path})"
            )
            return state

    state["state"] = "stopped"
    state["stopped_at"] = time.time()
    return state

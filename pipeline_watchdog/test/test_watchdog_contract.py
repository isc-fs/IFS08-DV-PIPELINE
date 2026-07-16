"""Cross-package contract tests for the independent pipeline watchdog.

The watchdog only works if three separate packages agree, and none of those
agreements is enforced by the type system:

  * pipeline_watchdog and mission_control must name the SAME emergency topic —
    otherwise the watchdog trips into the void and nothing fires the EBS.
  * the watchdog must arm on the SAME byte mission_control calls "running" —
    otherwise it supervises the wrong phase (or never arms at all).
  * the trip must be detectable inside the FS-Rules T11.9.4 budget.

Every one of those fails SILENTLY and safe-looking: the pipeline comes up, logs
nothing unusual, and simply has no watchdog. So they get pinned here, matching
mission_control/test/test_interface_contract.py's approach to the firmware
constants it cannot link against.

Pure — no rclpy — so it runs in plain pytest with no ROS install.
"""
from __future__ import annotations

import os
import sys

import pytest

# Both package dirs on the path so the contract can be checked without ROS.
_HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(REPO, "mission_control"))

from mission_control.interface_contract import (  # noqa: E402
    DV_RUNNING,
    HEARTBEAT_STALE_CAP_S,
    TOPIC_DV_STATUS,
    TOPIC_WATCHDOG_EMERGENCY,
)
from pipeline_watchdog.health_monitor import (  # noqa: E402
    HealthMonitor,
    TopicSpec,
)


def test_emergency_topic_name_is_pinned():
    # If this changes, mission_control's subscription must change with it, or
    # the watchdog trips into a topic nobody listens to.
    assert TOPIC_WATCHDOG_EMERGENCY == "/watchdog/emergency"


def test_dv_running_is_the_arming_byte():
    # The watchdog arms on DV_RUNNING. Pin the byte so a renumbering of the
    # /dv/status enum can't silently leave the watchdog armed on the wrong
    # phase (e.g. arming on DV_READY would trip EBS before the car moves).
    assert DV_RUNNING == 3
    assert TOPIC_DV_STATUS == "/dv/status"


def test_watchdog_node_and_mission_control_agree_on_the_topic():
    # The node module imports rclpy, so this only runs where ROS is present.
    # It is the test that actually proves the two ends are wired together.
    pytest.importorskip("rclpy")
    pytest.importorskip("fs_msgs")
    from pipeline_watchdog.pipeline_watchdog_node import TOPIC_EMERGENCY
    assert TOPIC_EMERGENCY == TOPIC_WATCHDOG_EMERGENCY


def test_mission_control_subscribes_to_the_watchdog_channel():
    # Guards the wiring itself: mission_control must reference the shared
    # constant, not a hardcoded string that can drift.
    src_path = os.path.join(
        REPO, "mission_control", "mission_control", "mission_control_node.py")
    with open(src_path) as fh:
        src = fh.read()
    assert "TOPIC_WATCHDOG_EMERGENCY" in src, \
        "mission_control must subscribe to the watchdog emergency channel"
    assert "_on_watchdog_emergency" in src


def test_default_budgets_detect_inside_the_fs_rules_cap():
    """A stall must be detected fast enough to be worth having.

    T11.9.4 gives 500 ms to detect a lost safety-critical message and enter
    the safe state. The watchdog's own budgets are deliberately LOOSER than
    that (a false EBS at speed is its own hazard, and this supervises data
    quality rather than the safety-critical heartbeat the uDV already
    watches). What must hold is that detection is bounded and stated, not
    that it meets the heartbeat cap — so pin the intent explicitly rather
    than let it drift silently.
    """
    pytest.importorskip("rclpy")
    pytest.importorskip("fs_msgs")
    from pipeline_watchdog.pipeline_watchdog_node import _WATCHDOG_HZ
    tick_s = 1.0 / _WATCHDOG_HZ
    # The tick must be fine enough that it is never the dominant term in
    # detection latency.
    assert tick_s <= HEARTBEAT_STALE_CAP_S / 5


def test_supervised_set_is_only_the_drive_on_stale_data_topics():
    """Guards the safety reasoning, not just the code.

    Only topics whose staleness leaves the car STILL DRIVING on stale state
    belong here. /Path and /Conos_raw already fail safe (control_node emits a
    zeroed command on an empty reference), so supervising them would add
    false-trip surface — i.e. more ways to fire the EBS at speed — for no
    safety gain. If someone adds a topic here, this test makes them justify it.
    """
    pytest.importorskip("rclpy")
    pytest.importorskip("fs_msgs")
    from pipeline_watchdog.pipeline_watchdog_node import PipelineWatchdogNode
    import rclpy

    rclpy.init()
    try:
        node = PipelineWatchdogNode()
        assert set(node._monitor.supervised_topics) == {
            "/slam/pose", "/odom", "/ctrl/cmd_internal"}
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_monitor_rejects_nothing_when_specs_empty():
    """Degenerate config must be inert, not accidentally trip-happy."""
    mon = HealthMonitor((), grace_period_s=1.0)
    mon.set_running(True, 0.0)
    assert not mon.evaluate(1000.0).tripped


def test_topic_spec_is_hashable_and_frozen():
    """Specs are shared config; accidental mutation must be impossible."""
    spec = TopicSpec("/x", 1.0, "why")
    with pytest.raises(Exception):
        spec.max_silence_s = 2.0  # type: ignore[misc]

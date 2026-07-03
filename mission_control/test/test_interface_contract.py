"""Unit tests for the stock-typed uDV ↔ mission_control interface contract.

Pure module — no rclpy / DDS — runs in plain pytest. Pins the AS / DV
byte values (mirrored in firmware) and the AMI→mission_id map.
"""
from __future__ import annotations

import os
import sys

import pytest

# Put the package dir (parent of test/) on the path so
# `import mission_control.interface_contract` resolves without ROS.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from mission_control.interface_contract import (  # noqa: E402
    AS_OFF,
    AS_EMERGENCY,
    AS_READY,
    AS_DRIVING,
    AS_FINISHED,
    DV_IDLE,
    DV_PREPARING,
    DV_READY,
    DV_RUNNING,
    DV_FINISHED,
    DV_EMERGENCY,
    DV_FAILED,
    HEARTBEAT_STALE_S,
    HEARTBEAT_STALE_CAP_S,
    ami_index_to_mission_id,
    mission_id_to_ami_index,
)


def test_as_state_bytes_match_fs_rules():
    # FS-Rules T14.9 / uDV MMEE byte ordering — mirrored in firmware.
    assert (AS_OFF, AS_EMERGENCY, AS_READY, AS_DRIVING, AS_FINISHED) == \
        (0, 1, 2, 3, 4)


def test_dv_status_bytes_are_distinct_and_ordered():
    bytes_ = [DV_IDLE, DV_PREPARING, DV_READY, DV_RUNNING,
              DV_FINISHED, DV_EMERGENCY, DV_FAILED]
    assert bytes_ == [0, 1, 2, 3, 4, 5, 6]
    assert len(set(bytes_)) == len(bytes_)


def test_heartbeat_stale_window_is_under_fs_rules_cap():
    # FS-Rules T11.9.4: a lost safety-critical message must be detected and
    # the safe state entered within 500 ms. The liveness window must stay
    # strictly under that cap (strict, since the pipeline still needs a tick
    # to act after detection). Mirrored firmware-side by the
    # DV_STATUS_STALE_MS < DV_STATUS_STALE_CAP_MS static_assert.
    assert HEARTBEAT_STALE_CAP_S == 0.5
    assert HEARTBEAT_STALE_S <= 0.5
    assert HEARTBEAT_STALE_S < HEARTBEAT_STALE_CAP_S


def test_node_uses_shared_heartbeat_window():
    # The reconciler's _ASSI_STALE_S must be the shared contract value, so
    # the pipeline uplink window and the firmware's DV_STATUS_STALE_MS can
    # never drift apart silently. mission_control_node pulls in rclpy; skip
    # (don't fail) when ROS isn't on the path, matching the pure-contract
    # tests above which never import it.
    pytest.importorskip("rclpy")
    from mission_control.mission_control_node import _ASSI_STALE_S
    assert _ASSI_STALE_S == HEARTBEAT_STALE_S


def test_detection_budget_leaves_reaction_margin():
    # T11.9.4 detection budget: the staleness watchdog only runs once per
    # reconcile tick, so worst-case detection = window + one tick period.
    # Demand >= 40 ms of reaction margin under the cap so tick granularity
    # can never silently eat the whole budget again (at the old 10 Hz tick
    # it was 0.4 + 0.1 = 0.5 s -> zero margin, right at the cap).
    pytest.importorskip("rclpy")
    from mission_control.mission_control_node import (
        _ASSI_STALE_S,
        _RECONCILE_HZ,
    )
    tick_s = 1.0 / _RECONCILE_HZ
    assert _ASSI_STALE_S + tick_s <= HEARTBEAT_STALE_CAP_S - 0.04


def test_dv_status_wire_cadence_matches_contract():
    # The firmware sizes DV_STATUS_STALE_MS (400 ms) as 4 missed cycles of
    # a 10 Hz /dv/status (dv_interface.h). The reconcile tick may run
    # faster for watchdog granularity, but the wire cadence must stay
    # 10 Hz and the throttle divisor exact, or the published period drifts
    # away from what the firmware window was sized against.
    pytest.importorskip("rclpy")
    from mission_control.mission_control_node import (
        _DV_STATUS_EVERY_N,
        _DV_STATUS_PUB_HZ,
        _RECONCILE_HZ,
    )
    assert _DV_STATUS_PUB_HZ == 10.0
    assert _RECONCILE_HZ % _DV_STATUS_PUB_HZ == 0
    assert _DV_STATUS_EVERY_N >= 1
    assert _DV_STATUS_EVERY_N * _DV_STATUS_PUB_HZ == _RECONCILE_HZ


def test_ami_index_maps_to_registry_mission_ids():
    # AMI ws2812.c index → mode_registry mission_id.
    assert ami_index_to_mission_id(4) == 1   # Track drive  → trackdrive
    assert ami_index_to_mission_id(3) == 2   # Autocross    → autocross
    assert ami_index_to_mission_id(1) == 3   # Acceleration → accel
    assert ami_index_to_mission_id(2) == 4   # Skidpad      → skidpad


def test_ami_non_autonomy_and_unknown_indices_are_no_mission():
    for idx in (0, 7, 8, 9):          # Manual, Shutdown, Aux1, Aux2
        assert ami_index_to_mission_id(idx) == 0
    for idx in (-1, 10, 99):          # out of range → never raise, no run
        assert ami_index_to_mission_id(idx) == 0


def test_ami_mapping_is_overridable():
    assert ami_index_to_mission_id(4, mapping={4: 5}) == 5


def test_mission_id_to_ami_index_round_trips():
    # Every runnable registry mission → an AMI index that maps back to it.
    for mission_id in (1, 2, 3, 4, 5):
        ami = mission_id_to_ami_index(mission_id)
        assert ami != 0
        assert ami_index_to_mission_id(ami) == mission_id


def test_mission_id_to_ami_index_zero_is_manual():
    assert mission_id_to_ami_index(0) == 0
    assert mission_id_to_ami_index(99) == 0

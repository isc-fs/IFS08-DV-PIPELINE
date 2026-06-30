"""Unit tests for the mission_control reconciler decision core.

Pure module — no rclpy. Pins the (AS state × mission × lifecycle) →
action table that replaces the old SetMission / RuntimeControl action
sequencing, including the mission-switch and straight-to-Driving cases.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from mission_control.interface_contract import (  # noqa: E402
    AS_OFF,
    AS_EMERGENCY,
    AS_READY,
    AS_DRIVING,
    AS_FINISHED,
    DV_IDLE,
    DV_READY,
    DV_RUNNING,
)
from mission_control.reconcile import (  # noqa: E402
    ReconcileAction,
    Target,
    is_runnable_mission,
    next_action,
    should_request_ebs,
    steady_dv_status,
    target_for,
)


TRACK = 1   # a runnable registry mission_id


# --------------------------------------------------------------------
# target_for
# --------------------------------------------------------------------

def test_target_ready_is_prepared():
    assert target_for(AS_READY, TRACK) is Target.PREPARED


def test_target_driving_is_running():
    assert target_for(AS_DRIVING, TRACK) is Target.RUNNING


def test_target_off_finished_emergency_unknown_are_down():
    for st in (AS_OFF, AS_FINISHED, AS_EMERGENCY, 99):
        assert target_for(st, TRACK) is Target.DOWN


def test_non_runnable_mission_collapses_to_down_even_when_driving():
    # A glitchy /ami/mission (0 / unmapped) can never start a run.
    assert target_for(AS_DRIVING, 0) is Target.DOWN
    assert target_for(AS_READY, 0) is Target.DOWN


# --------------------------------------------------------------------
# next_action — DOWN
# --------------------------------------------------------------------

def test_down_from_idle_is_noop():
    assert next_action(Target.DOWN, 0, 0, False) is ReconcileAction.NONE


def test_down_while_prepared_tears_down():
    assert next_action(Target.DOWN, 0, TRACK, False) is ReconcileAction.TEARDOWN


def test_down_while_active_tears_down():
    assert next_action(Target.DOWN, 0, TRACK, True) is ReconcileAction.TEARDOWN


# --------------------------------------------------------------------
# next_action — PREPARED
# --------------------------------------------------------------------

def test_prepared_from_idle_prepares():
    assert next_action(Target.PREPARED, TRACK, 0, False) is \
        ReconcileAction.PREPARE


def test_prepared_when_already_prepared_is_noop():
    assert next_action(Target.PREPARED, TRACK, TRACK, False) is \
        ReconcileAction.NONE


def test_prepared_with_wrong_mission_tears_down_first():
    assert next_action(Target.PREPARED, TRACK, 2, False) is \
        ReconcileAction.TEARDOWN


def test_prepared_while_active_tears_down():
    # AS dropped Driving→Ready: full teardown then re-prepare next tick.
    assert next_action(Target.PREPARED, TRACK, TRACK, True) is \
        ReconcileAction.TEARDOWN


# --------------------------------------------------------------------
# next_action — RUNNING
# --------------------------------------------------------------------

def test_running_from_idle_prepares_first():
    # Straight to Driving with nothing prepared → prepare before activate.
    assert next_action(Target.RUNNING, TRACK, 0, False) is \
        ReconcileAction.PREPARE


def test_running_when_prepared_activates():
    assert next_action(Target.RUNNING, TRACK, TRACK, False) is \
        ReconcileAction.ACTIVATE


def test_running_when_active_is_noop():
    assert next_action(Target.RUNNING, TRACK, TRACK, True) is \
        ReconcileAction.NONE


def test_running_mission_switch_midrun_tears_down():
    # Activated on mission 2 but AS now wants mission 1 → teardown first.
    assert next_action(Target.RUNNING, TRACK, 2, True) is \
        ReconcileAction.TEARDOWN


def test_running_wrong_mission_prepared_inactive_tears_down():
    assert next_action(Target.RUNNING, TRACK, 2, False) is \
        ReconcileAction.TEARDOWN


# --------------------------------------------------------------------
# EBS + status
# --------------------------------------------------------------------

def test_ebs_only_on_emergency():
    assert should_request_ebs(AS_EMERGENCY) is True
    for st in (AS_OFF, AS_READY, AS_DRIVING, AS_FINISHED):
        assert should_request_ebs(st) is False


def test_steady_status_mapping():
    assert steady_dv_status(0, False) == DV_IDLE
    assert steady_dv_status(TRACK, False) == DV_READY
    assert steady_dv_status(TRACK, True) == DV_RUNNING
    # activated dominates even if a mission id is set.
    assert steady_dv_status(0, True) == DV_RUNNING


def test_is_runnable_mission():
    assert is_runnable_mission(1) is True
    assert is_runnable_mission(0) is False
    assert is_runnable_mission(-1) is False

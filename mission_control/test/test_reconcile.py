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
from mission_control.interface_contract import FREE_RUN_MISSION_ID  # noqa: E402
from mission_control.reconcile import (  # noqa: E402
    ActiveLevel,
    EbsAction,
    ReconcileAction,
    Target,
    effective_mission_id,
    is_runnable_mission,
    next_action,
    next_ebs_action,
    should_request_ebs,
    steady_dv_status,
    target_for,
)


TRACK = 1   # a runnable registry mission_id
AUTOX = 2   # another runnable registry mission_id (== FREE_RUN_MISSION_ID)


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
    assert next_action(Target.DOWN, 0, 0, ActiveLevel.NONE) is \
        ReconcileAction.NONE


def test_down_while_prepared_tears_down():
    assert next_action(Target.DOWN, 0, TRACK, ActiveLevel.NONE) is \
        ReconcileAction.TEARDOWN


def test_down_while_active_tears_down():
    assert next_action(Target.DOWN, 0, TRACK, ActiveLevel.RUNNING) is \
        ReconcileAction.TEARDOWN


# --------------------------------------------------------------------
# next_action — PREPARED
# --------------------------------------------------------------------

def test_prepared_from_idle_prepares():
    assert next_action(Target.PREPARED, TRACK, 0, ActiveLevel.NONE) is \
        ReconcileAction.PREPARE


def test_prepared_when_already_prepared_is_noop():
    assert next_action(Target.PREPARED, TRACK, TRACK, ActiveLevel.NONE) is \
        ReconcileAction.NONE


def test_prepared_with_wrong_mission_tears_down_first():
    assert next_action(Target.PREPARED, TRACK, 2, ActiveLevel.NONE) is \
        ReconcileAction.TEARDOWN


def test_prepared_while_active_tears_down():
    # AS dropped Driving→Ready: full teardown then re-prepare next tick.
    assert next_action(Target.PREPARED, TRACK, TRACK, ActiveLevel.RUNNING) is \
        ReconcileAction.TEARDOWN


# --------------------------------------------------------------------
# next_action — RUNNING
# --------------------------------------------------------------------

def test_running_from_idle_prepares_first():
    # Straight to Driving with nothing prepared → prepare before activate.
    assert next_action(Target.RUNNING, TRACK, 0, ActiveLevel.NONE) is \
        ReconcileAction.PREPARE


def test_running_when_prepared_activates():
    assert next_action(Target.RUNNING, TRACK, TRACK, ActiveLevel.NONE) is \
        ReconcileAction.ACTIVATE


def test_running_when_active_is_noop():
    assert next_action(Target.RUNNING, TRACK, TRACK, ActiveLevel.RUNNING) is \
        ReconcileAction.NONE


def test_running_mission_switch_midrun_tears_down():
    # Activated on mission 2 but AS now wants mission 1 → teardown first.
    assert next_action(Target.RUNNING, TRACK, 2, ActiveLevel.RUNNING) is \
        ReconcileAction.TEARDOWN


def test_running_wrong_mission_prepared_inactive_tears_down():
    assert next_action(Target.RUNNING, TRACK, 2, ActiveLevel.NONE) is \
        ReconcileAction.TEARDOWN


# --------------------------------------------------------------------
# free-run — target_for floor
# --------------------------------------------------------------------

def test_free_run_off_matches_legacy_targets():
    # With the flag off, every mapping is exactly the pre-free-run table.
    assert target_for(AS_OFF, TRACK, free_run=False) is Target.DOWN
    assert target_for(AS_READY, TRACK, free_run=False) is Target.PREPARED
    assert target_for(AS_DRIVING, TRACK, free_run=False) is Target.RUNNING


def test_free_run_off_and_ready_no_mission_still_prepares_floor():
    # OFF / Ready / unknown all raise the floor when free_run is on, even
    # with no mission selected (the floor resolves autocross separately).
    for st in (AS_OFF, AS_READY, 99):
        assert target_for(st, 0, free_run=True) is Target.FLOOR


def test_free_run_driving_runs_full_only_with_runnable_mission():
    # A real selected mission + Driving → RUNNING (control reset + relay). A
    # standalone/manual "drive" (no runnable mission) stays on the FLOOR:
    # control keeps logging but the relay never opens.
    assert target_for(AS_DRIVING, TRACK, free_run=True) is Target.RUNNING
    assert target_for(AS_DRIVING, 0, free_run=True) is Target.FLOOR


def test_free_run_terminal_states_still_down():
    # Emergency + Finished win over the floor.
    assert target_for(AS_EMERGENCY, TRACK, free_run=True) is Target.DOWN
    assert target_for(AS_FINISHED, TRACK, free_run=True) is Target.DOWN


# --------------------------------------------------------------------
# free-run — effective_mission_id (selected-or-autocross)
# --------------------------------------------------------------------

def test_effective_mission_prefers_selection():
    assert effective_mission_id(
        TRACK, free_run=True, free_run_mission_id=AUTOX) == TRACK


def test_effective_mission_falls_back_to_autocross_when_free_run():
    assert effective_mission_id(
        0, free_run=True, free_run_mission_id=AUTOX) == AUTOX


def test_effective_mission_is_none_without_free_run():
    assert effective_mission_id(
        0, free_run=False, free_run_mission_id=AUTOX) == 0


# --------------------------------------------------------------------
# free-run — next_action for the floor + go hand-off
# --------------------------------------------------------------------

def test_floor_from_idle_prepares_all():
    # First bring-up of the floor: configure the whole stack.
    assert next_action(
        Target.FLOOR, AUTOX, 0, ActiveLevel.NONE) is \
        ReconcileAction.PREPARE


def test_floor_when_prepared_activates_whole_stack():
    # Prepared → bring the whole stack up as the FLOOR (control logging).
    assert next_action(
        Target.FLOOR, AUTOX, AUTOX, ActiveLevel.NONE) is \
        ReconcileAction.ACTIVATE_FLOOR


def test_floor_converged_is_noop():
    assert next_action(
        Target.FLOOR, AUTOX, AUTOX, ActiveLevel.FLOOR) is \
        ReconcileAction.NONE


def test_ready_to_driving_hand_off_resets_control():
    # The go hand-off: floor up for the mission (FLOOR) and AS now wants
    # RUNNING → clean-cycle control only, nothing torn down / re-prepared.
    assert next_action(
        Target.RUNNING, AUTOX, AUTOX, ActiveLevel.FLOOR) is \
        ReconcileAction.RESET_CONTROL


def test_running_straight_from_idle_activates_fresh():
    # Straight to Driving with the stack prepared but never floored → a plain
    # full activate already brings control up fresh (no reset needed).
    assert next_action(
        Target.RUNNING, AUTOX, AUTOX, ActiveLevel.NONE) is \
        ReconcileAction.ACTIVATE


def test_floor_from_running_tears_down_after_a_run():
    # Dropped back from Driving (RUNNING) to a floor state → tear down +
    # rebuild the floor clean.
    assert next_action(
        Target.FLOOR, AUTOX, AUTOX, ActiveLevel.RUNNING) is \
        ReconcileAction.TEARDOWN


def test_floor_mission_switch_tears_down_first():
    # Floor active for autocross, selection changed to trackdrive → tear
    # down before re-preparing the new mission.
    assert next_action(
        Target.FLOOR, TRACK, AUTOX, ActiveLevel.FLOOR) is \
        ReconcileAction.TEARDOWN


# --------------------------------------------------------------------
# EBS + status
# --------------------------------------------------------------------

def test_ebs_only_on_emergency():
    assert should_request_ebs(AS_EMERGENCY) is True
    for st in (AS_OFF, AS_READY, AS_DRIVING, AS_FINISHED):
        assert should_request_ebs(st) is False


# --------------------------------------------------------------------
# next_ebs_action — the "retry until acked" state machine
# --------------------------------------------------------------------

def _ebs(**kw):
    base = dict(emergency=True, acked=False,
                call_in_flight=False, service_ready=True)
    base.update(kw)
    return next_ebs_action(**base)


def test_ebs_dispatch_when_emergency_ready_and_idle():
    assert _ebs() is EbsAction.DISPATCH


def test_ebs_none_when_not_in_emergency():
    # Even if everything else says "go", no emergency → do nothing.
    assert _ebs(emergency=False) is EbsAction.NONE


def test_ebs_none_once_acked():
    # The ONLY latch: a positive ack stops further requests.
    assert _ebs(acked=True) is EbsAction.NONE


def test_ebs_wait_when_service_not_ready_does_not_latch():
    # Regression: an unavailable service must NOT kill the path — it waits
    # so a later tick retries once the server appears.
    assert _ebs(service_ready=False) is EbsAction.WAIT


def test_ebs_wait_while_call_in_flight():
    # Don't dispatch a duplicate while one is awaiting its response.
    assert _ebs(call_in_flight=True) is EbsAction.WAIT


def test_ebs_retries_after_a_dropped_call():
    # A call that went out but never acked leaves acked=False; once it is
    # no longer in flight and the service is ready we dispatch again.
    assert _ebs(call_in_flight=False, acked=False) is EbsAction.DISPATCH


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

"""Unit tests for car_supervisor.policy — AS state machine + AMI mapping.

Pure logic, no rclpy. Pins the safety-critical decisions: which phase
each AS state maps to, that actuation only happens in DRIVING, and the
AMI-index → registry-mission-id translation.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from car_supervisor.policy import (  # noqa: E402
    AS_DRIVING,
    AS_EMERGENCY,
    AS_FINISHED,
    AS_OFF,
    AS_READY,
    DEFAULT_AMI_TO_MISSION_ID,
    SupervisorPhase,
    ami_index_to_mission_id,
    is_runnable_mission,
    phase_for_as_state,
    should_actuate,
    should_trigger_ebs,
)


# ----------------------- AS state → phase -----------------------------

@pytest.mark.parametrize("as_state,phase", [
    (AS_OFF, SupervisorPhase.IDLE),
    (AS_EMERGENCY, SupervisorPhase.EMERGENCY),
    (AS_READY, SupervisorPhase.PREPARED),
    (AS_DRIVING, SupervisorPhase.DRIVING),
    (AS_FINISHED, SupervisorPhase.FINISHED),
])
def test_phase_for_each_as_state(as_state, phase):
    assert phase_for_as_state(as_state) is phase


@pytest.mark.parametrize("bad", [5, 99, 255, -1])
def test_unknown_as_state_is_failsafe_idle(bad):
    # Any unrecognised byte must map to IDLE — never actuate on garbage.
    assert phase_for_as_state(bad) is SupervisorPhase.IDLE


def test_as_state_byte_values_match_t14_9():
    # Guard the wire contract (FS-Rules T14.9 / uDV MMEE).
    assert (AS_OFF, AS_EMERGENCY, AS_READY, AS_DRIVING, AS_FINISHED) == \
        (0, 1, 2, 3, 4)


# ----------------------- actuation gating -----------------------------

def test_only_driving_actuates():
    for phase in SupervisorPhase:
        assert should_actuate(phase) is (phase is SupervisorPhase.DRIVING)


def test_only_emergency_triggers_ebs():
    for phase in SupervisorPhase:
        assert should_trigger_ebs(phase) is (phase is SupervisorPhase.EMERGENCY)


# ----------------------- AMI index → mission_id -----------------------

@pytest.mark.parametrize("ami_index,mission_id", [
    (1, 3),   # Acceleration → accel
    (2, 4),   # Skidpad      → skidpad
    (3, 2),   # Autocross    → autocross
    (4, 1),   # Track drive  → trackdrive
    (5, 5),   # EVS/EBS test → scruti
    (6, 5),   # Inspection   → scruti
])
def test_ami_runnable_missions_map(ami_index, mission_id):
    assert ami_index_to_mission_id(ami_index) == mission_id


@pytest.mark.parametrize("ami_index", [0, 7, 8, 9])
def test_ami_non_autonomy_slots_map_to_zero(ami_index):
    # Manual / Shutdown / Aux → no autonomy mission.
    assert ami_index_to_mission_id(ami_index) == 0


@pytest.mark.parametrize("ami_index", [10, 42, -1, 255])
def test_ami_out_of_range_is_no_mission(ami_index):
    # A glitchy index must never start an unintended run.
    assert ami_index_to_mission_id(ami_index) == 0


def test_ami_custom_mapping_overrides_default():
    custom = {4: 2}
    assert ami_index_to_mission_id(4, custom) == 2
    # absent keys fall through to 0, not the default table
    assert ami_index_to_mission_id(1, custom) == 0


def test_default_mapping_covers_all_ten_ami_slots():
    assert set(DEFAULT_AMI_TO_MISSION_ID.keys()) == set(range(10))


def test_default_mapping_targets_only_valid_registry_ids():
    # Registry mission_ids are 0 (none) + 1..5.
    assert set(DEFAULT_AMI_TO_MISSION_ID.values()) <= {0, 1, 2, 3, 4, 5}


# ----------------------- runnable predicate ---------------------------

@pytest.mark.parametrize("mid,runnable", [
    (0, False), (1, True), (5, True), (-1, False),
])
def test_is_runnable_mission(mid, runnable):
    assert is_runnable_mission(mid) is runnable

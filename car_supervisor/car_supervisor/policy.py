"""Pure supervision policy — AS state machine + AMI mission mapping.

No rclpy / ROS imports: every decision the car_supervisor makes about
*what* to do is a pure function of the uDV's AS state and the selected
AMI mission, so it can be exhaustively unit-tested with no ROS install.
car_supervisor_node owns the *how* (action clients, service calls, topic
I/O) and calls into here for the *what*.

The AS (Autonomous System) state is published by the uDV on /assi/state
(std_msgs/UInt8). Byte values are fixed by FS-Rules T14.9 and the uDV
MMEE (verified in IFS08-DV-uDV docs/PHYSICAL_TESTS.md P3):

    0x00 AS Off        0x01 AS Emergency   0x02 AS Ready
    0x03 AS Driving    0x04 AS Finished

The uDV — never the DVPC — owns this state machine. The supervisor only
*reacts* to it and must never actuate outside AS Driving.
"""
from __future__ import annotations

from enum import Enum


# AS state byte values on /assi/state (FS-Rules T14.9 / uDV MMEE).
AS_OFF = 0
AS_EMERGENCY = 1
AS_READY = 2
AS_DRIVING = 3
AS_FINISHED = 4


class SupervisorPhase(Enum):
    """What the supervisor should be doing for a given AS state."""

    IDLE = "idle"            # AS Off — torn down, no actuation
    PREPARED = "prepared"    # AS Ready — mission configured, no actuation
    DRIVING = "driving"      # AS Driving — runtime open, relay actuation
    EMERGENCY = "emergency"  # AS Emergency — EBS requested, no actuation
    FINISHED = "finished"    # AS Finished — runtime closed, no actuation


# AS state byte → supervisor phase. An unknown/garbage byte maps to the
# safe IDLE phase (fail-safe: never actuate on an unrecognised state).
_AS_STATE_TO_PHASE: dict[int, SupervisorPhase] = {
    AS_OFF: SupervisorPhase.IDLE,
    AS_EMERGENCY: SupervisorPhase.EMERGENCY,
    AS_READY: SupervisorPhase.PREPARED,
    AS_DRIVING: SupervisorPhase.DRIVING,
    AS_FINISHED: SupervisorPhase.FINISHED,
}


def phase_for_as_state(as_state: int) -> SupervisorPhase:
    """Map an AS state byte to the supervisor phase (IDLE if unknown)."""
    return _AS_STATE_TO_PHASE.get(int(as_state), SupervisorPhase.IDLE)


def should_actuate(phase: SupervisorPhase) -> bool:
    """True only in DRIVING — the one phase where commands reach the car.

    This is the safety invariant from 00_dv_pipeline_roadmap.md: the
    control output is "solo válido en ASState=DRIVING".
    """
    return phase is SupervisorPhase.DRIVING


def should_trigger_ebs(phase: SupervisorPhase) -> bool:
    """True only in EMERGENCY — the supervisor should request /force_ebs."""
    return phase is SupervisorPhase.EMERGENCY


# ---------------------------------------------------------------------
# AMI mission index → pipeline registry mission_id.
#
# The AMI board selects a mission 0..9 and the uDV republishes the index
# on /ami/mission (std_msgs/Int32). Verified AMI index→name table
# (IFS08-DV-uDV Core/Src/ws2812.c mission_colors):
#
#   0 Manual   1 Acceleration  2 Skidpad     3 Autocross  4 Track drive
#   5 EVS test 6 Inspection    7 Shutdown    8 Aux1       9 Aux2
#
# The pipeline registry (mode_manager.mode_registry) uses a DIFFERENT
# numbering: trackdrive=1, autocross=2, accel=3, skidpad=4, scruti=5.
# So an explicit mapping is required. mission_id 0 means "no autonomy
# mission / tear down".
#
# ⚠️  CONFIRM against AMI firmware before competition. Two open points
#     (flagged in docs/CAR_ADAPTATION.md):
#       - AMI 5 "EVS/EBS test" and AMI 6 "Inspection" are both mapped to
#         the registry "scruti" (scrutineering) mission for now.
#       - The uDV bench doc once referred to "mission 5 = track drive";
#         the firmware table (index 4 = Track drive) is authoritative.
# ---------------------------------------------------------------------
DEFAULT_AMI_TO_MISSION_ID: dict[int, int] = {
    0: 0,   # Manual        → no autonomy mission
    1: 3,   # Acceleration  → accel
    2: 4,   # Skidpad       → skidpad
    3: 2,   # Autocross     → autocross
    4: 1,   # Track drive   → trackdrive
    5: 5,   # EVS/EBS test  → scruti      (CONFIRM)
    6: 5,   # Inspection    → scruti      (CONFIRM)
    7: 0,   # Shutdown      → no mission
    8: 0,   # Aux1          → no mission
    9: 0,   # Aux2          → no mission
}


def ami_index_to_mission_id(
    ami_index: int,
    mapping: dict[int, int] | None = None,
) -> int:
    """Translate an AMI mission index to a pipeline registry mission_id.

    Returns 0 (no autonomy mission / tear down) for any index not in the
    mapping, including the non-autonomy AMI slots (Manual, Shutdown,
    Aux). Never raises — an out-of-range index is treated as "no
    mission" so a glitchy /ami/mission can't start an unintended run.
    """
    table = DEFAULT_AMI_TO_MISSION_ID if mapping is None else mapping
    return int(table.get(int(ami_index), 0))


def is_runnable_mission(mission_id: int) -> bool:
    """True if mission_id selects an actual autonomy mission (>0)."""
    return int(mission_id) > 0

"""Pure reconciler logic for mission_control's car/sim uDV interface.

mission_control no longer hosts the SetMission / RuntimeControl *actions*.
Instead it *reconciles* the autonomy lifecycle to what the uDV's AS state
machine demands, read level-triggered off /assi/state (+ the selected
mission off /ami/mission). This module is the pure decision core — no
rclpy — so the (subtle, safety-relevant) state logic is exhaustively
unit-tested; the node owns the ROS plumbing (the /activate_mode calls,
the /dv/status + /ctrl/cmd publishing, the staleness watchdog).

The model:

  AS state  →  a *target* lifecycle the pipeline should be in:

    AS_OFF / AS_FINISHED / unknown  → DOWN     (cleaned up)
    AS_READY                        → PREPARED (configured, inactive)
    AS_DRIVING                      → RUNNING  (activated)
    AS_EMERGENCY                    → DOWN  + request EBS

  A target that needs a mission collapses to DOWN when the selected
  mission isn't runnable (0 / unmapped) — a glitchy /ami/mission can
  never start an unintended run.

  Given the target and the pipeline's current (prepared_mission_id,
  activated), `next_action` emits ONE step toward convergence. Because
  each /activate_mode call is slow + async, the node applies one action
  per reconcile tick and re-evaluates when the call completes (or the
  AS state changes). Teardown is a full deactivate+cleanup (the
  ActivateMode contract has no "deactivate but stay configured"), which
  also gives the clean reset-between-runs SLAM needs.
"""
from __future__ import annotations

from enum import Enum

from mission_control.interface_contract import (
    AS_DRIVING,
    AS_EMERGENCY,
    AS_READY,
    DV_IDLE,
    DV_READY,
    DV_RUNNING,
)


class Target(Enum):
    """The lifecycle the AS state says the pipeline should be in."""

    DOWN = "down"          # deactivated + cleaned up
    PREPARED = "prepared"  # configured for the mission, inactive
    RUNNING = "running"    # activated


class ReconcileAction(Enum):
    """One step toward the target. The node maps these to /activate_mode."""

    NONE = "none"           # already converged — do nothing
    PREPARE = "prepare"     # activate_mode(desired mission, activate=False)
    ACTIVATE = "activate"   # activate_mode(prepared mission, activate=True)
    TEARDOWN = "teardown"   # activate_mode("", ...) — deactivate + cleanup


class EbsAction(Enum):
    """What the /force_ebs request loop should do this reconcile tick."""

    NONE = "none"          # not in emergency, or the request is already acked
    WAIT = "wait"          # want EBS but retry later (in flight / not ready)
    DISPATCH = "dispatch"  # call /force_ebs now


def is_runnable_mission(mission_id: int) -> bool:
    """True if mission_id selects an actual autonomy mission (>0)."""
    return int(mission_id) > 0


def should_request_ebs(as_state: int) -> bool:
    """True only in AS Emergency — the pipeline should request /force_ebs.

    (The pipeline-internal /ctrl/emergency path is handled separately by
    the node; this covers the AS-state-driven request.)
    """
    return int(as_state) == AS_EMERGENCY


def next_ebs_action(
    *,
    emergency: bool,
    acked: bool,
    call_in_flight: bool,
    service_ready: bool,
) -> EbsAction:
    """Decide the EBS request step so a dropped call is retried, not lost.

    The request is considered done ONLY once the uDV acks it (`acked`).
    Until then every reconcile tick re-evaluates: dispatch when the
    service is ready and no call is already in flight, otherwise wait and
    try again next tick. Crucially, an unready service or a failed call
    never latches `acked`, so a single dropped /force_ebs can't silently
    kill the EBS path for the rest of the session.

    Args:
        emergency: whether the node currently wants EBS asserted.
        acked: whether a prior /force_ebs call returned success.
        call_in_flight: whether a /force_ebs call is awaiting its response.
        service_ready: whether the /force_ebs client has a live server.
    """
    if not emergency or acked:
        return EbsAction.NONE
    if call_in_flight or not service_ready:
        return EbsAction.WAIT
    return EbsAction.DISPATCH


def target_for(as_state: int, desired_mission_id: int) -> Target:
    """Map (AS state, selected mission) to the desired lifecycle target.

    Fail-safe: any state that isn't explicitly Ready/Driving, and any
    non-runnable mission, maps to DOWN.
    """
    if not is_runnable_mission(desired_mission_id):
        return Target.DOWN
    if int(as_state) == AS_READY:
        return Target.PREPARED
    if int(as_state) == AS_DRIVING:
        return Target.RUNNING
    return Target.DOWN


def next_action(
    target: Target,
    desired_mission_id: int,
    prepared_mission_id: int,
    activated: bool,
) -> ReconcileAction:
    """Return the single step that moves current state toward `target`.

    Args:
        target: desired lifecycle (from `target_for`).
        desired_mission_id: registry mission_id the AS wants (>0 runnable).
        prepared_mission_id: mission currently configured (0 = none).
        activated: whether the autonomy nodes are active.
    """
    prepared = int(prepared_mission_id)
    desired = int(desired_mission_id)

    if target is Target.DOWN:
        if activated or prepared != 0:
            return ReconcileAction.TEARDOWN
        return ReconcileAction.NONE

    if target is Target.PREPARED:
        if activated:
            # Was running, AS dropped back to Ready — full teardown,
            # then re-prepare on a later tick.
            return ReconcileAction.TEARDOWN
        if prepared != desired:
            # Wrong (or no) mission configured.
            return (ReconcileAction.TEARDOWN if prepared != 0
                    else ReconcileAction.PREPARE)
        return ReconcileAction.NONE

    # target is RUNNING
    if prepared != desired:
        # Need the right mission configured first (mission switch or
        # straight-to-Driving before Ready ever prepared it).
        if activated or prepared != 0:
            return ReconcileAction.TEARDOWN
        return ReconcileAction.PREPARE
    if not activated:
        return ReconcileAction.ACTIVATE
    return ReconcileAction.NONE


def steady_dv_status(prepared_mission_id: int, activated: bool) -> int:
    """Map the converged lifecycle to a /dv/status byte.

    Covers the steady states only. The node overrides with the transient
    / terminal bytes it alone knows: DV_PREPARING (an activate_mode
    configure is in flight), DV_FAILED (a call failed), DV_FINISHED
    (/slam/finished), DV_EMERGENCY (/ctrl/emergency or AS Emergency).
    """
    if activated:
        return DV_RUNNING
    if int(prepared_mission_id) != 0:
        return DV_READY
    return DV_IDLE

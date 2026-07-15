"""Pure reconciler logic for mission_control's car/sim uDV interface.

mission_control no longer hosts the SetMission / RuntimeControl *actions*.
Instead it *reconciles* the autonomy lifecycle to what the uDV's AS state
machine demands, read level-triggered off /assi/state (+ the selected
mission off /ami/mission). This module is the pure decision core — no
rclpy — so the (subtle, safety-relevant) state logic is exhaustively
unit-tested; the node owns the ROS plumbing (the /activate_mode calls,
the /dv/status + /ctrl/cmd publishing, the staleness watchdog).

The model (normal, free_run=False — unchanged):

  AS state  →  a *target* lifecycle the pipeline should be in:

    AS_OFF / AS_FINISHED / unknown  → DOWN     (cleaned up)
    AS_READY                        → PREPARED (configured, inactive)
    AS_DRIVING                      → RUNNING  (activated)
    AS_EMERGENCY                    → DOWN  + request EBS

  A target that needs a mission collapses to DOWN when the selected
  mission isn't runnable (0 / unmapped) — a glitchy /ami/mission can
  never start an unintended run.

Free-run (free_run=True) adds an always-on *floor* for data collection:

    AS_OFF / AS_READY / unknown     → FLOOR   (the WHOLE autonomy stack —
                                       perception, SLAM, planning AND
                                       control — active; control logs its
                                       would-be commands on
                                       /ctrl/cmd_internal for pilot-vs-
                                       autonomy comparison, but the node
                                       does NOT relay them to /ctrl/cmd)
    AS_DRIVING (runnable mission)   → RUNNING (control CLEAN-RESET at the
                                       go edge, then relay opened)
    AS_DRIVING (no runnable mission)→ FLOOR   (standalone/manual "drive":
                                       control keeps logging, relay stays
                                       shut — the pipeline never actuates)
    AS_FINISHED / AS_EMERGENCY      → DOWN (terminal states still win;
                                       EMERGENCY also requests EBS)

  The floor prepares the operator-selected mission when one is dialed in,
  else FREE_RUN_MISSION_ID (autocross) — see `effective_mission_id`. Because
  the floor already runs the (selected) mission's whole stack, the
  OFF→Ready→Driving hand-off skips the expensive re-prep: it just
  RESET_CONTROLs (deactivate→activate control_node only), giving the run a
  clean controller with no SLAM reset or Numba re-JIT. control ran through
  the manual phase, so this reset is what stops the real run inheriting the
  manual-lap state. Two invariants keep the floor safe: the node relays
  /ctrl/cmd only at ActiveLevel.RUNNING (never on the FLOOR), and the
  /dv/status handshake byte still tracks the *real* AS state — so free-run
  cannot perturb the arm handshake.

  Given the target and the pipeline's current (prepared_mission_id,
  active_level), `next_action` emits ONE step toward convergence. Because
  each /activate_mode call is slow + async, the node applies one action
  per reconcile tick and re-evaluates when the call completes (or the
  AS state changes). A full teardown is a deactivate+cleanup of every
  node (the clean reset-between-runs SLAM needs).
"""
from __future__ import annotations

from enum import Enum

from mission_control.interface_contract import (
    AS_DRIVING,
    AS_EMERGENCY,
    AS_FINISHED,
    AS_READY,
    DV_IDLE,
    DV_READY,
    DV_RUNNING,
)


class Target(Enum):
    """The lifecycle the AS state says the pipeline should be in."""

    DOWN = "down"          # deactivated + cleaned up
    PREPARED = "prepared"  # configured for the mission, inactive
    FLOOR = "floor"        # whole stack active, control logging (relay shut)
    RUNNING = "running"    # whole stack active, control reset for the run


class ActiveLevel(Enum):
    """Which autonomy nodes are currently *activated*.

    Replaces the old `activated: bool` so the reconciler can distinguish the
    free-run floor (whole stack up, control logging) from a live run (control
    reset for the run, /ctrl/cmd relayed). Lifecycle-wise FLOOR and RUNNING are
    both "all nodes active"; they differ in whether control has been clean-
    reset for the current run and whether the node relays /ctrl/cmd.
    """

    NONE = "none"      # nothing activated (may still be prepared)
    FLOOR = "floor"    # all nodes active; control logging only (relay shut)
    RUNNING = "running"  # all nodes active; control reset for the run, relaying


class ReconcileAction(Enum):
    """One step toward the target. The node maps these to /activate_mode."""

    NONE = "none"           # already converged — do nothing
    PREPARE = "prepare"     # activate_mode(mission, activate=False) — all nodes
    ACTIVATE = "activate"   # activate_mode(mission, activate=True) — all nodes
    # Free-run: bring the whole stack up as the data-collection FLOOR. Same
    # activate_mode call as ACTIVATE (all nodes); the node just tracks it as
    # ActiveLevel.FLOOR so it does NOT relay /ctrl/cmd.
    ACTIVATE_FLOOR = "activate_floor"
    # Free-run go hand-off: clean-cycle control_node for the run.
    # activate_mode(mission, activate=True, reset_nodes=[control]) — control
    # deactivate→activate (fresh state), perception/SLAM untouched.
    RESET_CONTROL = "reset_control"
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


def effective_mission_id(
    desired_mission_id: int,
    *,
    free_run: bool,
    free_run_mission_id: int,
) -> int:
    """Mission the pipeline should prepare/run for.

    The operator-selected mission wins whenever it is runnable (>0) — so
    the free-run floor tracks the AMI selection and the "select mission,
    then arm" procedure hands off with nothing to re-prepare. With no
    runnable selection, free-run falls back to `free_run_mission_id`
    (autocross) so the floor still has a mission for SLAM/planning; a
    non-free-run pipeline with no selection has no mission (0).
    """
    if is_runnable_mission(desired_mission_id):
        return int(desired_mission_id)
    return int(free_run_mission_id) if free_run else 0


def target_for(
    as_state: int,
    desired_mission_id: int,
    *,
    free_run: bool = False,
) -> Target:
    """Map (AS state, selected mission) to the desired lifecycle target.

    Fail-safe (free_run=False, unchanged): any state that isn't explicitly
    Ready/Driving, and any non-runnable mission, maps to DOWN.

    free_run=True raises the floor: OFF / Ready / unknown → FLOOR (whole stack
    active, control logging), Driving with a runnable mission → RUNNING
    (control clean-reset, relay open), Driving without one → the floor (never
    actuate a standalone/manual mission). AS_FINISHED and AS_EMERGENCY still
    map to DOWN so a terminal state always wins over the floor.
    """
    st = int(as_state)
    if st == AS_EMERGENCY or st == AS_FINISHED:
        return Target.DOWN
    if st == AS_DRIVING:
        if is_runnable_mission(desired_mission_id):
            return Target.RUNNING
        return Target.FLOOR if free_run else Target.DOWN
    if st == AS_READY:
        if free_run:
            return Target.FLOOR
        return Target.PREPARED if is_runnable_mission(desired_mission_id) \
            else Target.DOWN
    # AS_OFF or any unknown byte.
    return Target.FLOOR if free_run else Target.DOWN


def next_action(
    target: Target,
    desired_mission_id: int,
    prepared_mission_id: int,
    active_level: ActiveLevel,
) -> ReconcileAction:
    """Return the single step that moves current state toward `target`.

    Args:
        target: desired lifecycle (from `target_for`).
        desired_mission_id: registry mission_id to prepare/run. This is the
            *effective* mission (see `effective_mission_id`) — the floor's
            fallback is already resolved by the caller.
        prepared_mission_id: mission currently configured (0 = none).
        active_level: which nodes are currently activated (ActiveLevel).
    """
    prepared = int(prepared_mission_id)
    desired = int(desired_mission_id)

    if target is Target.DOWN:
        if active_level is not ActiveLevel.NONE or prepared != 0:
            return ReconcileAction.TEARDOWN
        return ReconcileAction.NONE

    if target is Target.PREPARED:
        if active_level is not ActiveLevel.NONE:
            # Was running (or on the free-run floor), AS wants plain
            # PREPARED — full teardown, then re-prepare on a later tick.
            return ReconcileAction.TEARDOWN
        if prepared != desired:
            return (ReconcileAction.TEARDOWN if prepared != 0
                    else ReconcileAction.PREPARE)
        return ReconcileAction.NONE

    if target is Target.FLOOR:
        if prepared != desired:
            # Need the right mission configured first (mission switch, or
            # first bring-up of the floor).
            if active_level is not ActiveLevel.NONE or prepared != 0:
                return ReconcileAction.TEARDOWN
            return ReconcileAction.PREPARE
        if active_level is ActiveLevel.NONE:
            return ReconcileAction.ACTIVATE_FLOOR
        if active_level is ActiveLevel.FLOOR:
            return ReconcileAction.NONE
        # active_level is RUNNING — dropped back from Driving to a floor
        # state (run over). Tear down and rebuild the floor clean; the
        # hand-off we optimise is INTO Driving, not out of it.
        return ReconcileAction.TEARDOWN

    # target is RUNNING (live run — control reset + relay open)
    if prepared != desired:
        if active_level is not ActiveLevel.NONE or prepared != 0:
            return ReconcileAction.TEARDOWN
        return ReconcileAction.PREPARE
    if active_level is ActiveLevel.NONE:
        # Straight to Driving from a torn-down stack (never floored) — a
        # plain full activate brings control up fresh already.
        return ReconcileAction.ACTIVATE
    if active_level is ActiveLevel.FLOOR:
        # The go hand-off: whole stack is warm, clean-cycle control for the run.
        return ReconcileAction.RESET_CONTROL
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

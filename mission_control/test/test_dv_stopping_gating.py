"""DV_STOPPING emission gating — the byte must only ever appear mid-drive.

WHY THIS EXISTS (uDV#176 / uDV#177)
-----------------------------------
The firmware compares `/dv/status` for EQUALITY against the bytes it acts on
(READY=2, FINISHED=4, EMERGENCY=5, FAILED=6). Anything else — including our new
STOPPING=7 — is inert, which is what lets us ship byte 7 before the firmware
does. But there is one exception, and it bites:

    In AS Ready, an unrecognised byte means `dv_ready` is false → the uDV
    REFUSES GO. The car silently never launches.

So emitting 7 while arming is not "harmless and ignored", it is a car that
won't start with no obvious cause. The uDV team flagged this explicitly:
"don't emit STOPPING while we're arming or the car simply won't launch."

Latching `_stopping` is NOT a sufficient guard on its own: the latch clears only
once the stack is fully torn down, so a re-arm that beats the reset would sit in
AS Ready with 7 on the wire. The real guard is the AS state.

The second half of the file pins the FINISHED contract, because the firmware has
**no standstill check** on it (uDV#177): "only publish FINISHED once the car is
actually stopped", or the uDV opens the SDC at speed. That gate lives in
cone_slam's LapCounter; here we pin that mission_control cannot reorder the two.

Drives the real MissionControlNode — requires rclpy.
"""
from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")

from mission_control.interface_contract import (  # noqa: E402
    AS_DRIVING,
    AS_OFF,
    AS_READY,
    DV_EMERGENCY,
    DV_FINISHED,
    DV_IDLE,
    DV_READY,
    DV_RUNNING,
    DV_STOPPING,
)
from mission_control.mission_control_node import MissionControlNode  # noqa: E402
from mission_control.reconcile import ActiveLevel  # noqa: E402

TRACKDRIVE = 1


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = MissionControlNode()
    yield n
    n.destroy_node()


def _mid_run(n, as_state=AS_DRIVING):
    """Put the node in the state a live trackdrive run reaches."""
    n._as_state = as_state
    n._as_state_stamp = float("inf")      # never stale
    n._desired_mission_id = TRACKDRIVE
    n._prepared_mission_id = TRACKDRIVE
    n._active_level = ActiveLevel.RUNNING
    n._busy = False
    return n


# ------------------------------------------------- the launch-blocking bug

def test_stopping_is_never_emitted_while_arming(node):
    """THE BUG. A latched _stopping surviving into a re-arm must not put 7 on
    the wire during AS Ready — the uDV would read "not ready" and refuse GO,
    and the car would silently never launch."""
    n = _mid_run(node, as_state=AS_READY)
    n._stopping = True                     # latch survived from a prior run
    assert n._current_dv_status() != DV_STOPPING, \
        "emitted STOPPING while arming — uDV refuses GO, car never launches"


def test_arming_still_reports_ready_despite_a_stale_latch(node):
    """Avoiding byte 7 is only half of it — arming must still advertise
    readiness, or the uDV refuses GO for the opposite reason.

    ActiveLevel.NONE = prepared but not activated, which is exactly what the
    arming handshake looks like before the go.
    """
    n = _mid_run(node, as_state=AS_READY)
    n._active_level = ActiveLevel.NONE
    n._stopping = True
    assert n._current_dv_status() == DV_READY


def test_stopping_not_emitted_when_as_is_off(node):
    n = _mid_run(node, as_state=AS_OFF)
    n._stopping = True
    assert n._current_dv_status() == DV_IDLE


# ------------------------------------------------------ the intended path

def test_stopping_emitted_while_driving(node):
    """The one state it IS for."""
    n = _mid_run(node)
    n._stopping = True
    assert n._current_dv_status() == DV_STOPPING


def test_running_until_the_stop_is_requested(node):
    n = _mid_run(node)
    assert n._current_dv_status() == DV_RUNNING
    n._stopping = True
    assert n._current_dv_status() == DV_STOPPING


# ------------------------------------------------------------- priority

def test_finished_outranks_stopping(node):
    """Once stopped we ARE finished; STOPPING must not mask it, or the uDV
    never runs the AS Finished path and the mission never ends."""
    n = _mid_run(node)
    n._stopping = True
    n._finished = True
    assert n._current_dv_status() == DV_FINISHED


def test_emergency_outranks_stopping(node):
    """A real fault always outranks a tidy end-of-mission stop."""
    n = _mid_run(node)
    n._stopping = True
    n._emergency = True
    assert n._current_dv_status() == DV_EMERGENCY


def test_stopping_never_emitted_for_a_standalone_mission(node):
    """AMI 5/6 (EBS test / inspection) run without the pipeline. The firmware
    gates on mission_needs_pipeline anyway; belt-and-braces on our side."""
    n = _mid_run(node)
    n._desired_mission_id = 0              # not a runnable pipeline mission
    n._stopping = True
    assert n._current_dv_status() == DV_IDLE


# ------------------------------------------- the FINISHED standstill contract

def test_finished_is_only_ever_raised_by_slam(node):
    """The firmware has NO standstill check on byte 4 — it goes to AS Finished
    (EBS + SDC OPEN) the moment it sees it, at whatever speed. The standstill
    gate therefore lives entirely in cone_slam's LapCounter, and
    mission_control must be a pure relay of that decision: it must never
    synthesise FINISHED from its own state."""
    n = _mid_run(node)
    assert not n._finished
    n._stopping = True
    for _ in range(50):                    # a long stop, many ticks
        assert n._current_dv_status() == DV_STOPPING, \
            "mission_control invented FINISHED without /slam/finished"


# --------------------------------------------- lifecycle hygiene (regression)

def test_every_subscription_created_is_also_torn_down():
    """Mechanical invariant: any `self._sub_*` assigned a create_subscription()
    must appear in on_cleanup's teardown tuple.

    This exists because _sub_slam_stop_request was created but never destroyed
    — an edit that was supposed to add it to the tuple silently matched
    nothing, and nothing caught it: colcon builds fine, and no test reached
    cleanup. A leaked subscription survives cleanup→configure and then
    double-delivers to its callback.

    Source-level rather than behavioural on purpose: driving a real lifecycle
    cycle would test one path, whereas this cannot be fooled by a handle that
    someone forgets to add to the tuple in future.
    """
    import re
    import mission_control.mission_control_node as mod

    src = open(mod.__file__.replace(".pyc", ".py")).read()

    created = set(re.findall(r"(self\._sub_\w+)\s*=\s*self\.create_subscription",
                             src))
    assert created, "no subscriptions found — did the file layout change?"

    cleanup = src[src.index("def on_cleanup"):]
    cleanup = cleanup[:cleanup.index("return TransitionCallbackReturn")]

    missing = sorted(h for h in created if h not in cleanup)
    assert not missing, (
        f"subscription handles created but never torn down in on_cleanup: "
        f"{missing}. A leaked subscription survives cleanup→configure and "
        f"double-delivers to its callback.")

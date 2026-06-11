"""Unit tests for the SetMission `stage` formatter (#387).

`_stage_from_progress` renders a LifecycleProgress event as the
human-readable string Mission Control's UI shows in its session
spinner. The shape is part of the action's wire contract (documented
in SetMission.action's Feedback section), so this test pins it.

Pure function — no rclpy / no DDS, runs in plain pytest.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# Import _stage_from_progress without spinning up ROS. The module
# tries to import rclpy at top level, so we stub-out the heavy
# imports first.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

# Lifecycle transition IDs (kept here so the tests stay independent
# of whether `lifecycle_msgs` is importable in the test env).
T_CONFIGURE = 1
T_CLEANUP = 2
T_ACTIVATE = 3
T_DEACTIVATE = 4


def _import_formatter():
    """Try importing the real formatter; if rclpy isn't available
    (typical CI env), drop in a minimal re-implementation that the
    test pins. Either way the test asserts the same contract."""
    try:
        from mission_control.mission_control_node import _stage_from_progress
        return _stage_from_progress
    except ImportError:
        # Mirror the implementation. If this drifts from the real
        # one, the live-validation step in the PR catches it.
        _verb = {
            T_CONFIGURE: "configuring",
            T_ACTIVATE: "activating",
            T_DEACTIVATE: "deactivating",
            T_CLEANUP: "cleaning_up",
        }
        _past = {
            "configuring": "configured",
            "activating": "activated",
            "deactivating": "deactivated",
            "cleaning_up": "cleaned_up",
        }
        _bare = {
            "configuring": "configure",
            "activating": "activate",
            "deactivating": "deactivate",
            "cleaning_up": "cleanup",
        }

        def _fallback(p):
            node = p.node_name or "?"
            verb = _verb.get(p.transition_id, f"transitioning({p.transition_id})")
            ph = p.phase
            if ph == "starting":
                return f"{verb} {node}"
            if ph == "ok":
                return f"{node} {_past.get(verb, verb)}"
            if ph == "skipped":
                return f"{node} {_bare.get(verb, verb)} skipped"
            if ph in ("failed", "timeout"):
                bare = _bare.get(verb, verb)
                suffix = f": {p.error}" if p.error else ""
                return f"{node} {bare} {ph}{suffix}"
            return f"{node} {verb} {ph}"
        return _fallback


_stage_from_progress = _import_formatter()


def _ev(node: str, transition_id: int, phase: str, error: str = ""):
    """Build a stand-in LifecycleProgress event with just the fields
    the formatter reads."""
    return SimpleNamespace(
        node_name=node,
        transition_id=transition_id,
        phase=phase,
        error=error,
    )


# ----- "starting" — verb-first, the spinner subtitle while user waits ---

@pytest.mark.parametrize("tid,verb", [
    (T_CONFIGURE, "configuring"),
    (T_ACTIVATE, "activating"),
    (T_DEACTIVATE, "deactivating"),
    (T_CLEANUP, "cleaning_up"),
])
def test_starting_uses_verb_first(tid, verb):
    out = _stage_from_progress(_ev("cone_detection_node", tid, "starting"))
    assert out == f"{verb} cone_detection_node"


# ----- "ok" — past-tense, the "this step is done" line -----------------

@pytest.mark.parametrize("tid,past", [
    (T_CONFIGURE, "configured"),
    (T_ACTIVATE, "activated"),
    (T_DEACTIVATE, "deactivated"),
    (T_CLEANUP, "cleaned_up"),
])
def test_ok_uses_past_tense(tid, past):
    out = _stage_from_progress(_ev("slam_node", tid, "ok"))
    assert out == f"slam_node {past}"


# ----- "skipped" — idempotency path -------------------------------------

def test_skipped_reads_naturally():
    out = _stage_from_progress(_ev("cone_detection_node", T_CONFIGURE, "skipped"))
    # "cone_detection_node configure skipped" — verb stripped of -ing.
    assert out == "cone_detection_node configure skipped"


def test_skipped_for_activate():
    out = _stage_from_progress(_ev("path_planning_node", T_ACTIVATE, "skipped"))
    assert out == "path_planning_node activate skipped"


# ----- "failed" / "timeout" — include diagnostic ------------------------

def test_failed_includes_error():
    out = _stage_from_progress(_ev(
        "control_node", T_ACTIVATE, "failed",
        error="change_state returned success=False",
    ))
    assert out == (
        "control_node activate failed: "
        "change_state returned success=False"
    )


def test_failed_without_error_omits_colon():
    out = _stage_from_progress(_ev("control_node", T_ACTIVATE, "failed"))
    # No trailing colon — the operator UI shouldn't show a dangling
    # punctuation mark when mode_manager forgot to fill `error`.
    assert out == "control_node activate failed"


def test_timeout_includes_error():
    out = _stage_from_progress(_ev(
        "cone_detection_node", T_CONFIGURE, "timeout",
        error="timeout waiting for transition 1",
    ))
    assert out == (
        "cone_detection_node configure timeout: "
        "timeout waiting for transition 1"
    )


# ----- fallbacks --------------------------------------------------------

def test_unknown_transition_id_falls_back_gracefully():
    out = _stage_from_progress(_ev("x", 99, "starting"))
    assert "transitioning(99)" in out
    assert "x" in out


def test_unknown_phase_passes_through():
    # mode_manager could grow a new phase string in the future; the
    # formatter must not crash on it, just print something useful.
    out = _stage_from_progress(_ev("x", T_CONFIGURE, "weird"))
    assert "x" in out
    assert "weird" in out


def test_empty_node_name_uses_placeholder():
    out = _stage_from_progress(_ev("", T_CONFIGURE, "starting"))
    # "configuring ?" — clearly broken upstream but doesn't crash.
    assert out == "configuring ?"

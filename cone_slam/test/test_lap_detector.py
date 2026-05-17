"""Unit tests for LapDetector — big-orange-gate crossing detector."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cone_slam.landmark_db import Landmark, LandmarkDb
from cone_slam.lap_detector import LapDetector
from cone_slam.phase1_mapper import Pose2D


def _make_big_orange_landmark(lid: int, x: float, y: float) -> Landmark:
    return Landmark(
        id=lid,
        position=np.array([x, y, 0.0]),
        n_observations=3,
        last_seen_step=0,
        sigma_xy=0.20,
        is_big_orange=True,
    )


def _make_yellow_landmark(lid: int, x: float, y: float) -> Landmark:
    return Landmark(
        id=lid,
        position=np.array([x, y, 0.0]),
        n_observations=3,
        last_seen_step=0,
        sigma_xy=0.20,
        is_big_orange=False,
    )


# ---------------------------------------------------------------------
# Gate identification
# ---------------------------------------------------------------------

def test_gate_not_armed_without_big_orange_cones() -> None:
    """No big-orange cones near spawn → detector stays in BEFORE_START."""
    d = LapDetector()
    landmarks = [_make_yellow_landmark(0, 2.0, 1.5)]
    d.observe(Pose2D(0.0, 0.0, 0.0), landmarks)
    assert d.state == "BEFORE_START"
    assert not d.gate_armed


def test_gate_armed_when_two_big_orange_near_spawn() -> None:
    """Two big-orange cones within search radius + spacing → gate
    snapshots; state → APPROACHING."""
    d = LapDetector()
    landmarks = [
        _make_big_orange_landmark(0, 2.0, -1.5),
        _make_big_orange_landmark(1, 2.0, +1.5),
    ]
    d.observe(Pose2D(0.0, 0.0, 0.0), landmarks)
    assert d.gate_armed
    assert d.state == "APPROACHING"


def test_gate_ignores_far_away_big_orange_pairs() -> None:
    """Big-orange pair >5 m from spawn point shouldn't arm the gate."""
    d = LapDetector(max_gate_search_radius_m=5.0)
    landmarks = [
        _make_big_orange_landmark(0, 50.0, -1.5),
        _make_big_orange_landmark(1, 50.0, +1.5),
    ]
    d.observe(Pose2D(0.0, 0.0, 0.0), landmarks)
    assert not d.gate_armed


def test_gate_ignores_too_wide_big_orange_pairs() -> None:
    """Big-orange pair > max_pair_spacing apart shouldn't arm —
    likely separate orange cones, not a real gate."""
    d = LapDetector(max_gate_pair_spacing_m=6.0)
    landmarks = [
        _make_big_orange_landmark(0, 2.0, -10.0),
        _make_big_orange_landmark(1, 2.0, +10.0),
    ]
    d.observe(Pose2D(0.0, 0.0, 0.0), landmarks)
    assert not d.gate_armed


# ---------------------------------------------------------------------
# Crossing detection
# ---------------------------------------------------------------------

def test_crossing_doesnt_fire_at_start() -> None:
    """Standing on the start line shouldn't fire — min_lap_distance
    not yet met."""
    d = LapDetector(min_lap_distance_m=30.0)
    landmarks = [
        _make_big_orange_landmark(0, 2.0, -1.5),
        _make_big_orange_landmark(1, 2.0, +1.5),
    ]
    # Stand still on the start line and call observe a few times.
    for _ in range(5):
        fired = d.observe(Pose2D(0.0, 0.0, 0.0), landmarks)
        assert not fired


def test_crossing_fires_after_min_lap_distance() -> None:
    """Drive a full loop (synthetic) and verify the crossing fires
    once when the car returns to and passes the gate, but NOT
    before min_lap_distance is met."""
    d = LapDetector(min_lap_distance_m=30.0)
    landmarks = [
        _make_big_orange_landmark(0, 2.0, -1.5),
        _make_big_orange_landmark(1, 2.0, +1.5),
    ]

    # Arm the gate at spawn.
    assert not d.observe(Pose2D(0.0, 0.0, 0.0), landmarks)
    assert d.gate_armed

    # Synthetic loop: drive 50 m forward, then 50 m back. The
    # "back" leg recrosses x=2 (the gate midpoint) at some point.
    fired = False
    # Forward leg: x from 0 to 50.
    for x in np.linspace(0.0, 50.0, 200):
        if d.observe(Pose2D(float(x), 0.0, 0.0), landmarks):
            # Forward direction shouldn't fire (we never sign-flipped
            # from negative back through zero — we started AT the gate
            # so we're already past it).
            fired = True
    assert not fired
    # Return leg: x from 50 back to -5 (crosses x=2 from + to -).
    fired_on_return = []
    for x in np.linspace(50.0, -5.0, 200):
        if d.observe(Pose2D(float(x), 0.0, 0.0), landmarks):
            fired_on_return.append(float(x))
    assert len(fired_on_return) == 1, (
        f"expected exactly one crossing fire, got {len(fired_on_return)}: "
        f"{fired_on_return}"
    )


def test_crossing_doesnt_refire_once_crossed() -> None:
    """After lap_completed fires, additional ticks must not refire
    until reset_for_next_lap() is called."""
    d = LapDetector(min_lap_distance_m=10.0)
    landmarks = [
        _make_big_orange_landmark(0, 2.0, -1.5),
        _make_big_orange_landmark(1, 2.0, +1.5),
    ]
    d.observe(Pose2D(0.0, 0.0, 0.0), landmarks)
    # Drive forward past min_distance then back across the gate.
    for x in np.linspace(0.0, 20.0, 100):
        d.observe(Pose2D(float(x), 0.0, 0.0), landmarks)
    # Now cross back through.
    first_fire = None
    for x in np.linspace(20.0, -5.0, 100):
        if d.observe(Pose2D(float(x), 0.0, 0.0), landmarks):
            first_fire = x
            break
    assert first_fire is not None
    # Continue driving — no more fires.
    extra_fires = 0
    for x in np.linspace(-5.0, 50.0, 200):
        if d.observe(Pose2D(float(x), 0.0, 0.0), landmarks):
            extra_fires += 1
    assert extra_fires == 0


def test_reset_for_next_lap_rearms() -> None:
    d = LapDetector(min_lap_distance_m=10.0)
    landmarks = [
        _make_big_orange_landmark(0, 2.0, -1.5),
        _make_big_orange_landmark(1, 2.0, +1.5),
    ]
    d.observe(Pose2D(0.0, 0.0, 0.0), landmarks)
    for x in np.linspace(0.0, 20.0, 100):
        d.observe(Pose2D(float(x), 0.0, 0.0), landmarks)
    for x in np.linspace(20.0, -5.0, 100):
        if d.observe(Pose2D(float(x), 0.0, 0.0), landmarks):
            break
    assert d.state == "CROSSED"
    d.reset_for_next_lap()
    assert d.state == "APPROACHING"

"""Tests for the cross-colour DA gate (#269).

The colour gate used to be a hard bucket — observations of colour X
could only match landmarks of colour X. That broke whenever the
upstream body_y classifier flipped a cone's colour mid-run (e.g. on
sharp yaw rotations through hairpins): same physical cone got a new
tag, DA could not bridge the gap, every observation in the scan came
back as "new", and the cascade-detector zeroed all associations.

The new contract is:
  - Same-colour pairs use the full DISTANCE_GATE_M (2 m).
  - Cross-colour pairs use a tighter CROSS_COLOR_DISTANCE_GATE_M (1 m).
  - Mahalanobis χ² applies to both.

These tests pin both halves: same-colour matches at 1.5 m still work,
cross-colour matches at 0 m work (the recovery case), and cross-colour
matches across the corridor (3 m blue-yellow spacing) get rejected.
"""
from __future__ import annotations

import numpy as np

from cone_slam.color_classifier import ConeColor
from cone_slam.data_association import (
    CROSS_COLOR_DISTANCE_GATE_M,
    DISTANCE_GATE_M,
    Observation,
    associate,
)
from cone_slam.landmark_db import LandmarkDb


def _car_at_origin_facing_x() -> tuple[float, float, float]:
    return (0.0, 0.0, 0.0)


def test_same_colour_match_at_full_gate() -> None:
    """Sanity: same-colour obs at ~1.5 m offset still matches."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(ConeColor.YELLOW, np.array([5.0, 0.0, 0.0]), step=0)

    # Obs 1.5 m off — within DISTANCE_GATE_M=2 m for same-colour.
    obs = [Observation(body_x=5.0, body_y=1.5, height=0.3,
                       color=ConeColor.YELLOW)]
    matches = associate(obs, *pose, db)
    assert matches[0].landmark_id == 0


def test_cross_colour_flicker_recovers() -> None:
    """The motivating regression: cone created as BLUE in frame N,
    re-observed as YELLOW in frame N+1 because yaw rotation flipped
    the body_y classifier output. Offset is essentially zero, so the
    cross-colour gate (1 m) admits the match."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(ConeColor.BLUE, np.array([5.0, 0.5, 0.0]), step=0)

    # Same physical cone, now tagged YELLOW. Same body coords.
    obs = [Observation(body_x=5.0, body_y=0.5, height=0.3,
                       color=ConeColor.YELLOW)]
    matches = associate(obs, *pose, db)
    assert matches[0].landmark_id == 0, (
        "YELLOW obs at the BLUE landmark's exact position must "
        "associate via the cross-colour gate"
    )


def test_cross_corridor_cones_dont_cross_match() -> None:
    """Two physically distinct cones, opposite colours, on opposite
    sides of a 3 m corridor. The cross-colour gate (1 m) must reject
    any cross-corridor match — that was the failure mode of the fully
    colour-blind variant."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(ConeColor.BLUE,   np.array([5.0, +1.5, 0.0]), step=0)
    db.create(ConeColor.YELLOW, np.array([5.0, -1.5, 0.0]), step=0)

    # YELLOW obs at the BLUE landmark's position. With the same-colour
    # gate (2 m) it would also reach the actual YELLOW landmark 3 m
    # away, but the cross-colour gate (1 m) means it can ONLY associate
    # with the BLUE landmark right under it (offset = 0, cross-colour).
    obs = [Observation(body_x=5.0, body_y=+1.5, height=0.3,
                       color=ConeColor.YELLOW)]
    matches = associate(obs, *pose, db)
    assert matches[0].landmark_id == 0, (
        "YELLOW obs at BLUE landmark's xy must associate with that "
        "BLUE landmark (cross-colour, 0 m offset), not the YELLOW "
        "landmark 3 m across the corridor"
    )


def test_cross_colour_outside_tight_gate_rejected() -> None:
    """Cross-colour obs at 1.5 m offset — within same-colour gate
    (2 m) but outside cross-colour gate (1 m). Must be rejected, no
    new landmark inferred."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(ConeColor.YELLOW, np.array([5.0, 0.0, 0.0]), step=0)

    # BLUE obs 1.5 m from a YELLOW landmark — cross-colour, outside
    # the 1 m gate.
    obs = [Observation(body_x=5.0, body_y=1.5, height=0.3,
                       color=ConeColor.BLUE)]
    matches = associate(obs, *pose, db)
    assert matches[0].landmark_id == -1, (
        f"BLUE obs 1.5 m from YELLOW landmark must NOT associate "
        f"(cross-colour gate=1 m), got id={matches[0].landmark_id}"
    )


def test_far_cone_creates_new_landmark() -> None:
    """A cone farther than DISTANCE_GATE_M from any landmark must be
    flagged as new (-1) regardless of colour."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(ConeColor.YELLOW, np.array([5.0, 0.0, 0.0]), step=0)

    far = DISTANCE_GATE_M + 1.0
    obs = [Observation(body_x=5.0, body_y=far, height=0.3,
                       color=ConeColor.YELLOW)]
    matches = associate(obs, *pose, db)
    assert matches[0].landmark_id == -1


def test_empty_db_marks_all_obs_as_new() -> None:
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    obs = [
        Observation(body_x=5.0, body_y=1.0, height=0.3, color=ConeColor.YELLOW),
        Observation(body_x=6.0, body_y=2.0, height=0.3, color=ConeColor.BLUE),
    ]
    matches = associate(obs, *pose, db)
    assert all(m.landmark_id == -1 for m in matches)


def test_empty_obs_returns_empty() -> None:
    db = LandmarkDb()
    db.create(ConeColor.YELLOW, np.array([5.0, 0.0, 0.0]), step=0)
    matches = associate([], *_car_at_origin_facing_x(), db)
    assert matches == []


def test_hungarian_prefers_same_colour_when_both_available() -> None:
    """Two landmarks within both gates: a YELLOW one at exact obs
    position, a BLUE one 0.5 m away. YELLOW obs must pick the YELLOW
    landmark (same-colour, 0 m). The Hungarian cost is Euclidean —
    closer same-colour wins."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(ConeColor.BLUE,   np.array([5.0, 0.5, 0.0]), step=0)  # id 0
    db.create(ConeColor.YELLOW, np.array([5.0, 0.0, 0.0]), step=0)  # id 1

    obs = [Observation(body_x=5.0, body_y=0.0, height=0.3,
                       color=ConeColor.YELLOW)]
    matches = associate(obs, *pose, db)
    assert matches[0].landmark_id == 1, (
        f"Expected match to YELLOW landmark (id=1, same-colour, 0 m), "
        f"got id={matches[0].landmark_id}"
    )


def test_cross_colour_constants_relationship() -> None:
    """Cross-colour gate must be tighter than same-colour gate."""
    assert CROSS_COLOR_DISTANCE_GATE_M < DISTANCE_GATE_M

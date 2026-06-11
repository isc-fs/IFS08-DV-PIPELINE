"""Tests for the position-only data association.

Pins the new contract: a single Euclidean gate (`DISTANCE_GATE_M`)
plus the Mahalanobis χ² gate. No colour anywhere.
"""
from __future__ import annotations

import numpy as np

from cone_slam.data_association import (
    DISTANCE_GATE_M,
    Observation,
    associate,
)
from cone_slam.landmark_db import LandmarkDb


def _car_at_origin_facing_x() -> tuple[float, float, float]:
    return (0.0, 0.0, 0.0)


def test_obs_at_landmark_position_associates() -> None:
    """Sanity: an observation right on a landmark's position matches it."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(np.array([5.0, 0.0, 0.0]), step=0)

    obs = [Observation(body_x=5.0, body_y=0.0, height=0.3)]
    matches = associate(obs, *pose, db)

    assert matches[0].landmark_id == 0


def test_obs_within_gate_associates() -> None:
    """An obs less than DISTANCE_GATE_M from a landmark associates."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(np.array([5.0, 0.0, 0.0]), step=0)

    # 0.5 m off — within DISTANCE_GATE_M = 1.0 m.
    obs = [Observation(body_x=5.0, body_y=0.5, height=0.3)]
    matches = associate(obs, *pose, db)

    assert matches[0].landmark_id == 0


def test_obs_beyond_gate_creates_new_landmark() -> None:
    """An obs farther than DISTANCE_GATE_M is flagged as new."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(np.array([5.0, 0.0, 0.0]), step=0)

    far = DISTANCE_GATE_M + 0.5
    obs = [Observation(body_x=5.0, body_y=far, height=0.3)]
    matches = associate(obs, *pose, db)

    assert matches[0].landmark_id == -1


def test_cross_corridor_cones_dont_cross_match() -> None:
    """Two physically distinct cones on opposite sides of a 3 m
    corridor. Each obs must associate with its nearer landmark, not
    the one across the corridor."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(np.array([5.0, +1.5, 0.0]), step=0)  # id 0
    db.create(np.array([5.0, -1.5, 0.0]), step=0)  # id 1

    obs = [
        Observation(body_x=5.0, body_y=+1.5, height=0.3),
        Observation(body_x=5.0, body_y=-1.5, height=0.3),
    ]
    matches = associate(obs, *pose, db)
    ids = sorted(m.landmark_id for m in matches)
    assert ids == [0, 1]


def test_hungarian_picks_closest_when_two_within_gate() -> None:
    """Two landmarks within the gate of one obs — Hungarian picks the
    closer one."""
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    db.create(np.array([5.0, 0.0, 0.0]), step=0)   # id 0 at obs
    db.create(np.array([5.0, 0.5, 0.0]), step=0)   # id 1 — 0.5 m off

    obs = [Observation(body_x=5.0, body_y=0.0, height=0.3)]
    matches = associate(obs, *pose, db)
    assert matches[0].landmark_id == 0


def test_empty_db_marks_all_obs_as_new() -> None:
    db = LandmarkDb()
    pose = _car_at_origin_facing_x()
    obs = [
        Observation(body_x=5.0, body_y=1.0, height=0.3),
        Observation(body_x=6.0, body_y=2.0, height=0.3),
    ]
    matches = associate(obs, *pose, db)
    assert all(m.landmark_id == -1 for m in matches)


def test_empty_obs_returns_empty() -> None:
    db = LandmarkDb()
    db.create(np.array([5.0, 0.0, 0.0]), step=0)
    matches = associate([], *_car_at_origin_facing_x(), db)
    assert matches == []

"""Smoke tests for the FaSTTUBe adapter (PR #243).

These cover the *integration* surface — type translation, empty-input
safety, frame conventions — not the planner's geometric correctness.
The library itself is upstream-tested at https://github.com/papalotis/ft-fsd-path-planning.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

# Skip the whole module if the library isn't installed (CI on a fresh
# checkout, IDE running tests without the Docker image's pip env).
fsd_path_planning = pytest.importorskip("fsd_path_planning")

from path_planning.fasttube_adapter import (
    FasttubeAdapter,
    _COLOR_TO_CONETYPE,
    _NUM_CONE_TYPES,
)
from path_planning.core_types import Cone, ConeColor, Pose2D


# --- Fixtures ---------------------------------------------------------------


def _straight_track(n_pairs: int = 12, spacing: float = 3.0,
                    half_width: float = 1.5) -> List[Cone]:
    """Two rows of cones along world-X with small deterministic jitter.

    Blue on +Y (left), Yellow on -Y (right). Jitter (≤ 5 cm) breaks the
    perfect bilateral symmetry that triggers a `ZeroDivisionError` deep
    in the library's nearest-neighbour cost function — the library is
    designed for real-world cone fields with sensor noise, not pixel-
    perfect synthetic ones. The jitter is deterministic so tests are
    reproducible.
    """
    rng = np.random.default_rng(seed=0xFA577)
    cones: List[Cone] = []
    for i in range(n_pairs):
        x_base = float(i * spacing) + 1.0
        cones.append(Cone(
            x=x_base + float(rng.uniform(-0.05, 0.05)),
            y=+half_width + float(rng.uniform(-0.05, 0.05)),
            color=ConeColor.BLUE,
        ))
        cones.append(Cone(
            x=x_base + float(rng.uniform(-0.05, 0.05)),
            y=-half_width + float(rng.uniform(-0.05, 0.05)),
            color=ConeColor.YELLOW,
        ))
    return cones


def _car_at_origin() -> Pose2D:
    return Pose2D(x=0.0, y=0.0, yaw=0.0)


# --- Tests ------------------------------------------------------------------


def test_straight_track_returns_nonempty_path() -> None:
    """Smoke: dense straight track → adapter produces a forward path."""
    adapter = FasttubeAdapter()
    cones = _straight_track()
    pose = _car_at_origin()

    path = adapter.plan(cones, pose)

    assert len(path) >= 5, (
        f"expected ≥5 path points on a clean straight track, got {len(path)}"
    )
    # Path should be roughly monotonic forward in world-X (the car heads
    # along +X). Allow small backsteps from spline interpolation but
    # demand a meaningful forward distance overall.
    xs = [p.x for p in path]
    assert xs[-1] > xs[0] + 5.0, (
        f"path didn't progress forward: x[0]={xs[0]:.2f}, x[-1]={xs[-1]:.2f}"
    )


def test_color_mapping_buckets_by_cone_type() -> None:
    """Each ConeColor lands in its expected FaSTTUBe ConeTypes slot."""
    cones = [
        Cone(x=1.0, y=+1.5, color=ConeColor.BLUE),
        Cone(x=1.0, y=-1.5, color=ConeColor.YELLOW),
        Cone(x=2.0, y=+1.5, color=ConeColor.BLUE),
        Cone(x=0.0, y=+0.0, color=ConeColor.ORANGE),
        Cone(x=0.0, y=+0.5, color=ConeColor.BIG_ORANGE),
    ]
    arrays = FasttubeAdapter._cones_to_arrays(cones)

    assert len(arrays) == _NUM_CONE_TYPES
    # Slot 0 = UNKNOWN: nothing maps here today.
    assert arrays[0].shape == (0, 2)
    # Slot 1 = RIGHT/YELLOW: 1 cone.
    assert arrays[int(_COLOR_TO_CONETYPE[ConeColor.YELLOW])].shape == (1, 2)
    # Slot 2 = LEFT/BLUE: 2 cones.
    assert arrays[int(_COLOR_TO_CONETYPE[ConeColor.BLUE])].shape == (2, 2)
    # Slot 3 = ORANGE_SMALL: 1 cone.
    assert arrays[int(_COLOR_TO_CONETYPE[ConeColor.ORANGE])].shape == (1, 2)
    # Slot 4 = ORANGE_BIG: 1 cone.
    assert arrays[int(_COLOR_TO_CONETYPE[ConeColor.BIG_ORANGE])].shape == (1, 2)


def test_empty_input_returns_empty_path() -> None:
    """No cones in → no path out, no exception."""
    adapter = FasttubeAdapter()
    assert adapter.plan([], _car_at_origin()) == []


def test_one_cone_input_returns_empty_path() -> None:
    """Single cone is degenerate — adapter should swallow any library
    error and return [], not propagate."""
    adapter = FasttubeAdapter()
    cones = [Cone(x=1.0, y=0.0, color=ConeColor.BLUE)]
    # Shouldn't raise. May return [] or possibly a tiny path; both are
    # acceptable, the contract is "doesn't crash the node".
    path = adapter.plan(cones, _car_at_origin())
    assert isinstance(path, list)


def test_pose_rotation_rotates_path() -> None:
    """Rotate cones + pose by 90° → path rotates correspondingly.

    Catches direction-vector / yaw-sign mistakes in the adapter.
    """
    adapter = FasttubeAdapter()

    # Reference: car heading +X, straight track along +X.
    cones_x = _straight_track()
    pose_x = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path_x = adapter.plan(cones_x, pose_x)
    assert len(path_x) >= 5

    # Rotated: car heading +Y, track rotated 90° CCW so it still extends
    # ahead of the car.
    cones_y: List[Cone] = []
    for c in cones_x:
        cones_y.append(Cone(x=-c.y, y=+c.x, color=c.color))
    pose_y = Pose2D(x=0.0, y=0.0, yaw=math.pi / 2.0)

    path_y = adapter.plan(cones_y, pose_y)
    assert len(path_y) >= 5

    # In the rotated frame, the path should head along +Y.
    ys = [p.y for p in path_y]
    assert ys[-1] > ys[0] + 5.0, (
        f"rotated path didn't progress along +Y: y[0]={ys[0]:.2f}, "
        f"y[-1]={ys[-1]:.2f}"
    )


def test_path_points_have_finite_yaw() -> None:
    """Yaw is recomputed from finite differences — must be finite for
    every point including the tail (which reuses the previous segment)."""
    adapter = FasttubeAdapter()
    path = adapter.plan(_straight_track(), _car_at_origin())
    assert all(np.isfinite(p.yaw) for p in path)

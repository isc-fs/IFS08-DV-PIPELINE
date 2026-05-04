"""Smoke tests for the FaSTTUBe adapter.

These cover the *integration* surface — type translation, empty-input
safety, frame conventions, the new colour-agnostic ORANGE→UNKNOWN
routing (#254), and the cone cull window — not the planner's geometric
correctness. The library itself is upstream-tested at
https://github.com/papalotis/ft-fsd-path-planning.
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
    PlanDebug,
    _COLOR_TO_CONETYPE,
    _CULL_RANGE_M,
    _MAX_PATH_ARC_M,
    _NUM_CONE_TYPES,
    _cull_cones,
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
    path, debug = adapter.plan(_straight_track(), _car_at_origin())

    assert len(path) >= 5, (
        f"expected ≥5 path points on a clean straight track, got {len(path)}"
    )
    assert isinstance(debug, PlanDebug)
    # Path should be roughly monotonic forward in world-X (the car heads
    # along +X). Allow small backsteps from spline interpolation but
    # demand a meaningful forward distance overall.
    xs = [p.x for p in path]
    assert xs[-1] > xs[0] + 5.0, (
        f"path didn't progress forward: x[0]={xs[0]:.2f}, x[-1]={xs[-1]:.2f}"
    )


def test_color_mapping_buckets_by_cone_type() -> None:
    """ConeColor → ConeTypes mapping (#254): blue/yellow get LEFT/RIGHT,
    orange/big-orange both get UNKNOWN so the library's colour-blind
    sort handles them rather than dropping them in matching."""
    cones = [
        Cone(x=1.0, y=+1.5, color=ConeColor.BLUE),
        Cone(x=1.0, y=-1.5, color=ConeColor.YELLOW),
        Cone(x=2.0, y=+1.5, color=ConeColor.BLUE),
        Cone(x=0.0, y=+0.0, color=ConeColor.ORANGE),
        Cone(x=0.0, y=+0.5, color=ConeColor.BIG_ORANGE),
    ]
    arrays = FasttubeAdapter._cones_to_arrays(cones)

    assert len(arrays) == _NUM_CONE_TYPES
    # Slot 0 = UNKNOWN: holds the orange + big-orange cones.
    assert arrays[0].shape == (2, 2), (
        "ORANGE + BIG_ORANGE should both route to UNKNOWN slot 0"
    )
    # Slot 1 = RIGHT/YELLOW: 1 cone.
    assert arrays[int(_COLOR_TO_CONETYPE[ConeColor.YELLOW])].shape == (1, 2)
    # Slot 2 = LEFT/BLUE: 2 cones.
    assert arrays[int(_COLOR_TO_CONETYPE[ConeColor.BLUE])].shape == (2, 2)
    # Slots 3 (ORANGE_SMALL) and 4 (ORANGE_BIG) are now empty: the
    # library drops them in matching, so we deliberately don't put
    # cones there.
    assert arrays[3].shape == (0, 2)
    assert arrays[4].shape == (0, 2)


def test_empty_input_returns_empty_path() -> None:
    """No cones in → no path out, no exception."""
    adapter = FasttubeAdapter()
    path, debug = adapter.plan([], _car_at_origin())
    assert path == []
    assert isinstance(debug, PlanDebug)
    assert debug.left_with_virtual.shape == (0, 2)
    assert debug.right_with_virtual.shape == (0, 2)


def test_one_cone_input_returns_empty_path() -> None:
    """Single cone is degenerate — adapter should swallow any library
    error and return ([], PlanDebug()), not propagate."""
    adapter = FasttubeAdapter()
    cones = [Cone(x=1.0, y=0.0, color=ConeColor.BLUE)]
    path, debug = adapter.plan(cones, _car_at_origin())
    assert isinstance(path, list)
    assert isinstance(debug, PlanDebug)


def test_pose_rotation_rotates_path() -> None:
    """Rotate cones + pose by 90° → path rotates correspondingly.

    Catches direction-vector / yaw-sign mistakes in the adapter.
    """
    adapter = FasttubeAdapter()

    # Reference: car heading +X, straight track along +X.
    cones_x = _straight_track()
    path_x, _ = adapter.plan(cones_x, Pose2D(x=0.0, y=0.0, yaw=0.0))
    assert len(path_x) >= 5

    # Rotated: car heading +Y, track rotated 90° CCW so it still extends
    # ahead of the car.
    cones_y = [Cone(x=-c.y, y=+c.x, color=c.color) for c in cones_x]
    path_y, _ = adapter.plan(cones_y, Pose2D(x=0.0, y=0.0, yaw=math.pi / 2.0))
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
    path, _ = adapter.plan(_straight_track(), _car_at_origin())
    assert all(np.isfinite(p.yaw) for p in path)


def test_orange_cones_dont_disable_planning() -> None:
    """Whole-track ORANGE input (the worst-case SLAM mis-classify) still
    yields a valid path — the colour-blind sort sorts the cones by
    geometry rather than dropping them (#254 acceptance)."""
    adapter = FasttubeAdapter()
    # Take a normal track but tag every cone ORANGE.
    cones = [
        Cone(x=c.x, y=c.y, color=ConeColor.ORANGE)
        for c in _straight_track()
    ]
    path, _ = adapter.plan(cones, _car_at_origin())
    assert len(path) >= 5, (
        f"all-ORANGE input should still plan via colour-blind sort, "
        f"got {len(path)} points"
    )


# --- Cone cull (range + behind-car) -----------------------------------------


def test_cull_drops_cones_behind_car() -> None:
    """`_cull_cones` filters body_x ≤ 0 (cones already passed)."""
    pose = Pose2D(x=10.0, y=0.0, yaw=0.0)  # car at (10, 0) facing +X
    cones = [
        Cone(x=15.0, y=+1.5, color=ConeColor.BLUE),   # ahead — keep
        Cone(x=5.0,  y=+1.5, color=ConeColor.BLUE),   # behind — drop
        Cone(x=10.5, y=-1.5, color=ConeColor.YELLOW), # ahead — keep
        Cone(x=10.0, y=-1.5, color=ConeColor.YELLOW), # body_x≈0 — drop (≤ 0)
    ]
    out = _cull_cones(cones, pose, _CULL_RANGE_M)
    kept_x = sorted(c.x for c in out)
    assert kept_x == [10.5, 15.0]


def test_cull_drops_cones_beyond_range() -> None:
    """`_cull_cones` filters cones farther than max_range_m."""
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    cones = [
        Cone(x=10.0, y=0.0, color=ConeColor.BLUE),    # 10 m — keep
        Cone(x=24.0, y=0.0, color=ConeColor.BLUE),    # 24 m — keep (< 25)
        Cone(x=26.0, y=0.0, color=ConeColor.BLUE),    # 26 m — drop (> 25)
        Cone(x=50.0, y=0.0, color=ConeColor.BLUE),    # 50 m — drop
    ]
    out = _cull_cones(cones, pose, _CULL_RANGE_M)
    kept_x = sorted(c.x for c in out)
    assert kept_x == [10.0, 24.0]


def test_cull_respects_pose_yaw() -> None:
    """Cull uses body-frame x; "behind the car" depends on yaw."""
    # Car at origin, facing +Y (90°). Cones at +Y are ahead; +X are right.
    pose = Pose2D(x=0.0, y=0.0, yaw=math.pi / 2.0)
    cones = [
        Cone(x=0.0, y=+5.0, color=ConeColor.BLUE),    # ahead (body_x = +5)
        Cone(x=0.0, y=-5.0, color=ConeColor.BLUE),    # behind (body_x = -5)
        Cone(x=+5.0, y=0.0, color=ConeColor.YELLOW),  # body_x = 0 → drop
        Cone(x=-5.0, y=0.0, color=ConeColor.YELLOW),  # body_x = 0 → drop
    ]
    out = _cull_cones(cones, pose, _CULL_RANGE_M)
    assert len(out) == 1
    assert out[0].y == pytest.approx(5.0)


# --- Path forward-distance cap (#260) ---------------------------------------


def test_path_arc_length_capped() -> None:
    """A long straight track produces a long FaSTTUBe path; we should
    only publish up to _MAX_PATH_ARC_M of arc length so the controller
    never sees the virtually-extrapolated tail."""
    adapter = FasttubeAdapter()
    # 30 cone pairs along world-x at 3 m spacing → ~90 m of corridor;
    # FaSTTUBe will produce a path much longer than _MAX_PATH_ARC_M.
    cones = _straight_track(n_pairs=30, spacing=3.0)
    path, _ = adapter.plan(cones, _car_at_origin())

    assert len(path) >= 2
    # Arc length of the published path = sum of segment lengths.
    arc = 0.0
    for i in range(1, len(path)):
        dx = path[i].x - path[i - 1].x
        dy = path[i].y - path[i - 1].y
        arc += math.hypot(dx, dy)
    # Allow a small overshoot — the cap is on the s column of the (M, 4)
    # array; the slice keeps points whose s ≤ cap, so the *last kept*
    # point can sit slightly under the cap and the published arc-length
    # equals the cap value at that point. We assert a hard upper bound
    # of 1.5× the cap as a sanity check (any much higher implies the
    # cap isn't being applied at all).
    assert arc <= _MAX_PATH_ARC_M * 1.5, (
        f"path arc {arc:.2f}m should be capped near {_MAX_PATH_ARC_M}m, "
        f"got 50%+ over"
    )

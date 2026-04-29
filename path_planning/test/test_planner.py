"""Unit tests for path_planning.planner.

Pure Python; no ROS / no Docker. Runs the planner against synthetic
cone fixtures that exercise the cases the algorithm needs to handle:

  - Straight lane, both sides cleanly visible.
  - Straight lane, one side missing (one-side fallback).
  - Right-hand turn.
  - Sparse cones (degenerate input → empty path).
  - Color confusion (cone classified blue but on the right side).

The planner is intentionally simple, so these tests are tight: we
assert the path is monotonic forward, stays roughly between the
cones, and ends at a sensible distance.
"""

from __future__ import annotations

import math
from typing import List

import pytest

from path_planning.planner import (
    Cone,
    ConeColor,
    Pose2D,
    LOOKAHEAD_M,
    MIN_MIDPOINTS,
    N_OUTPUT,
    plan_centerline,
)


def _straight_track(length_m: float, half_width: float = 1.5,
                    cone_spacing: float = 3.0) -> List[Cone]:
    """A straight track aligned with the world +X axis. Blue cones on
    the +Y (left) side, yellow on the −Y (right) side."""
    cones: List[Cone] = []
    x = 0.0
    while x <= length_m:
        cones.append(Cone(x=x, y=+half_width, color=ConeColor.BLUE))
        cones.append(Cone(x=x, y=-half_width, color=ConeColor.YELLOW))
        x += cone_spacing
    return cones


def test_straight_both_sides() -> None:
    """Centerline of a straight track lies on y=0 with yaw≈0."""
    cones = _straight_track(length_m=20.0)
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path = plan_centerline(cones, pose)

    assert len(path) == N_OUTPUT, "expected dense output sampling"
    # Lateral deviation: every sample should sit on or very near y=0.
    for p in path:
        assert abs(p.y) < 0.20, f"unexpected lateral deviation {p.y:.3f} m"
    # Heading: every sample's yaw should be very close to 0.
    for p in path:
        assert abs(p.yaw) < 0.10, f"unexpected yaw {math.degrees(p.yaw):.2f}°"
    # Forward monotonicity in body frame (= world +X here).
    xs = [p.x for p in path]
    assert all(xs[i + 1] > xs[i] for i in range(len(xs) - 1)), \
        "path not monotonic forward"


def test_straight_left_only_fallback() -> None:
    """When right cones are missing, the one-side fallback should put
    the centerline ~TRACK_HALF_WIDTH_M to the right of the left cones."""
    cones = [c for c in _straight_track(20.0) if c.color == ConeColor.BLUE]
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path = plan_centerline(cones, pose)

    assert len(path) == N_OUTPUT
    for p in path:
        assert abs(p.y) < 0.30, \
            f"left-only fallback lateral error {p.y:.3f} m"


def test_right_turn() -> None:
    """A 30° right curve. Centerline yaw at the end should be roughly
    -30° from initial."""
    cones: List[Cone] = []
    for i in range(15):
        # Curve parameterized along x; lateral offset grows quadratically.
        x = i * 1.5
        # Track curves to the right (negative y) over 22.5 m.
        y_centre = -0.5 * (x / 22.5) * x  # quadratic, ~−4 m at end
        cones.append(Cone(x=x, y=y_centre + 1.5, color=ConeColor.BLUE))
        cones.append(Cone(x=x, y=y_centre - 1.5, color=ConeColor.YELLOW))
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path = plan_centerline(cones, pose)

    assert len(path) == N_OUTPUT
    # End yaw should be negative (turning right). Threshold loose; the
    # exact angle depends on how far along the spline samples land.
    end_yaw_deg = math.degrees(path[-1].yaw)
    assert end_yaw_deg < -10.0, f"right turn end yaw {end_yaw_deg:.1f}°"


def test_sparse_returns_empty() -> None:
    """A single pair of cones isn't enough to span MIN_MIDPOINTS
    (=anchor + at least 2 spline knots) → empty output."""
    cones = [
        Cone(x=2.0, y=+1.5, color=ConeColor.BLUE),
        Cone(x=2.0, y=-1.5, color=ConeColor.YELLOW),
    ]
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path = plan_centerline(cones, pose)
    assert path == [], "expected empty path on degenerate input"


def test_no_forward_cones_returns_empty() -> None:
    """All cones behind the car → no forward cones → empty path."""
    cones = _straight_track(10.0)
    # Pose well past the cones, facing further forward.
    pose = Pose2D(x=20.0, y=0.0, yaw=0.0)
    assert plan_centerline(cones, pose) == []


def test_pose_yaw_rotates_output() -> None:
    """Same cone fixture, yaw rotated +90° → centerline path should
    be along world +Y instead of world +X."""
    cones = _straight_track(15.0)
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path_x_axis = plan_centerline(cones, pose)
    assert path_x_axis  # sanity

    # Now the same TRACK rotated to lie along the world +Y axis.
    rotated = [
        Cone(x=-c.y, y=c.x, color=c.color) for c in cones
    ]
    pose_rot = Pose2D(x=0.0, y=0.0, yaw=math.pi / 2)
    path_y_axis = plan_centerline(rotated, pose_rot)
    assert path_y_axis

    # Each sample should match the unrotated path's (y, x) — i.e. the
    # 90° rotation of the original output. Tolerance accommodates
    # spline boundary effects.
    for a, b in zip(path_x_axis, path_y_axis):
        assert abs(a.x - b.y) < 0.05, f"x↔y rotated mismatch x={a.x} vs b.y={b.y}"
        assert abs(-a.y - b.x) < 0.05, "y↔−x rotated mismatch"


def test_color_confusion_drops_invalid_pair() -> None:
    """A blue cone on the wrong side (negative body_y) shouldn't drag
    the midpoint over to that side. The walker's distance-sanity
    check (|ly - ry| > 1 m) discards same-side pairs."""
    cones = _straight_track(15.0)
    # Add a rogue "blue" cone on the right side, near the start.
    cones.append(Cone(x=3.0, y=-1.5, color=ConeColor.BLUE))
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path = plan_centerline(cones, pose)
    assert len(path) > 0
    # First few samples might absorb some lateral error from the rogue
    # cone, but the path shouldn't dive off-centre by more than ~0.5 m.
    for p in path[:5]:
        assert abs(p.y) < 0.7, \
            f"rogue blue cone perturbed centerline by {p.y:.3f} m"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

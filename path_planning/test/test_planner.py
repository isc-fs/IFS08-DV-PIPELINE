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
import random
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
    plan_centerline_with_debug,
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


def test_straight_left_only_returns_empty() -> None:
    """When only one side of cones is visible, Delaunay can't form
    non-degenerate triangles. The planner returns empty rather than
    fabricating a path — the autonomy must not drive when it can't
    see both sides of the corridor."""
    cones = [c for c in _straight_track(20.0) if c.color == ConeColor.BLUE]
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path = plan_centerline(cones, pose)
    assert path == [], "Delaunay refuses collinear-only input"


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
    """A single pair of cones can't be Delaunay-triangulated (need
    ≥3 non-collinear points). The planner returns empty rather than
    fabricating a path — the autonomy must not drive when it doesn't
    have enough information to plan."""
    cones = [
        Cone(x=2.0, y=+1.5, color=ConeColor.BLUE),
        Cone(x=2.0, y=-1.5, color=ConeColor.YELLOW),
    ]
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path = plan_centerline(cones, pose)
    assert path == [], "Delaunay needs ≥3 non-collinear cones"


def test_no_forward_cones_returns_empty() -> None:
    """All cones behind the car → no forward cones → empty path.
    The autonomy must not drive when it has no forward visibility."""
    cones = _straight_track(10.0)
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


def _tight_corner_track(
    approach_len: float = 6.0,
    centerline_radius: float = 4.5,
    track_width: float = 3.0,
    inside_chord: float = 2.5,
    sweep_deg: float = 90.0,
) -> List[Cone]:
    """Build a synthetic tight 90°-by-default right turn.

    Geometry:
      - Straight approach along +X for `approach_len` metres at half_width
        cone spacing 1.5 m (matches FS straight-track conventions).
      - Then a circular arc bending right with centerline radius
        `centerline_radius` (FS minimum is ~4.5 m per TI 3.1.x).
      - Inside (yellow) cones at `centerline_radius - track_width / 2`,
        outside (blue) cones at `centerline_radius + track_width / 2`.
      - Inside-arc cone spacing is `inside_chord` m. Outside cones
        share the same angular indices as inside (so each blue/yellow
        pair forms a centerline crossing) — outside chord is therefore
        wider than inside, matching the real-track geometry.
    """
    half_width = track_width / 2.0

    cones: List[Cone] = []

    # Straight approach: +X heading, ±half_width.
    x = 0.0
    while x <= approach_len:
        cones.append(Cone(x=x, y=+half_width, color=ConeColor.BLUE))
        cones.append(Cone(x=x, y=-half_width, color=ConeColor.YELLOW))
        x += 1.5

    # Circle centre is `centerline_radius` to the right of the corner
    # entry (which sits at (approach_len, 0)). Right turn → centre at
    # (approach_len, -R).
    cx = approach_len
    cy = -centerline_radius

    inner_radius = centerline_radius - half_width
    outer_radius = centerline_radius + half_width

    # Step in arc-angle so that the inside-arc chord between adjacent
    # cones is `inside_chord`. theta_step = 2*arcsin(chord / 2R).
    theta_step = 2.0 * math.asin(min(1.0, inside_chord / (2.0 * inner_radius)))
    sweep = math.radians(sweep_deg)
    n_steps = max(2, int(round(sweep / theta_step)))

    # Sweep starts at theta = pi/2 (corner entry, due-north of centre)
    # and decreases — that traces a clockwise arc, i.e. a right turn.
    for i in range(n_steps + 1):
        theta = math.pi / 2.0 - i * (sweep / n_steps)
        c, s = math.cos(theta), math.sin(theta)
        cones.append(Cone(
            x=cx + outer_radius * c,
            y=cy + outer_radius * s,
            color=ConeColor.BLUE,
        ))
        cones.append(Cone(
            x=cx + inner_radius * c,
            y=cy + inner_radius * s,
            color=ConeColor.YELLOW,
        ))

    return cones


def test_tight_corner_45_deg() -> None:
    """A 45° right turn at FS-minimum radius. The Delaunay+search
    planner should find a path through it without dropping out — at
    45° the heading change is moderate and any planner that handles
    a generic curve has to handle this.

    Acceptance is intentionally loose: we just want the path to be
    non-empty and to make some progress around the corner. The
    `test_tight_corner_90_deg` case below is the harder one.
    """
    cones = _tight_corner_track(sweep_deg=45.0)
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path, debug = plan_centerline_with_debug(cones, pose)
    assert path, (
        f"45° tight corner returned empty path. "
        f"rejections={debug.rejections} "
        f"selected={0 if debug.selected_midpoints is None else len(debug.selected_midpoints)} "
        f"candidates={0 if debug.candidate_midpoints is None else len(debug.candidate_midpoints)}"
    )


def test_tight_corner_90_deg() -> None:
    """A full 90° right turn at FS-minimum radius. The hard case.
    Pre-#180-fix this completed only ~30° of sweep; with the
    same-color-edge filter the path now traces the full corner.
    """
    cones = _tight_corner_track(sweep_deg=90.0)
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    path, debug = plan_centerline_with_debug(cones, pose)
    assert path, (
        f"90° tight corner returned empty path. "
        f"rejections={debug.rejections} "
        f"selected={0 if debug.selected_midpoints is None else len(debug.selected_midpoints)} "
        f"candidates={0 if debug.candidate_midpoints is None else len(debug.candidate_midpoints)}"
    )
    end_yaw_deg = math.degrees(path[-1].yaw)
    # Demand most of the corner: −60° on a 90° sweep is acceptable,
    # leaving ~30° margin for spline-end attenuation. Pre-fix the
    # planner stopped at ≈ −31°.
    assert end_yaw_deg < -60.0, (
        f"90° tight corner end yaw {end_yaw_deg:.1f}° — planner "
        f"didn't progress around the corner. "
        f"rejections={debug.rejections}"
    )


def test_tight_corner_with_noise_doesnt_invert() -> None:
    """The pre-#180-fix planner could pick a same-color cross-corner
    Delaunay edge as a midpoint candidate; under realistic position
    noise (10–20 cm sigma) this midpoint sometimes won the heading-
    biased score and sent the path the *wrong way* through the
    corner (e.g. end_yaw +7.6° on a right turn that should reach
    −80°+). The same-color-edge filter in _build_edges closes this
    failure mode at source.

    Sweep noise/seed combinations that previously produced an
    inverted path; assert each now completes the right turn.
    """
    failures = []
    # Seeds 0..3 at three noise levels chosen to span the regime
    # where the original failure was reproducible (noise ≥ 5 cm).
    for noise in (0.05, 0.10, 0.15, 0.20):
        for seed in range(4):
            rng = random.Random(seed)
            cones = []
            half = 1.5
            x = 0.0
            while x <= 6.0:
                cones.append(Cone(x=x + rng.gauss(0, noise),
                                  y=+half + rng.gauss(0, noise),
                                  color=ConeColor.BLUE))
                cones.append(Cone(x=x + rng.gauss(0, noise),
                                  y=-half + rng.gauss(0, noise),
                                  color=ConeColor.YELLOW))
                x += 1.5
            cx, cy, R = 6.0, -4.5, 4.5
            inner_r, outer_r = R - half, R + half
            theta_step = 2.0 * math.asin(min(1.0, 2.5 / (2.0 * inner_r)))
            n_steps = max(2, int(round((math.pi / 2.0) / theta_step)))
            for i in range(n_steps + 1):
                theta = math.pi / 2.0 - i * (math.pi / 2.0 / n_steps)
                c, s = math.cos(theta), math.sin(theta)
                cones.append(Cone(
                    x=cx + outer_r * c + rng.gauss(0, noise),
                    y=cy + outer_r * s + rng.gauss(0, noise),
                    color=ConeColor.BLUE,
                ))
                cones.append(Cone(
                    x=cx + inner_r * c + rng.gauss(0, noise),
                    y=cy + inner_r * s + rng.gauss(0, noise),
                    color=ConeColor.YELLOW,
                ))
            pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
            path, debug = plan_centerline_with_debug(cones, pose)
            if not path:
                failures.append(
                    f"noise={noise} seed={seed}: empty path "
                    f"rejections={debug.rejections}"
                )
                continue
            end_yaw_deg = math.degrees(path[-1].yaw)
            # Right turn must NOT invert: end yaw must be negative.
            # Some seeds produce an honest dead-end at ~−30°; that's
            # acceptable for this regression test (it's not the bug
            # we're fixing here). The bug we're fixing is end yaw
            # being *positive* on a right turn — wrong-side selection.
            if end_yaw_deg > -5.0:
                failures.append(
                    f"noise={noise} seed={seed}: end_yaw={end_yaw_deg:+.1f}° "
                    f"— path inverted on a right turn. "
                    f"rejections={debug.rejections}"
                )
    assert not failures, "Wrong-direction failures:\n  " + "\n  ".join(failures)


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

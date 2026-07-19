"""Exhaustive tests for the deterministic skidpad reference (pure, ROS-free).

Skidpad scoring is lap-time on a fixed track, so the geometry MUST be exactly
the FS-Rules figure-eight — a wrong radius or a backward lap is a DNF. These
tests pin: the rule dimensions, every circle sample's radius, the crossing
returns, arc-length monotonicity, the forward-only window (no self-crossing in
view), and the monotone progress tracker (can't snap back onto an earlier lap).
"""
from __future__ import annotations

import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from skidpad.skidpad_reference import (  # noqa: E402
    CENTER_DISTANCE_M,
    INNER_DIAMETER_M,
    OUTER_DIAMETER_M,
    ReferencePath,
    SkidpadGeometry,
    SkidpadProgress,
    build_reference,
)

R = 9.125  # lane centreline radius from the rules


# --------------------------------------------------------------- geometry

def test_rule_dimensions_are_exact():
    """The dimensions are the rule (D4.1.2), not tuning — pin them."""
    assert INNER_DIAMETER_M == 15.25
    assert OUTER_DIAMETER_M == 21.25
    assert CENTER_DISTANCE_M == 18.25


def test_lane_radius_is_the_mean_and_equals_half_center_distance():
    g = SkidpadGeometry()
    assert g.lane_radius_m == pytest.approx(9.125)
    # The self-consistency the whole model relies on:
    assert g.lane_radius_m == pytest.approx(g.center_distance_m / 2.0)


def test_inconsistent_geometry_raises():
    with pytest.raises(ValueError, match="not meet at the crossing"):
        SkidpadGeometry(center_distance_m=20.0)  # no longer 2*lane_radius


def test_bad_lap_count_raises():
    with pytest.raises(ValueError):
        SkidpadGeometry(laps_per_side=0)


# ------------------------------------------------------- path construction

def _ref(**kw):
    return build_reference(SkidpadGeometry(**kw))


def test_total_length_matches_the_closed_form():
    g = SkidpadGeometry()
    ref = build_reference(g)
    # entry + 2*right + 2*left circles + exit = entry + 4*(2πR) + exit
    expected = g.entry_len_m + 2 * g.laps_per_side * (2 * math.pi * R) + g.exit_len_m
    assert ref.total_length == pytest.approx(expected, abs=0.05)


def test_starts_at_spawn_heading_plus_x():
    ref = _ref()
    p0 = ref.points[0]
    assert (p0.x, p0.y, p0.s) == pytest.approx((0.0, 0.0, 0.0))
    assert p0.yaw == pytest.approx(0.0)


def test_arc_length_is_strictly_monotone():
    ref = _ref()
    for a, b in zip(ref.points, ref.points[1:]):
        assert b.s > a.s, f"s not increasing at ({a.s}, {b.s})"


def test_every_right_circle_sample_is_on_the_right_circle():
    g = SkidpadGeometry()
    ref = build_reference(g)
    cx, cy = g.entry_len_m, -R          # right centre
    right = [p for p in ref.points if p.curvature < -1e-9]
    assert right, "no right-circle samples found"
    for p in right:
        assert math.hypot(p.x - cx, p.y - cy) == pytest.approx(R, abs=1e-6)
        assert p.curvature == pytest.approx(-1.0 / R)


def test_every_left_circle_sample_is_on_the_left_circle():
    g = SkidpadGeometry()
    ref = build_reference(g)
    cx, cy = g.entry_len_m, +R          # left centre
    left = [p for p in ref.points if p.curvature > 1e-9]
    assert left
    for p in left:
        assert math.hypot(p.x - cx, p.y - cy) == pytest.approx(R, abs=1e-6)
        assert p.curvature == pytest.approx(1.0 / R)


def test_straights_have_zero_curvature_and_y_zero():
    g = SkidpadGeometry()
    ref = build_reference(g)
    straight = [p for p in ref.points if abs(p.curvature) < 1e-9]
    for p in straight:
        assert abs(p.y) < 1e-9
        assert p.yaw == pytest.approx(0.0)


def test_right_is_clockwise_left_is_counterclockwise():
    """Sign of curvature encodes the mandated direction (D4.2): first the
    right circle (CW, −), then the left (CCW, +). Order matters."""
    ref = _ref()
    curvs = [p.curvature for p in ref.points]
    first_neg = next(i for i, c in enumerate(curvs) if c < -1e-9)
    first_pos = next(i for i, c in enumerate(curvs) if c > 1e-9)
    assert first_neg < first_pos, "right (CW) circle must come before left (CCW)"


def test_returns_to_the_crossing_each_lap():
    """Every circle lap must pass back through the crossing (entry_len, 0) —
    the figure-eight's shared point. Count near-crossing returns."""
    g = SkidpadGeometry()
    ref = build_reference(g)
    cx = g.entry_len_m
    # samples within 5 cm of the crossing, de-duplicated into clusters
    near = [p.s for p in ref.points if math.hypot(p.x - cx, p.y) < 0.05]
    clusters = []
    for s in near:
        if not clusters or s - clusters[-1] > 1.0:
            clusters.append(s)
    # crossing visited at: entry-end, after each of 4 circle laps = 5 times
    # (entry arrival, R lap1, R lap2, L lap1, L lap2); exit starts there too.
    assert len(clusters) >= 5, f"expected >=5 crossing returns, got {len(clusters)}"


# ------------------------------------------------------------ sample_at

def test_sample_at_clamps_ends():
    ref = _ref()
    assert ref.sample_at(-5.0).s == 0.0
    assert ref.sample_at(ref.total_length + 5.0).s == pytest.approx(ref.total_length)


def test_sample_at_interpolates_on_a_straight():
    g = SkidpadGeometry(entry_len_m=10.0)
    ref = build_reference(g)
    p = ref.sample_at(4.0)           # 4 m into the entry straight
    assert p.x == pytest.approx(4.0, abs=1e-6)
    assert p.y == pytest.approx(0.0, abs=1e-9)
    assert p.s == pytest.approx(4.0)


def test_sample_at_stays_on_circle_between_samples():
    g = SkidpadGeometry()
    ref = build_reference(g)
    cx, cy = g.entry_len_m, -R
    s_mid = g.entry_len_m + math.pi * R   # ~half the first right lap
    p = ref.sample_at(s_mid)
    # interpolation of a chord sits just inside the circle — within a mm at 5cm spacing
    assert math.hypot(p.x - cx, p.y - cy) == pytest.approx(R, abs=2e-3)


# --------------------------------------------------------------- window

def test_window_starts_exactly_at_s0_and_is_forward_only():
    ref = _ref()
    s0 = ref.total_length / 2.0
    w = ref.window(s0, 12.0)
    assert w[0].s == pytest.approx(s0)
    assert all(p.s >= s0 - 1e-9 for p in w)
    assert w[-1].s == pytest.approx(min(s0 + 12.0, ref.total_length))
    # monotone within the window
    for a, b in zip(w, w[1:]):
        assert b.s >= a.s


def test_window_never_contains_the_self_crossing_twice():
    """The whole point of a forward window: at the crossing, the reference is
    only ever present once (the next lap), never the ambiguous 4-way star."""
    g = SkidpadGeometry()
    ref = build_reference(g)
    cx = g.entry_len_m
    # put s0 right at a crossing return (~end of first right lap)
    s0 = g.entry_len_m + 2 * math.pi * R - 0.01
    w = ref.window(s0, 8.0)
    near_crossing = [p for p in w if math.hypot(p.x - cx, p.y) < 0.10]
    # The correct "single pass" property: the near-crossing samples form ONE
    # contiguous arc-length cluster (the reference threads the crossing once in
    # this forward window), NOT the ambiguous 4-way star a full-path nearest-
    # point search would see. Contiguity, not count.
    assert near_crossing, "expected the crossing in view"
    s_span = near_crossing[-1].s - near_crossing[0].s
    assert s_span < 1.0, f"near-crossing samples span {s_span:.2f} m → >1 pass"


def test_window_clamps_at_path_end():
    ref = _ref()
    w = ref.window(ref.total_length - 1.0, 12.0)
    assert w[-1].s == pytest.approx(ref.total_length)


# -------------------------------------------------------- SkidpadProgress

def _drive_along(prog: ReferencePath, progress: SkidpadProgress, step=0.25):
    """Simulate perfect driving: sample the reference, feed pose+distance."""
    s = 0.0
    while s < prog.total_length:
        p = prog.sample_at(s)
        progress.update(step, p.x, p.y)
        s += step


def test_progress_is_monotone_and_reaches_the_end():
    ref = _ref()
    prog = SkidpadProgress(ref)
    last = -1.0
    s = 0.0
    while s < ref.total_length:
        p = ref.sample_at(s)
        cur = prog.update(0.25, p.x, p.y)
        assert cur >= last - 1e-9, "progress went backward"
        last = cur
        s += 0.25
    assert prog.remaining() == pytest.approx(0.0, abs=0.5)


def test_progress_does_not_snap_back_at_the_crossing():
    """The critical property: when the car is AT the crossing on lap 2, the
    forward-only refine must NOT relocate s to the lap-1 crossing."""
    g = SkidpadGeometry()
    ref = build_reference(g)
    prog = SkidpadProgress(ref)
    # drive exactly to the end of the first right lap (a crossing)
    s_target = g.entry_len_m + 2 * math.pi * R
    s = 0.0
    while s < s_target:
        p = ref.sample_at(s)
        prog.update(0.25, p.x, p.y)
        s += 0.25
    s_before = prog.s
    # now the car is at the crossing (x=entry_len, y=0). Feed that exact pose
    # again — s must not jump backward to the earlier lap's crossing.
    cross = ref.sample_at(s_target)
    prog.update(0.0, cross.x, cross.y)
    assert prog.s >= s_before - 1e-6, "snapped back onto an earlier lap"


def test_progress_tolerates_lateral_drift():
    """A pose offset laterally from the reference must still advance s (the
    forward search finds the nearest forward point, not stall)."""
    ref = _ref()
    prog = SkidpadProgress(ref)
    s = 0.0
    while s < 20.0:
        p = ref.sample_at(s)
        # 0.3 m lateral drift off the path
        prog.update(0.25, p.x, p.y + 0.3)
        s += 0.25
    assert prog.s > 15.0, "drift stalled progress"


def test_is_finished_requires_end_and_standstill():
    ref = _ref()
    prog = SkidpadProgress(ref)
    prog.s = ref.total_length                # at the end
    assert not prog.is_finished(speed_mps=2.0)   # still moving → not finished
    assert prog.is_finished(speed_mps=0.0)       # stopped → finished


def test_is_finished_false_mid_path():
    ref = _ref()
    prog = SkidpadProgress(ref)
    prog.s = ref.total_length / 2.0
    assert not prog.is_finished(speed_mps=0.0)   # stopped mid-course ≠ finished


def test_three_laps_per_side_changes_length():
    two = _ref(laps_per_side=2).total_length
    three = _ref(laps_per_side=3).total_length
    assert three - two == pytest.approx(2 * (2 * math.pi * R), abs=0.05)

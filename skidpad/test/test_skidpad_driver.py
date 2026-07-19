"""Tests for the deterministic skidpad driver (pure, ROS-free).

The driver is the per-tick brain: pose+speed in, path window + finish out. These
pin the properties the controller and mission_control depend on: progress only
advances, the window is a forward non-self-intersecting slice that empties at the
end, and `finished` latches exactly once (at end + standstill) and never
un-latches on a stale pose.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from skidpad.skidpad_driver import SkidpadDriver  # noqa: E402
from skidpad.skidpad_reference import (  # noqa: E402
    SkidpadGeometry,
    build_reference,
)

R = 9.125


def _driver(**kw):
    return SkidpadDriver(build_reference(SkidpadGeometry()), **kw)


def _drive_reference(driver: SkidpadDriver, *, step_m=0.25, speed=2.0,
                     stop_at_end=True):
    """Feed the driver perfect on-reference poses from spawn to the end.
    Returns the list of DriverOutput. Optionally appends stopped poses at the
    end so the standstill finish gate can fire."""
    ref = driver.reference
    outs = []
    s = 0.0
    while s < ref.total_length:
        p = ref.sample_at(s)
        outs.append(driver.step(p.x, p.y, speed))
        s += step_m
    if stop_at_end:
        end = ref.sample_at(ref.total_length)
        for _ in range(3):                       # a few stopped ticks
            outs.append(driver.step(end.x, end.y, 0.0))
    return outs


# ------------------------------------------------------------- construction

def test_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        _driver(window_len_m=0.0)


def test_builds_reference_from_geometry_when_none_given():
    d = SkidpadDriver(geometry=SkidpadGeometry())
    assert d.reference.total_length > 0.0


# ----------------------------------------------------------------- progress

def test_first_step_advances_zero_but_anchors():
    d = _driver()
    out = d.step(0.0, 0.0, 0.0)
    assert out.progress_s == pytest.approx(0.0, abs=1e-6)
    assert not out.finished


def test_progress_advances_by_position_delta():
    d = _driver()
    d.step(0.0, 0.0, 2.0)          # anchor at spawn
    out = d.step(3.0, 0.0, 2.0)    # moved 3 m down the entry straight
    assert out.progress_s == pytest.approx(3.0, abs=0.1)


def test_progress_is_monotone_over_a_full_run():
    d = _driver()
    last = -1.0
    for out in _drive_reference(d, stop_at_end=False):
        assert out.progress_s >= last - 1e-9
        last = out.progress_s


def test_reset_returns_progress_to_spawn():
    d = _driver()
    _drive_reference(d)
    assert d.progress_s > 0.0
    d.reset()
    assert d.progress_s == 0.0
    out = d.step(0.0, 0.0, 0.0)
    assert out.progress_s == pytest.approx(0.0)
    assert not out.finished


# ------------------------------------------------------------------- window

def test_window_is_forward_and_bounded():
    d = _driver(window_len_m=8.0)
    d.step(0.0, 0.0, 2.0)
    out = d.step(20.0 - 8.0, 0.0, 2.0)   # some way along; still on a straight/circle
    assert out.path, "expected a non-empty window mid-course"
    s0 = out.path[0].s
    span = out.path[-1].s - s0
    assert span <= 8.0 + 1e-6
    for a, b in zip(out.path, out.path[1:]):
        assert b.s >= a.s                # monotone forward


def test_window_never_self_intersects_at_the_crossing():
    """Drive to a crossing return and assert the emitted window threads it once
    (single contiguous near-crossing cluster), not the 4-way star."""
    g = SkidpadGeometry()
    d = SkidpadDriver(build_reference(g), window_len_m=8.0)
    ref = d.reference
    cx = g.entry_len_m
    # advance progress to just before the end of the first right lap
    s_target = g.entry_len_m + 2 * math.pi * R - 0.05
    s = 0.0
    while s < s_target:
        p = ref.sample_at(s)
        out = d.step(p.x, p.y, 2.0)
        s += 0.25
    near = [pt for pt in out.path if math.hypot(pt.x - cx, pt.y) < 0.10]
    if near:
        assert near[-1].s - near[0].s < 1.0   # one pass only


def test_window_empties_at_the_end():
    d = _driver()
    outs = _drive_reference(d, stop_at_end=True)
    assert outs[-1].path == [], "window must be empty once past the end (coast)"


# -------------------------------------------------------------------- finish

def test_not_finished_mid_course_even_when_stopped():
    d = _driver()
    d.step(0.0, 0.0, 0.0)
    # stopped halfway down the entry straight — not at the end
    out = d.step(4.0, 0.0, 0.0)
    assert not out.finished


def test_not_finished_at_end_while_moving():
    d = _driver()
    outs = _drive_reference(d, speed=2.0, stop_at_end=False)
    # reached the end but never stopped → never finished
    assert not outs[-1].finished


def test_finishes_at_end_and_standstill():
    d = _driver()
    outs = _drive_reference(d, stop_at_end=True)
    assert outs[-1].finished


def test_finish_latches_and_survives_a_stale_backward_pose():
    d = _driver()
    _drive_reference(d, stop_at_end=True)
    assert d.step(d.reference.sample_at(d.reference.total_length).x,
                  d.reference.sample_at(d.reference.total_length).y, 0.0).finished
    # a stale pose from early in the run must NOT un-finish
    out = d.step(1.0, 0.0, 0.0)
    assert out.finished


def test_remaining_reaches_zero_at_the_end():
    d = _driver()
    outs = _drive_reference(d, stop_at_end=True)
    assert outs[-1].remaining_m == pytest.approx(0.0, abs=0.5)


def test_lateral_drift_still_advances_and_finishes():
    """Poses offset laterally from the reference must still progress and finish
    — the forward-refine finds the nearest forward point, not stall."""
    d = _driver()
    ref = d.reference
    s = 0.0
    out = None
    while s < ref.total_length:
        p = ref.sample_at(s)
        out = d.step(p.x, p.y + 0.25, 2.0)   # 25 cm off the path
        s += 0.25
    end = ref.sample_at(ref.total_length)
    for _ in range(3):
        out = d.step(end.x, end.y + 0.25, 0.0)
    assert out.finished

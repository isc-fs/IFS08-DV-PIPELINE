"""Pure Pursuit controller tests.

Focused on the curvature-adaptive lookahead added in #260 — straight
paths should still use the velocity-based Ld; tight-curve paths should
have Ld capped at β·R_local. The geometric Pure Pursuit math (κ from
chase target) is tested incidentally via the steering output sign.
"""
from __future__ import annotations

import math
from typing import List

import pytest

from control.controllers.pure_pursuit import (
    PurePursuit,
    _max_kappa_in_window,
)
from control.reference import ReferenceTrajectory
from control.state import VehicleState


# --- Fixtures ---------------------------------------------------------------


def _straight_path(length_m: float = 20.0, n: int = 41) -> ReferenceTrajectory:
    """Path along world-X from origin. Curvature = 0."""
    xs = [i * length_m / (n - 1) for i in range(n)]
    ys = [0.0] * n
    return ReferenceTrajectory.from_xy(xs, ys)


def _circular_path(radius_m: float, n: int = 41,
                   arc_rad: float = math.pi / 2) -> ReferenceTrajectory:
    """Counter-clockwise arc of given radius starting at origin tangent
    to +x. κ = 1/radius. Used to test the lookahead cap on tight bends."""
    xs: List[float] = []
    ys: List[float] = []
    for i in range(n):
        theta = arc_rad * i / (n - 1)
        # Centre at (0, radius); start at (0, 0) tangent to +x.
        xs.append(radius_m * math.sin(theta))
        ys.append(radius_m - radius_m * math.cos(theta))
    return ReferenceTrajectory.from_xy(xs, ys)


def _state_at_origin(speed: float = 5.0) -> VehicleState:
    return VehicleState(x=0.0, y=0.0, yaw=0.0, vx=speed, vy=0.0)


# --- max-kappa helper -------------------------------------------------------


def test_max_kappa_zero_on_straight() -> None:
    ref = _straight_path()
    assert _max_kappa_in_window(ref, 0, 10.0) == pytest.approx(0.0, abs=1e-6)


def test_max_kappa_matches_circle() -> None:
    """A 5 m radius circle should report κ ≈ 1/5 = 0.2 m⁻¹."""
    ref = _circular_path(radius_m=5.0)
    k = _max_kappa_in_window(ref, 0, 5.0)
    assert k == pytest.approx(0.2, rel=0.05)


def test_max_kappa_window_zero_returns_at_least_anchor() -> None:
    ref = _circular_path(radius_m=5.0)
    # Zero window — should still return |κ| at the anchor (the first
    # path point falls within s_anchor + 0).
    k = _max_kappa_in_window(ref, 0, 0.0)
    assert k > 0.0


# --- adaptive lookahead -----------------------------------------------------


def test_lookahead_uses_velocity_law_on_straight() -> None:
    """Straight path → R_local = ∞ → Ld is the unconstrained k·v + L_min."""
    pp = PurePursuit(lookahead_min=1.5, lookahead_k=0.5, lookahead_max=8.0,
                     lookahead_radius_factor=0.7)
    ref = _straight_path()
    # Drive the controller and cross-check the resulting steering. On a
    # straight ahead with car centred and heading +x, output is ≈ 0.
    steer = pp.compute(_state_at_origin(speed=5.0), ref)
    assert abs(steer) < 1e-3, (
        f"straight-path steering should be ~0, got {steer:+.4f}"
    )


def test_lookahead_capped_on_tight_curve() -> None:
    """Tight 4 m radius curve at 5 m/s.

    Without the cap: Ld = 1.5 + 0.5·5 = 4.0 m, equal to the radius —
    Pure Pursuit's chase target lands at the apex / past it, classic
    apex-cutting setup. With the β=0.7 cap: Ld = min(4.0, 0.7·4) = 2.8 m,
    chase target lands inside the curve. We can't directly read Ld from
    `compute()`, but we can verify the output via comparison: with the
    cap on, the output steering on the same setup is *larger* (more
    aggressive turn-in) than without, because the chase target is now
    closer to the car and at a sharper angle.
    """
    pp_capped = PurePursuit(lookahead_min=1.5, lookahead_k=0.5,
                            lookahead_max=8.0, lookahead_radius_factor=0.7)
    pp_uncapped = PurePursuit(lookahead_min=1.5, lookahead_k=0.5,
                              lookahead_max=8.0,
                              # 1.0 + tiny margin makes the cap inactive
                              # (Ld must be < R, but with factor 1.0 the
                              # cap equals R itself which the velocity
                              # law already produces at this speed).
                              lookahead_radius_factor=10.0)

    ref = _circular_path(radius_m=4.0)
    state = _state_at_origin(speed=5.0)

    steer_capped = pp_capped.compute(state, ref)
    steer_uncapped = pp_uncapped.compute(state, ref)

    # Both turn left (positive κ in our convention → positive steering
    # for left turn after the bicycle model). Capped one is more
    # aggressive because the chase target is closer.
    assert steer_capped > 0, f"capped steering should be left-turn (>0), got {steer_capped}"
    assert steer_uncapped > 0
    assert abs(steer_capped) >= abs(steer_uncapped) - 1e-6, (
        f"capped Ld should produce equal-or-stronger steering on a tight "
        f"curve: capped={steer_capped:+.4f}, uncapped={steer_uncapped:+.4f}"
    )


def test_lookahead_floors_at_lmin_on_extreme_curves() -> None:
    """A 1 m radius bend would push Ld → 0.7 m, below L_min=1.5. The
    floor must keep it at 1.5 — Pure Pursuit destabilises if the chase
    target collapses to the car."""
    pp = PurePursuit(lookahead_min=1.5, lookahead_k=0.5, lookahead_max=8.0,
                     lookahead_radius_factor=0.7)
    ref = _circular_path(radius_m=1.0)
    # Just check it doesn't blow up / NaN.
    steer = pp.compute(_state_at_origin(speed=5.0), ref)
    assert math.isfinite(steer)
    # Output should be saturated-ish (deep into the steering range).
    assert abs(steer) > 0.1


def test_empty_path_returns_zero() -> None:
    pp = PurePursuit()
    ref = ReferenceTrajectory()  # empty
    assert pp.compute(_state_at_origin(speed=5.0), ref) == 0.0

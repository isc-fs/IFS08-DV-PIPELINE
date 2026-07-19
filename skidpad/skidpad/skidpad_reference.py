"""Deterministic skidpad reference path — pure, ROS-free.

Skidpad is the one dynamic discipline whose track is FULLY defined by the rules
(FS-Rules 2026 D4.1), so there is nothing to perceive or map: we drive a
pre-computed figure-eight and track it with the EKF pose. This module builds
that reference and answers "where am I along it", with NO rclpy / numpy so every
branch is unit-testable off-node (mirrors cone_slam.lap_counter,
mission_control.reconcile).

## Geometry (FS-Rules D4.1.2)

  * inner circles Ø15.25 m  → inner radius 7.625 m
  * outer circles Ø21.25 m  → outer radius 10.625 m
  * the 3 m lane centreline is the mean → radius 9.125 m  (what we drive)
  * circle centres 18.25 m apart → each centre is 9.125 m from the crossing,
    exactly the lane radius, so both centreline circles pass through the
    crossing point. That is the figure-eight's self-intersection.

## Frame & procedure

REP-103 (x forward, y left). The path frame's origin is the **spawn**, heading
+x — the operator confirmed the entry pose is repeatable, so we anchor here and
never need a cone fix. Segments (D4.2 procedure: establish + timed on each side):

  entry straight → RIGHT circle ×2 (CW) → LEFT circle ×2 (CCW) → exit straight

Both circles pass through the crossing at (entry_len, 0); every lap returns
there heading +x. Right is driven clockwise (curvature −1/R), left
counter-clockwise (+1/R); straights are 0.

## Why a local WINDOW, not the whole path

The figure-eight crosses itself at the crossing — the same (x, y) occurs at four
different arc-lengths. A controller doing nearest-point tracking would be
ambiguous there. So progress is tracked as a monotone arc-length `s` (advanced
by distance travelled, refined by a *forward-only* pose projection), and the
controller is fed only `window(s, L)` — the next L metres — which never
self-intersects. This is exactly approach (b): fixed path, odometry drives
progress, lateral control corrects drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# FS-Rules 2026 D4.1.2 — the numbers are the rule, not tuning.
INNER_DIAMETER_M = 15.25
OUTER_DIAMETER_M = 21.25
CENTER_DISTANCE_M = 18.25


@dataclass(frozen=True)
class SkidpadGeometry:
    """Skidpad layout. Defaults are the FS-Rules D4.1.2 dimensions; only
    entry/exit lane lengths are venue/operator choices."""

    inner_diameter_m: float = INNER_DIAMETER_M
    outer_diameter_m: float = OUTER_DIAMETER_M
    center_distance_m: float = CENTER_DISTANCE_M
    entry_len_m: float = 8.0     # straight from spawn to the crossing
    exit_len_m: float = 15.0     # straight after the last lap (clear + decel)
    laps_per_side: int = 2       # D4.2: establish lap + timed lap, each side

    @property
    def lane_radius_m(self) -> float:
        """Centreline radius of the 3 m driving lane — what we actually drive."""
        return (self.inner_diameter_m + self.outer_diameter_m) / 4.0

    def __post_init__(self) -> None:
        # The centreline circles must pass through the crossing, or the
        # figure-eight the rules describe is not self-consistent with our
        # single-radius model. This catches a mistyped dimension.
        half_centers = self.center_distance_m / 2.0
        if abs(half_centers - self.lane_radius_m) > 1e-6:
            raise ValueError(
                f"inconsistent skidpad geometry: half centre-distance "
                f"{half_centers:.4f} != lane radius {self.lane_radius_m:.4f}; "
                f"the centreline circles would not meet at the crossing")
        if self.laps_per_side < 1:
            raise ValueError("laps_per_side must be >= 1")


@dataclass(frozen=True)
class PathPoint:
    """One reference sample. `s` is cumulative arc length from the spawn."""

    x: float
    y: float
    yaw: float        # heading of the tangent, rad
    curvature: float  # signed 1/R (left +, right −); 0 on straights
    s: float


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class ReferencePath:
    """Immutable arc-length reference for the whole skidpad run.

    Built once from a SkidpadGeometry; queried per control tick for the local
    window and for finish. Dense enough (default ~5 cm spacing) that linear
    interpolation between samples is sub-centimetre.
    """

    def __init__(self, points: list[PathPoint]) -> None:
        if len(points) < 2:
            raise ValueError("reference needs >= 2 points")
        self._pts = points
        self.total_length = points[-1].s

    @property
    def points(self) -> list[PathPoint]:
        return self._pts

    def sample_at(self, s: float) -> PathPoint:
        """Interpolate the reference at arc length `s` (clamped to the path)."""
        if s <= 0.0:
            return self._pts[0]
        if s >= self.total_length:
            return self._pts[-1]
        # Binary search for the segment containing s.
        lo, hi = 0, len(self._pts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._pts[mid].s <= s:
                lo = mid
            else:
                hi = mid
        a, b = self._pts[lo], self._pts[hi]
        span = b.s - a.s
        t = 0.0 if span <= 0.0 else (s - a.s) / span
        return PathPoint(
            x=a.x + t * (b.x - a.x),
            y=a.y + t * (b.y - a.y),
            yaw=a.yaw + t * _wrap_pi(b.yaw - a.yaw),
            curvature=a.curvature + t * (b.curvature - a.curvature),
            s=s,
        )

    def window(self, s0: float, length: float) -> list[PathPoint]:
        """The reference from `s0` forward by `length` metres.

        This is what the controller tracks — a strictly forward slice, so the
        figure-eight's self-crossing is never in view. Always starts exactly at
        s0 (interpolated) and ends at min(s0+length, total_length)."""
        s0 = max(0.0, min(s0, self.total_length))
        s_end = min(s0 + length, self.total_length)
        out = [self.sample_at(s0)]
        for p in self._pts:
            if p.s <= s0:
                continue
            if p.s >= s_end:
                break
            out.append(p)
        if out[-1].s < s_end:
            out.append(self.sample_at(s_end))
        return out


def build_reference(
    geom: SkidpadGeometry | None = None,
    *,
    point_spacing_m: float = 0.05,
) -> ReferencePath:
    """Construct the full skidpad reference: entry → R×N → L×N → exit."""
    g = geom or SkidpadGeometry()
    r = g.lane_radius_m
    cross_x = g.entry_len_m          # crossing is at (entry_len, 0)
    pts: list[PathPoint] = []
    s = 0.0

    def push(x: float, y: float, yaw: float, curv: float) -> None:
        nonlocal s
        if pts:
            s += math.hypot(x - pts[-1].x, y - pts[-1].y)
        pts.append(PathPoint(x, y, yaw, curv, s))

    # --- entry straight: spawn (0,0) → crossing (cross_x, 0), heading +x ------
    n_entry = max(1, round(g.entry_len_m / point_spacing_m))
    for i in range(n_entry + 1):
        push(g.entry_len_m * i / n_entry, 0.0, 0.0, 0.0)

    # per-circle angular step
    n_circle = max(8, round((2.0 * math.pi * r) / point_spacing_m))
    dphi = 2.0 * math.pi / n_circle

    # --- RIGHT circle ×N, clockwise. Centre (cross_x, -r). ------------------
    # Start φ = +π/2 (car at crossing, above the centre), φ decreasing.
    cr_x, cr_y = cross_x, -r
    for _lap in range(g.laps_per_side):
        for i in range(1, n_circle + 1):
            phi = math.pi / 2.0 - dphi * i
            x = cr_x + r * math.cos(phi)
            y = cr_y + r * math.sin(phi)
            # CW tangent (φ decreasing): heading of −dP/dφ.
            yaw = math.atan2(-math.cos(phi), math.sin(phi))
            push(x, y, yaw, -1.0 / r)

    # --- LEFT circle ×N, counter-clockwise. Centre (cross_x, +r). -----------
    # Start ψ = −π/2 (car at crossing, below the centre), ψ increasing.
    cl_x, cl_y = cross_x, r
    for _lap in range(g.laps_per_side):
        for i in range(1, n_circle + 1):
            psi = -math.pi / 2.0 + dphi * i
            x = cl_x + r * math.cos(psi)
            y = cl_y + r * math.sin(psi)
            # CCW tangent (ψ increasing): heading of +dP/dψ.
            yaw = math.atan2(math.cos(psi), -math.sin(psi))
            push(x, y, yaw, 1.0 / r)

    # --- exit straight: crossing → +x by exit_len, heading +x ---------------
    n_exit = max(1, round(g.exit_len_m / point_spacing_m))
    for i in range(1, n_exit + 1):
        push(cross_x + g.exit_len_m * i / n_exit, 0.0, 0.0, 0.0)

    return ReferencePath(pts)


class SkidpadProgress:
    """Monotone arc-length tracker along a ReferencePath.

    `s` only ever moves forward. Each tick you give it the distance travelled
    (from odometry) and the current pose; it advances `s` by the distance, then
    refines within a small FORWARD window by projecting the pose onto the
    reference — so lateral drift doesn't stall progress and the self-crossing
    can't snap `s` backward onto an earlier lap.
    """

    def __init__(
        self,
        path: ReferencePath,
        *,
        refine_window_m: float = 3.0,
        refine_step_m: float = 0.05,
    ) -> None:
        self.path = path
        self.s = 0.0
        self._refine_window = refine_window_m
        self._refine_step = refine_step_m

    def update(self, distance_travelled: float, pose_x: float,
               pose_y: float) -> float:
        """Advance by `distance_travelled` (>=0), then forward-refine on pose.
        Returns the new arc length `s`."""
        if distance_travelled > 0.0:
            self.s = min(self.s + distance_travelled, self.path.total_length)
        # Forward-only refinement: search [s, s+window] for the closest point,
        # never behind s (that would jump onto an earlier lap at the crossing).
        best_s, best_d2 = self.s, float("inf")
        s = self.s
        s_end = min(self.s + self._refine_window, self.path.total_length)
        while s <= s_end:
            p = self.path.sample_at(s)
            d2 = (p.x - pose_x) ** 2 + (p.y - pose_y) ** 2
            if d2 < best_d2:
                best_d2, best_s = d2, s
            s += self._refine_step
        self.s = best_s
        return self.s

    def remaining(self) -> float:
        return max(0.0, self.path.total_length - self.s)

    def is_finished(self, speed_mps: float, *, standstill_mps: float = 0.5,
                    finish_margin_m: float = 0.5) -> bool:
        """True once the car has reached the end of the reference AND stopped.
        Standstill gate mirrors LapCounter — AS Finished must be stationary."""
        return (self.remaining() <= finish_margin_m
                and abs(speed_mps) <= standstill_mps)

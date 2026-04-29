"""Centerline planner — forward-walking midpoint with cubic-spline smoothing.

Pure-Python, no ROS deps. Designed to be testable in isolation against
synthetic cone fixtures.

# Algorithm

Given a set of (color-tagged) cone positions in the world frame and a
car pose (x, y, yaw), compute a smooth centerline ahead of the car as
a sequence of (x, y, yaw) waypoints in the world frame.

  1. Project every cone into body frame (forward-X, left-Y).
  2. Drop cones outside a forward "corridor of interest"
     (BODY_X_MIN ≤ body_x ≤ LOOKAHEAD_M, |body_y| ≤ HALF_CORRIDOR_M).
     Cones behind the car or far to the side don't help.
  3. Sort the remaining LEFT and RIGHT cones by body_x ascending.
  4. Pair: walk forward in steps of STEP_M starting from BODY_X_MIN.
     At each target_x, find the closest LEFT and RIGHT cones in the
     window [target_x - STEP_M, target_x + STEP_M].
       - both found → midpoint at ( target_x, (Ly + Ry) / 2 )
       - only one found → offset from that cone by ±TRACK_HALF_WIDTH_M
       - neither found → stop walking; we ran out of cones.
  5. Always anchor the path at the car (body origin) as the first
     point. Without this anchor a Pure-Pursuit / Stanley controller
     looking for "the path point closest to me" can pick a midpoint
     several meters ahead and act on a stale lateral error.
  6. Cubic spline through the (anchor + midpoints) sequence in body
     frame. Sample N_OUTPUT points along the spline. Compute yaw at
     each sample from the spline derivative.
  7. Project samples back to world frame and emit.

# Why "forward-walking" and not Delaunay or full pairing

  - Robust to one missing side. FS courses routinely have stretches
    where the inside-of-corner cones are out of LiDAR FoV. The
    one-side fallback (step 4) keeps emitting a path where Delaunay
    would refuse to triangulate cleanly.
  - O(n) in cone count, no triangulation, no global graph. Fits
    inside a 10 Hz callback without any optimization.
  - Color-aware. We trust the cone-color tags coming from cone_slam
    (color is locked at first observation in LandmarkDb). If the
    upstream classifier ever fails, color confusion shows up here as
    a midpoint that lands ON a cone instead of between the corridors,
    which is easy to spot in viz.

# What this deliberately does NOT do

  - No Delaunay triangulation. urinay-style triangle filtering is
    neat but adds a configuration surface (max edge length, min
    angle) that's hard to tune and easy to break on sparse cones.
  - No tree search / multi-hypothesis. The car never has to choose
    between two plausible paths in FS — the cone gates are
    well-defined.
  - No loop closure. Path planning is local; loop closure (if any)
    lives in cone_slam.
  - No skidpad / acceleration mission specialization. Trackdrive only
    for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


# ----- tuning ----------------------------------------------------------------

# Body-frame X range we keep cones in. BODY_X_MIN is slightly negative
# so we don't drop cones "next to the car" the very first frame after
# the car crosses them — that lateral pair still defines the corridor
# the car is currently inside.
BODY_X_MIN = -0.5         # m, behind-the-car threshold (drop further-back cones)
LOOKAHEAD_M = 18.0        # m, drop cones beyond this forward distance

# Lateral cutoff. FS course is ≥ 3 m wide; cones ±5 m off the car's
# longitudinal axis are confidently "this lane". Wider would let the
# planner pull cones from a parallel section of track on tight loops.
HALF_CORRIDOR_M = 5.0

# Forward-walking step. Smaller = more midpoints (smoother spline,
# slower); larger = fewer midpoints, can miss tight curves. 1.5 m
# matches typical FS cone spacing along the racing line.
STEP_M = 1.5

# Standard FS track half-width (centerline → cone). Used for the
# one-side fallback when only LEFT or only RIGHT cones are visible
# at a given target_x.
TRACK_HALF_WIDTH_M = 1.5

# Output path density. 30 samples over ~18 m of lookahead = ~0.6 m
# spacing, fine enough for Stanley/Pure-Pursuit lookahead lookups.
N_OUTPUT = 30

# Minimum number of midpoints (including the car-anchor) before we'll
# emit a path. Two anchor + 1 midpoint is degenerate; require ≥ 3
# total so the cubic spline has something to interpolate.
MIN_MIDPOINTS = 3


# ----- public types ----------------------------------------------------------


class ConeColor(IntEnum):
    """Local mirror of cone_slam.color_classifier.ConeColor.

    Duplicated to keep this module independent of the cone_slam
    package — the planner is a downstream consumer and shouldn't
    import from upstream packages. The numeric values match.
    """

    YELLOW = 0       # right side of track
    BLUE = 1         # left side of track
    ORANGE = 2       # small orange (centerline / waypoint)
    BIG_ORANGE = 3   # large orange (start/finish)


@dataclass(frozen=True)
class Cone:
    x: float          # world frame
    y: float          # world frame
    color: ConeColor


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float        # radians


@dataclass(frozen=True)
class PathPoint:
    x: float          # world frame
    y: float          # world frame
    yaw: float        # radians, tangent direction


# ----- helpers ---------------------------------------------------------------


def _world_to_body(
    pts_xy: np.ndarray, pose: Pose2D,
) -> np.ndarray:
    """Project (N, 2) world-frame points into body frame."""
    if pts_xy.shape[0] == 0:
        return pts_xy
    c, s = np.cos(pose.yaw), np.sin(pose.yaw)
    R_w2b = np.array([[ c, s],
                      [-s, c]])
    return (pts_xy - np.array([pose.x, pose.y])) @ R_w2b.T


def _body_to_world(
    pts_xy: np.ndarray, pose: Pose2D,
) -> np.ndarray:
    """Project (N, 2) body-frame points into world frame."""
    if pts_xy.shape[0] == 0:
        return pts_xy
    c, s = np.cos(pose.yaw), np.sin(pose.yaw)
    R_b2w = np.array([[c, -s],
                      [s,  c]])
    return pts_xy @ R_b2w.T + np.array([pose.x, pose.y])


def _walk_midpoints_body(
    left_body: np.ndarray, right_body: np.ndarray,
) -> np.ndarray:
    """Forward-walking midpoint generator in body frame.

    Both inputs are (N, 2) sorted by body_x ascending. Returns the
    midpoint sequence starting from the car (0, 0) and walking forward
    by consuming cones in body_x order. Each cone is used at most
    once: once a cone is matched at a midpoint, it's "behind" the
    walker for subsequent steps.
    """
    pts: List[Tuple[float, float]] = [(0.0, 0.0)]   # car anchor
    li = ri = 0   # next-unused index into left_body / right_body
    nl, nr = left_body.shape[0], right_body.shape[0]
    last_x = max(BODY_X_MIN, 0.0)

    while li < nl or ri < nr:
        # Pick the side whose next-unused cone is closest in body_x.
        # That side gets to "advance" the walker; we then look at the
        # other side for a partner within ±STEP_M of that body_x.
        if li < nl and ri < nr:
            lead_left = left_body[li, 0] <= right_body[ri, 0]
        else:
            lead_left = li < nl

        if lead_left:
            lx = float(left_body[li, 0])
            ly = float(left_body[li, 1])
            li += 1
            if lx > LOOKAHEAD_M:
                break
            if lx <= last_x:
                # Already past this body_x — skip rather than going
                # backwards in the spline.
                continue
            partner = _consume_partner(right_body, ri, lx, STEP_M)
            if partner is not None:
                ry = float(right_body[partner, 1])
                ri = partner + 1
                if (ly - ry) > 1.0:
                    pts.append((lx, 0.5 * (ly + ry)))
                    last_x = lx
            else:
                pts.append((lx, ly - TRACK_HALF_WIDTH_M))
                last_x = lx
        else:
            rx = float(right_body[ri, 0])
            ry = float(right_body[ri, 1])
            ri += 1
            if rx > LOOKAHEAD_M:
                break
            if rx <= last_x:
                continue
            partner = _consume_partner(left_body, li, rx, STEP_M)
            if partner is not None:
                ly = float(left_body[partner, 1])
                li = partner + 1
                if (ly - ry) > 1.0:
                    pts.append((rx, 0.5 * (ly + ry)))
                    last_x = rx
            else:
                pts.append((rx, ry + TRACK_HALF_WIDTH_M))
                last_x = rx

    return np.array(pts)


def _consume_partner(
    cones_body: np.ndarray, start_idx: int,
    target_x: float, half_window: float,
) -> Optional[int]:
    """Among cones[start_idx:], return the index closest to target_x
    in body_x within ± half_window. None if no cone is in range."""
    if start_idx >= cones_body.shape[0]:
        return None
    sub = cones_body[start_idx:, 0]
    dx = np.abs(sub - target_x)
    in_window = dx <= half_window
    if not in_window.any():
        return None
    rel = int(np.argmin(np.where(in_window, dx, np.inf)))
    return start_idx + rel


def _spline_through(midpoints_body: np.ndarray) -> np.ndarray:
    """Fit a cubic spline through (N, 2) midpoints, sample N_OUTPUT
    points along arc length, return (N_OUTPUT, 3) of (x, y, yaw)
    in body frame.

    Pre: midpoints_body.shape[0] >= MIN_MIDPOINTS.
    """
    # Parameterize by cumulative arc length so closely-spaced midpoints
    # don't dominate the spline curvature. Strictly increasing s is
    # required by CubicSpline; with the forward-walking pattern this
    # is naturally true (each step advances target_x by STEP_M).
    diffs = np.diff(midpoints_body, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])

    cs_x = CubicSpline(s, midpoints_body[:, 0], bc_type="natural")
    cs_y = CubicSpline(s, midpoints_body[:, 1], bc_type="natural")

    s_dense = np.linspace(s[0], s[-1], num=N_OUTPUT)
    px = cs_x(s_dense)
    py = cs_y(s_dense)
    # Tangent direction from analytical derivative — smoother than
    # finite differences on the sampled output.
    dx = cs_x(s_dense, 1)
    dy = cs_y(s_dense, 1)
    pyaw = np.arctan2(dy, dx)

    return np.column_stack([px, py, pyaw])


# ----- public API ------------------------------------------------------------


def plan_centerline(cones: List[Cone], pose: Pose2D) -> List[PathPoint]:
    """Compute a centerline ahead of the car.

    Args:
        cones: full world-frame cone list. Color is honoured;
               UNKNOWN / centerline / start-finish cones are ignored.
        pose: current car pose in world frame.

    Returns:
        A list of PathPoints in world frame, ordered along arc length
        from the car forward. Empty list if no path could be computed
        (e.g., not enough cones, or no forward cones).
    """
    if not cones:
        return []

    # Bucket by color. We only use BLUE (LEFT) and YELLOW (RIGHT) for
    # path generation; ORANGE (centerline) is reserved for downstream
    # mission control and BIG_ORANGE for cone_slam loop-closure work.
    left_world = np.array(
        [[c.x, c.y] for c in cones if c.color == ConeColor.BLUE],
        dtype=float,
    ).reshape(-1, 2)
    right_world = np.array(
        [[c.x, c.y] for c in cones if c.color == ConeColor.YELLOW],
        dtype=float,
    ).reshape(-1, 2)

    # World → body and corridor filter.
    left_body = _world_to_body(left_world, pose)
    right_body = _world_to_body(right_world, pose)

    def _filter(b: np.ndarray) -> np.ndarray:
        if b.shape[0] == 0:
            return b
        mask = (
            (b[:, 0] >= BODY_X_MIN)
            & (b[:, 0] <= LOOKAHEAD_M)
            & (np.abs(b[:, 1]) <= HALF_CORRIDOR_M)
        )
        return b[mask]

    left_body = _filter(left_body)
    right_body = _filter(right_body)

    if left_body.shape[0] == 0 and right_body.shape[0] == 0:
        return []

    # Sort by forward distance for the walker.
    if left_body.shape[0]:
        left_body = left_body[np.argsort(left_body[:, 0])]
    if right_body.shape[0]:
        right_body = right_body[np.argsort(right_body[:, 0])]

    midpoints_body = _walk_midpoints_body(left_body, right_body)
    if midpoints_body.shape[0] < MIN_MIDPOINTS:
        return []

    samples_body = _spline_through(midpoints_body)

    # Project (x, y) back to world; rotate yaws by car yaw.
    samples_world_xy = _body_to_world(samples_body[:, :2], pose)
    yaws_world = samples_body[:, 2] + pose.yaw

    return [
        PathPoint(x=float(samples_world_xy[i, 0]),
                  y=float(samples_world_xy[i, 1]),
                  yaw=float(yaws_world[i]))
        for i in range(samples_world_xy.shape[0])
    ]

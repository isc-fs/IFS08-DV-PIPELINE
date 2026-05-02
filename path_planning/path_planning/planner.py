"""Centerline planner — Delaunay triangulation + BFS midpoint search.

Pure-Python, no ROS deps. Designed to be testable in isolation against
synthetic cone fixtures.

# Algorithm (urinay-style; see https://github.com/origovi/urinay)

Given a set of (color-tagged) cone positions in the world frame and a
car pose (x, y, yaw), compute a smooth centerline ahead of the car as
a sequence of (x, y, yaw) waypoints in the world frame.

  1. Project every cone into body frame (forward-X, left-Y).
  2. Drop cones outside a forward "corridor of interest"
     (BODY_X_MIN ≤ body_x ≤ LOOKAHEAD_M, |body_y| ≤ HALF_CORRIDOR_M).
  3. Delaunay-triangulate the surviving cone set (color-blind, like
     urinay master).
  4. Extract the unique edge set; for each edge precompute its
     midpoint, its length (track-width estimate at that location),
     and the two adjacent triangles. Drop edges longer than
     MAX_EDGE_LEN_M (almost certainly a track-spanning chord, not a
     track-width edge) or in triangles whose minimum interior angle
     is below MIN_TRI_ANGLE_RAD (sliver triangles confuse the search).
  5. Best-first search ("BFS with single-best-leaf"):
     - Seed at the car (body origin).
     - At each step, gather edge-midpoints within SEARCH_RADIUS_M of
       the current leaf. Filter through the seven-gate validity check
       (heading change limit, distance limit, no self-edges, no
       intersections with prior segments, track-width minimum, no
       backward "U-turn" candidates).
     - Score each survivor with a heuristic combining
         heading-change   (dominant)
         distance         (small)
         track-width consistency (tie-breaker)
       Pick the lowest-heuristic candidate, append, repeat.
     - Stop when no candidate passes the gates, or after MAX_PATH_PTS
       steps, or when the path has reached LOOKAHEAD_M of arc length.
  6. The car (body origin) is always the first path point.
  7. Project samples back to world frame and emit.

# Why Delaunay (vs forward-walking midpoint)

  - Handles asymmetric cone counts natively: when one side is sparse,
    the triangulation just produces longer edges on that side, which
    the edge-length filter strips, so the path naturally relies on
    triangles formed by the dense side's neighbours.
  - No "pair the closest cone in body_x" heuristic that flips
    tick-to-tick: the triangulation is geometrically defined and
    deterministic given the same cone set.
  - Cone-color-blind by construction. We trust the geometric
    structure of the cone field to encode the corridor.

# What this MVP deliberately does NOT do (yet)

  - Persistent midline (committed edges across ticks). Each tick
    re-plans from scratch. Robust enough for trackdrive at our
    speeds; the cost is some redundant work.
  - Loop closure. The controller has its own stop-latch on the
    big-orange finish gate.
  - Multi-hypothesis search with deferred commit. We pick the best
    candidate greedily at each step.
  - Failsafe parameter-multiplier retry. If the search starves, we
    return whatever path we have (clamped to MIN_MIDPOINTS).

These can be added in later commits on this branch as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Set, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import Delaunay, QhullError


# ----- tuning ----------------------------------------------------------------

# Body-frame X range we keep cones in. BODY_X_MIN is slightly negative
# so cones immediately next to the car (just behind the front axle in
# body frame) still contribute to triangulation.
BODY_X_MIN = -2.0         # m
LOOKAHEAD_M = 18.0        # m, drop cones beyond this forward distance

# Lateral cutoff: ignore cones in parallel sections of track on tight loops.
HALF_CORRIDOR_M = 8.0     # m

# Triangle / edge filters. urinay defaults (autocross.yml) translated.
MAX_EDGE_LEN_M = 7.0      # drop triangles with any edge longer than this
MIN_TRI_ANGLE_RAD = 0.35  # drop slivers (interior angle < ~20°)

# Search parameters.
# SEARCH_RADIUS_M was 5.0 m, which exactly excluded the very-first step
# from the car anchor (0,0) to the closest pair-midpoint at body_x ≈ 5
# (a 5 m gate spacing is FS-standard). Bumped to 7.0 so the first hop
# can pick that midpoint up; later hops between adjacent midpoints are
# typically 2–4 m so the wider radius isn't restrictive.
SEARCH_RADIUS_M = 7.0           # candidate gather radius around each leaf
MAX_HEADING_DELTA_RAD = 0.8     # ~46°: per-step heading change cap
MIN_MIDPOINT_DIST_M = 0.81      # don't pick a candidate within this of leaf
MIN_TRACK_WIDTH_M = 2.0         # the candidate's edge must be ≥ this long

# Heuristic weights. Heading dominates; track-width consistency is the
# tie-breaker; raw distance contributes little (urinay convention).
W_DIST = 0.1
W_TW_DIFF = 0.8

# Output path density. 30 samples over ~18 m of lookahead = ~0.6 m
# spacing, fine enough for Pure-Pursuit lookahead lookups.
N_OUTPUT = 30

# Minimum midpoints (incl. car anchor) before we'll emit a path.
MIN_MIDPOINTS = 3

# Maximum path-point count from the search; bounds runtime.
MAX_PATH_PTS = 25


# ----- public types ----------------------------------------------------------


class ConeColor(IntEnum):
    """Local mirror of cone_slam.color_classifier.ConeColor."""

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


# ----- frame helpers ---------------------------------------------------------


def _world_to_body(pts_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Project (N, 2) world-frame points into body frame."""
    if pts_xy.shape[0] == 0:
        return pts_xy
    c, s = np.cos(pose.yaw), np.sin(pose.yaw)
    R_w2b = np.array([[ c, s],
                      [-s, c]])
    return (pts_xy - np.array([pose.x, pose.y])) @ R_w2b.T


def _body_to_world(pts_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Project (N, 2) body-frame points into world frame."""
    if pts_xy.shape[0] == 0:
        return pts_xy
    c, s = np.cos(pose.yaw), np.sin(pose.yaw)
    R_b2w = np.array([[c, -s],
                      [s,  c]])
    return pts_xy @ R_b2w.T + np.array([pose.x, pose.y])


# ----- triangulation + edge extraction --------------------------------------


@dataclass
class _Edge:
    """A Delaunay-triangle edge between two cones (i, j) in body frame."""
    i: int                   # cone index
    j: int                   # cone index
    midpoint: np.ndarray     # (2,) midpoint in body frame
    length: float            # edge length (track-width estimate)


def _build_edges(
    cones_body: np.ndarray,
    colors_body: Optional[np.ndarray] = None,
) -> List[_Edge]:
    """Triangulate `cones_body` (N, 2) and return the unique edge set
    after triangle-quality filtering. Each edge appears once even if
    shared by two triangles.

    If `colors_body` (N,) is provided, edges between two same-color
    cones are dropped: their midpoints are not centerline crossings.
    On a tight corner the Delaunay triangulation routinely produces
    a long same-color edge that spans the corner (e.g. an outside
    cone before the apex paired with an outside cone past the apex).
    Its midpoint sits roughly along the straight extension of the
    approach, which then beats the real corner midpoints in the
    heading-biased score and steers the path the wrong way. Filtering
    same-color edges here closes that failure mode at the source.
    """
    n = cones_body.shape[0]
    if n < 3:
        return []
    try:
        tri = Delaunay(cones_body)
    except (QhullError, ValueError):
        return []

    seen: Set[Tuple[int, int]] = set()
    edges: List[_Edge] = []
    for simplex in tri.simplices:
        # Reject sliver triangles by minimum interior angle. A sliver
        # (one angle very small) means two of its three edges are
        # nearly the same line, which produces midpoints that snap
        # back and forth between near-identical positions.
        if _min_triangle_angle(cones_body[simplex]) < MIN_TRI_ANGLE_RAD:
            continue
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(simplex[a]), int(simplex[b])
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            if colors_body is not None and colors_body[i] == colors_body[j]:
                continue
            pi, pj = cones_body[i], cones_body[j]
            length = float(np.linalg.norm(pi - pj))
            # Long-edge filter: drop track-spanning chords. These show
            # up at the start/end of the cone field and at sharp
            # hairpins where the inside of the corner has no cones,
            # so the triangulation reaches across the track to find
            # neighbours.
            if length > MAX_EDGE_LEN_M:
                continue
            edges.append(_Edge(
                i=i, j=j,
                midpoint=0.5 * (pi + pj),
                length=length,
            ))
    return edges


def _min_triangle_angle(pts: np.ndarray) -> float:
    """Smallest interior angle (radians) of the triangle formed by
    the three rows of `pts`."""
    a, b, c = pts[0], pts[1], pts[2]
    angles = []
    for v0, v1, v2 in ((a, b, c), (b, c, a), (c, a, b)):
        u = v1 - v0
        w = v2 - v0
        cos = float(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-12))
        cos = max(-1.0, min(1.0, cos))
        angles.append(np.arccos(cos))
    return float(min(angles))


# ----- search ----------------------------------------------------------------


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _search_path_body(
    edges: List[_Edge],
    rejections: Optional[dict] = None,
) -> np.ndarray:
    """Greedy best-first walk over edge midpoints in body frame.

    Seeds at the car (origin). At each step picks the candidate
    midpoint that minimises the urinay heuristic, subject to the
    seven-gate validity filter. Returns (M, 2) midpoint array
    starting at the car anchor.

    If `rejections` is provided, increments per-gate counters
    every time a candidate edge is filtered out. Useful for
    diagnosing why a path is shorter than expected.
    """
    pts: List[np.ndarray] = [np.zeros(2)]    # car anchor at body origin
    used: Set[int] = set()                   # edge indices already on path
    prev_heading = 0.0                       # last segment heading (rad)
    prev_tw = 0.0                            # last edge's track width

    def _bump(reason: str) -> None:
        if rejections is not None:
            rejections[reason] = rejections.get(reason, 0) + 1

    while len(pts) < MAX_PATH_PTS:
        leaf = pts[-1]
        # Lookahead-budget cutoff: stop when path arc length already
        # spans LOOKAHEAD_M of forward distance.
        if leaf[0] >= LOOKAHEAD_M:
            break

        best_idx = -1
        best_score = float("inf")
        for k, e in enumerate(edges):
            if k in used:
                continue
            mp = e.midpoint
            d = float(np.linalg.norm(mp - leaf))
            if d < MIN_MIDPOINT_DIST_M or d > SEARCH_RADIUS_M:
                _bump("distance")
                continue
            if e.length < MIN_TRACK_WIDTH_M:
                _bump("track_width")
                continue

            # Heading from current leaf to candidate.
            head = float(np.arctan2(mp[1] - leaf[1], mp[0] - leaf[0]))
            # First step uses the leaf-to-cand heading as the seed,
            # so heading delta is 0; subsequent steps compare to prev.
            dhead = abs(_wrap_pi(head - prev_heading)) if len(pts) > 1 else 0.0
            if dhead > MAX_HEADING_DELTA_RAD:
                _bump("heading_delta")
                continue

            # Forward-only: candidate must be in front of the leaf in
            # the segment's heading frame. After the first segment
            # this is enforced by the heading-delta cap; for the very
            # first segment we require body-x forward of the leaf so
            # the path doesn't snap to a midpoint behind the car.
            if len(pts) == 1 and mp[0] <= leaf[0] + 1e-3:
                _bump("behind_leaf")
                continue

            # Self-intersection: don't allow the new segment to cross
            # any previous segment of the path. Only matters once the
            # path has at least two committed segments — in practice
            # the heading cap catches most U-turn shapes anyway, but
            # this guard prevents the rare hairpin self-cross.
            if len(pts) >= 3 and _segment_crosses_any(pts, leaf, mp):
                _bump("self_intersect")
                continue

            score = (
                (1.0 - W_DIST) * np.sqrt(dhead / (np.pi * 0.5))
                + W_DIST * (d / SEARCH_RADIUS_M)
                + W_TW_DIFF * (abs(e.length - prev_tw) / MAX_EDGE_LEN_M
                               if prev_tw > 0.0 else 0.0)
            )
            if score < best_score:
                best_score = score
                best_idx = k

        if best_idx < 0:
            break

        e = edges[best_idx]
        used.add(best_idx)
        pts.append(e.midpoint.copy())
        prev_heading = float(np.arctan2(
            e.midpoint[1] - leaf[1], e.midpoint[0] - leaf[0]))
        prev_tw = e.length

    return np.array(pts)


def _segment_crosses_any(
    pts: List[np.ndarray], a: np.ndarray, b: np.ndarray,
) -> bool:
    """True iff the segment a→b crosses any prior path segment
    pts[i]→pts[i+1]. Skips the segment ending at `a` (shared endpoint)."""
    for i in range(len(pts) - 2):
        c, d = pts[i], pts[i + 1]
        if _segments_intersect(a, b, c, d):
            return True
    return False


def _segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
) -> bool:
    """Standard 2D segment intersection (proper crossing, no
    collinear-overlap edge cases — adequate for our generic
    triangulated midpoints)."""
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))


# ----- spline smoothing -----------------------------------------------------


def _spline_through(midpoints_body: np.ndarray) -> np.ndarray:
    """Fit a PCHIP cubic Hermite spline through (N, 2) midpoints,
    sample N_OUTPUT points along arc length. Returns (N_OUTPUT, 3)
    of (x, y, yaw) in body frame.

    PCHIP (vs natural CubicSpline) is monotonic-preserving per
    coordinate and never overshoots, so an unevenly-spaced midpoint
    sequence produces a sane path.
    """
    diffs = np.diff(midpoints_body, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])

    cs_x = PchipInterpolator(s, midpoints_body[:, 0])
    cs_y = PchipInterpolator(s, midpoints_body[:, 1])

    s_dense = np.linspace(s[0], s[-1], num=N_OUTPUT)
    px = cs_x(s_dense)
    py = cs_y(s_dense)
    dx = cs_x.derivative()(s_dense)
    dy = cs_y.derivative()(s_dense)
    pyaw = np.arctan2(dy, dx)

    return np.column_stack([px, py, pyaw])


# ----- public API ------------------------------------------------------------


@dataclass
class PlanDebug:
    """Intermediate planner state, exposed for visualisation only.

    All arrays are in world frame (already projected from body frame
    by `plan_centerline_with_debug`) so the consumer can publish them
    on a TF that doesn't follow the car. Empty arrays are valid: at
    early ticks (few cones, no triangulation) they communicate that
    the planner failed at a specific stage.

    `rejections` is a per-search-step diagnostic accumulator, populated
    by `_search_path_body`. Counts how many candidate edges were
    discarded by each gate across the whole walk. Useful when a path
    comes back too short and we need to know which filter was the
    bottleneck without re-running with a debugger. Keys: distance,
    track_width, heading_delta, behind_leaf, self_intersect.
    """
    triangulation_edges: np.ndarray = None  # (E, 2, 2) — endpoints per edge
    candidate_midpoints: np.ndarray = None  # (E, 2) — one per edge
    selected_midpoints: np.ndarray = None   # (M, 2) — best-first chosen
    rejections: dict = None                  # gate_name -> count


def plan_centerline(cones: List[Cone], pose: Pose2D) -> List[PathPoint]:
    """Compute a centerline ahead of the car. Public API; no debug data.
    """
    path, _ = plan_centerline_with_debug(cones, pose)
    return path


def plan_centerline_with_debug(
    cones: List[Cone], pose: Pose2D,
) -> Tuple[List[PathPoint], PlanDebug]:
    """Same as plan_centerline, but also returns intermediate state
    (triangulation edges, candidate midpoints, selected midpoints) for
    visualisation. The ROS node uses this to publish a Lichtblick /
    Foxglove debug overlay on /path_planning/delaunay/*.

    All debug arrays are in world frame.
    """
    debug = PlanDebug(
        triangulation_edges=np.empty((0, 2, 2)),
        candidate_midpoints=np.empty((0, 2)),
        selected_midpoints=np.empty((0, 2)),
        rejections={},
    )

    if not cones:
        return [], debug

    track_cones = [c for c in cones if c.color in (ConeColor.BLUE, ConeColor.YELLOW)]
    if len(track_cones) < 3:
        return [], debug

    cones_world = np.array([[c.x, c.y] for c in track_cones], dtype=float)
    cones_body = _world_to_body(cones_world, pose)
    colors_body = np.array([int(c.color) for c in track_cones])

    mask = (
        (cones_body[:, 0] >= BODY_X_MIN)
        & (cones_body[:, 0] <= LOOKAHEAD_M)
        & (np.abs(cones_body[:, 1]) <= HALF_CORRIDOR_M)
    )
    cones_body = cones_body[mask]
    colors_body = colors_body[mask]
    if cones_body.shape[0] < 3:
        return [], debug

    edges = _build_edges(cones_body, colors_body)
    if not edges:
        return [], debug

    # Capture debug snapshot — body→world projection happens once for
    # all debug arrays so they share a single transform.
    if edges:
        edge_pts_body = np.array([
            [cones_body[e.i], cones_body[e.j]] for e in edges
        ])  # (E, 2, 2)
        # Flatten endpoints for batch transform, then reshape.
        flat = edge_pts_body.reshape(-1, 2)
        flat_world = _body_to_world(flat, pose)
        debug.triangulation_edges = flat_world.reshape(-1, 2, 2)

        cand_body = np.array([e.midpoint for e in edges])  # (E, 2)
        debug.candidate_midpoints = _body_to_world(cand_body, pose)

    midpoints_body = _search_path_body(edges, rejections=debug.rejections)
    if midpoints_body.shape[0] >= 1:
        debug.selected_midpoints = _body_to_world(midpoints_body, pose)

    if midpoints_body.shape[0] < MIN_MIDPOINTS:
        return [], debug

    samples_body = _spline_through(midpoints_body)
    samples_world_xy = _body_to_world(samples_body[:, :2], pose)
    yaws_world = samples_body[:, 2] + pose.yaw

    path = [
        PathPoint(x=float(samples_world_xy[i, 0]),
                  y=float(samples_world_xy[i, 1]),
                  yaw=float(yaws_world[i]))
        for i in range(samples_world_xy.shape[0])
    ]
    return path, debug

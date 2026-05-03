"""Adapter for the FaSTTUBe Formula Student path planning library.

Wraps `fsd_path_planning.PathPlanner` (https://github.com/papalotis/ft-fsd-path-planning,
MIT, by FaSTTUBe — papalotis is the maintainer's GitHub handle) to match the
input/output contract the ROS node expects: list of `Cone` + `Pose2D` in,
list of `PathPoint` out.

Why FaSTTUBe replaces our Delaunay+walker planner (PR #243):
  - The walker was poisoned by spurious orange-classified cones in the cone
    soup (see fix/241 SLAM audit) — Delaunay edges through ghost orange cones
    looked cross-colour to the walker, so it picked midpoints on the wrong
    side of the track.
  - The walker also struggled at one-sided observation regions (PR #189) —
    when only one side's cones are visible, only same-colour Delaunay edges
    exist and the walker's midpoints sit on the outside arc.
  - FaSTTUBe sorts each side independently and matches across sides, so
    orange ghosts get dropped naturally and one-sided regions are inferred
    from the present side's geometry.

This adapter is intentionally thin. The real algorithm lives upstream;
we own the type translation, the empty/exception guards, and the body-frame
yaw recomputation the controller wants.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner

from path_planning.core_types import Cone, ConeColor, PathPoint, Pose2D

logger = logging.getLogger(__name__)


# Map our ConeColor → FaSTTUBe ConeTypes. Convention agrees (blue=left,
# yellow=right) but the integer codes differ:
#   ours:    YELLOW=0, BLUE=1, ORANGE=2, BIG_ORANGE=3
#   theirs:  UNKNOWN=0, RIGHT/YELLOW=1, LEFT/BLUE=2, ORANGE_SMALL=3, ORANGE_BIG=4
# We map explicitly via this dict — never integer-cast.
_COLOR_TO_CONETYPE = {
    ConeColor.YELLOW: ConeTypes.RIGHT,         # = 1
    ConeColor.BLUE: ConeTypes.LEFT,            # = 2
    ConeColor.ORANGE: ConeTypes.ORANGE_SMALL,  # = 3
    ConeColor.BIG_ORANGE: ConeTypes.ORANGE_BIG, # = 4
}

# FaSTTUBe expects exactly 5 cone arrays indexed 0..4 (one per ConeTypes).
# Slot 0 is UNKNOWN (cones with no colour info) — we don't currently
# produce those because cone_slam always assigns a colour at landmark
# creation. Kept as an empty array for forward compatibility.
_NUM_CONE_TYPES = 5


class FasttubeAdapter:
    """Wraps a single `PathPlanner` instance and translates I/O.

    Lifetime: construct once, reuse across plan calls. The library is
    stateless for trackdrive (the mission we ship), so the same instance
    serves every cone callback. Skidpad/acceleration are stateful and
    tracked separately in issue #243 — out of scope for this PR.
    """

    def __init__(self, mission: MissionTypes = MissionTypes.trackdrive) -> None:
        self._planner = PathPlanner(mission)
        self._mission = mission
        # Rate-limited error logging — papalotis can raise on degenerate
        # cone fields (very few cones, all same colour, malformed
        # geometry). We must never let that take down the node, but we
        # also don't want a crash loop spamming syslog.
        self._last_error_log_time: float = 0.0

    def plan(self, cones: List[Cone], pose: Pose2D) -> List[PathPoint]:
        """Compute centerline path; returns [] on degenerate input or library failure.

        Path points are in the same world frame as `cones` and `pose`
        (the SLAM `odom` frame in our pipeline). Yaw is recomputed from
        finite differences along the path because FaSTTUBe outputs
        `(s, x, y, curvature)` and the downstream controller only reads
        `(x, y)` from the published `nav_msgs/Path` anyway.
        """
        if not cones:
            return []

        global_cones = self._cones_to_arrays(cones)
        car_position = np.array([pose.x, pose.y], dtype=np.float64)
        # Unit vector form (preferred over yaw float per the README
        # example — fewer wrap-around bugs).
        car_direction = np.array(
            [np.cos(pose.yaw), np.sin(pose.yaw)], dtype=np.float64
        )

        try:
            path = self._planner.calculate_path_in_global_frame(
                global_cones, car_position, car_direction
            )
        except Exception as e:  # pylint: disable=broad-except
            # Library may raise on tiny/degenerate cone fields. Return []
            # so the node logs its own plan_empty counter and the
            # controller keeps the previous reference. Rate-limit at one
            # log per ~5 s.
            import time
            now = time.monotonic()
            if now - self._last_error_log_time > 5.0:
                logger.warning(
                    "fasttube planner raised %s: %s (n_cones=%d)",
                    type(e).__name__, e, len(cones),
                )
                self._last_error_log_time = now
            return []

        if path is None or path.size == 0:
            return []

        # path is (M, 4) with columns (s, x, y, curvature). We only need
        # x, y — yaw recomputed below, curvature is dropped (controller
        # builds its own from the lookahead point).
        xy = np.asarray(path[:, 1:3], dtype=np.float64)
        if xy.shape[0] < 2:
            return []
        return _xy_to_path_points(xy)

    @staticmethod
    def _cones_to_arrays(cones: List[Cone]) -> List[np.ndarray]:
        """Bucket cones into the 5-array list FaSTTUBe expects."""
        buckets: List[List[List[float]]] = [[] for _ in range(_NUM_CONE_TYPES)]
        for c in cones:
            cone_type = _COLOR_TO_CONETYPE.get(c.color)
            if cone_type is None:
                # Unrecognised colour code — drop rather than crash.
                # Should not happen given ConeColor is an IntEnum, but
                # be defensive against future enum extensions.
                continue
            buckets[int(cone_type)].append([c.x, c.y])
        return [
            np.asarray(b, dtype=np.float64).reshape(-1, 2)
            for b in buckets
        ]


def _xy_to_path_points(xy: np.ndarray) -> List[PathPoint]:
    """Convert (M, 2) world-frame samples to PathPoint with finite-difference yaw."""
    n = xy.shape[0]
    out: List[PathPoint] = []
    for i in range(n):
        if i + 1 < n:
            dx = xy[i + 1, 0] - xy[i, 0]
            dy = xy[i + 1, 1] - xy[i, 1]
        else:
            # Last point: reuse previous segment's heading so the path
            # tangent is continuous at the tip.
            dx = xy[i, 0] - xy[i - 1, 0]
            dy = xy[i, 1] - xy[i - 1, 1]
        yaw = float(np.arctan2(dy, dx))
        out.append(PathPoint(x=float(xy[i, 0]), y=float(xy[i, 1]), yaw=yaw))
    return out

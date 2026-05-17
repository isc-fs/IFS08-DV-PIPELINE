"""Phase 1 mapper for the two-phase cone SLAM (#496).

Phase 1 runs during the exploration lap (or the entire run for
single-lap missions like autocross). It takes:

  - the latest pose from an external source (production: `/odom` from
    odometry_filter_node; testing: ground-truth from
    `/testing_only/odom`),
  - each scan's body-frame cone observations from
    `/Conos_raw` / cone_detection_node,

and accumulates them into a `LandmarkDb` of world-frame cone
landmarks with per-landmark sigma. On lap completion, the lifecycle
node freezes the DB into a `FrozenMap` (see `frozen_map.py`) and
hands off to `Phase2Localiser`.

## Why no internal pose estimator (option D)

The pre-rewrite cone_graph_slam tried to jointly optimise pose +
landmarks + IMU biases in an iSAM2 factor graph. That ran into a
~6 m position bias that established itself in the first 3 s and
never recovered, with no DA cascade — i.e. the math itself was
mis-converging, not the data association. Replacing that with a
small fixed-state EKF re-introduces the same family of failure
modes (filter inconsistency, linearisation drift on long mapping
laps).

Post-#526 the odometry_filter EKF is reliable enough to serve as
the pose source for Phase 1, so the mapper here doesn't carry its
own state. Phase 1 is therefore a thin layer on top of dead
reckoning + cone observation accumulation. Loop closure is
deliberately not implemented: lap 1 is open-loop-on-arrival in
trackdrive, and autocross is single-lap by spec.

If `/odom` ever has drift > Phase 1's `pose_error_max_m` threshold
on the regression bag, that's an odometry_filter problem; we fix
it there, not here.

## Public API

    mapper = Phase1Mapper(landmark_db=db, da_gate_m=1.0,
                          obs_sigma_m=0.20)
    mapper.observe_scan(pose, observations)
    snapshot = mapper.snapshot_for_freeze()   # at lap completion

This module is ROS-free; `slam_node.py` wires it to /odom + /Conos_raw.

"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from cone_slam.landmark_db import Landmark, LandmarkDb


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Pose2D:
    """SE(2) pose in the world frame.

    `yaw` is the heading in rad, wrapped to (-π, π]. We use 2D here
    because FSD tracks are on a flat ground plane and the cone
    observations are projected onto z=0; carrying a full 3D pose
    would require lifting cone observations to 3D too, which buys
    nothing.
    """
    x: float
    y: float
    yaw: float

    @classmethod
    def from_ros_pose(cls, ros_pose) -> "Pose2D":
        """Build a Pose2D from a geometry_msgs.msg.Pose. Imported
        deferred so this module stays ROS-free for the unit tests."""
        p = ros_pose.position
        q = ros_pose.orientation
        # 2D yaw from quaternion.
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return cls(x=float(p.x), y=float(p.y), yaw=float(math.atan2(siny, cosy)))


@dataclass(frozen=True)
class Observation:
    """One cone observation in the vehicle body frame.

    Body frame convention matches REP-103 base_link: x forward,
    y left, z up. The mapper rotates (body_x, body_y) into world
    frame using the pose's yaw, then translates by the pose's
    (x, y).

    `sigma_m` is the assumed observation noise (the post-cluster
    centroid uncertainty cone_detection emits). Cone_detection's
    output typically has sigma in the 5-15 cm range; conservative
    default is 20 cm. Each observation can carry its own sigma if
    the perception stack ever exports per-cone uncertainty.
    """
    body_x: float
    body_y: float
    is_big_orange: bool = False
    sigma_m: float = 0.20


# ---------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------

class Phase1Mapper:
    """Accumulates cone observations into a world-frame LandmarkDb."""

    def __init__(
        self,
        landmark_db: LandmarkDb,
        *,
        da_gate_m: float = 1.0,
        obs_sigma_m: float = 0.20,
    ) -> None:
        """
        Args:
            landmark_db: the DB the mapper writes into. Caller owns it
                so the lifecycle node can publish its contents on every
                tick without re-wrapping.
            da_gate_m: nearest-neighbour gate radius. An observation
                whose projected world-frame position is closer than
                this to an existing landmark associates to that
                landmark; otherwise a new landmark is created.
                1.0 m is conservative for FSD courses (cones are
                spaced ~3 m apart at the tightest). Tune lower for
                dense tracks if false-associations show up.
            obs_sigma_m: default per-observation sigma applied when
                an `Observation` doesn't carry an explicit sigma.
        """
        self._db = landmark_db
        self._da_gate_m = float(da_gate_m)
        self._obs_sigma_m = float(obs_sigma_m)
        # Per-scan step counter. Used as `last_seen_step` on each
        # landmark so we can later detect cones that haven't been
        # seen for a while (Phase 1 doesn't act on it, but the field
        # already exists in landmark_db and we keep it consistent for
        # downstream consumers).
        self._step: int = 0

    # ----- public API -----

    @property
    def db(self) -> LandmarkDb:
        return self._db

    @property
    def step(self) -> int:
        return self._step

    def observe_scan(
        self,
        pose: Pose2D,
        observations: Iterable[Observation],
    ) -> dict:
        """Process one cone scan.

        Returns a per-scan summary dict so the caller can log it as a
        SLAM_OBS-style diagnostic:
            n_obs, n_assoc, n_new, n_big_orange_landmarks.
        """
        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        n_obs = 0
        n_assoc = 0
        n_new = 0
        for obs in observations:
            n_obs += 1
            sigma = obs.sigma_m if obs.sigma_m > 0 else self._obs_sigma_m
            world_x = pose.x + c * obs.body_x - s * obs.body_y
            world_y = pose.y + s * obs.body_x + c * obs.body_y
            world_xy = np.array([world_x, world_y])

            match = self._nearest_landmark(world_xy)
            if match is not None and self._within_gate(match, world_xy):
                # Associate.
                match.update_running_mean(world_xy, sigma)
                # is_big_orange "wins on True" — once any observation
                # of a landmark is tagged big-orange, the landmark
                # stays big-orange. The lap detector relies on this
                # being stable across observations.
                if obs.is_big_orange and not match.is_big_orange:
                    match.is_big_orange = True
                self._db.mark_observed(match.id, self._step)
                n_assoc += 1
            else:
                # New landmark.
                init_pos3 = np.array([world_x, world_y, 0.0])
                self._db.create(
                    init_pos3,
                    step=self._step,
                    sigma_xy=sigma,
                    is_big_orange=obs.is_big_orange,
                )
                n_new += 1

        self._step += 1
        n_big_orange = sum(1 for lm in self._db if lm.is_big_orange)
        return {
            "step": self._step,
            "n_obs": n_obs,
            "n_assoc": n_assoc,
            "n_new": n_new,
            "n_map": len(self._db),
            "n_big_orange": n_big_orange,
        }

    def snapshot_for_freeze(self) -> list[Landmark]:
        """Return a shallow-copy list of landmarks for handoff to
        `FrozenMap`. The Landmark objects themselves are still
        references into the DB — caller must treat them as
        read-only after the freeze."""
        return list(self._db)

    # ----- internals -----

    def _nearest_landmark(self, world_xy: np.ndarray) -> Optional[Landmark]:
        """O(N) nearest-landmark scan. With ~200 cones on an autocross
        track and ~10 Hz scans, N is small enough that a KD-tree is
        not yet warranted. If profiling later shows this is hot, swap
        in scipy.spatial.cKDTree against a snapshotted positions
        array (rebuild on landmark create — landmark count grows
        only during Phase 1)."""
        x, y = float(world_xy[0]), float(world_xy[1])
        best: Optional[Landmark] = None
        best_d = float("inf")
        for lm in self._db:
            dx = x - lm.position[0]
            dy = y - lm.position[1]
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d = d
                best = lm
        return best

    def _within_gate(self, lm: Landmark, world_xy: np.ndarray) -> bool:
        dx = float(world_xy[0]) - lm.position[0]
        dy = float(world_xy[1]) - lm.position[1]
        return math.hypot(dx, dy) <= self._da_gate_m

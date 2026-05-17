"""Big-orange-gate lap-completion detector.

A FSD track is bounded by yellow / blue cones (right / left side)
plus a pair of BIG ORANGE cones at the start-finish line. The
detector watches the car's progress relative to the big-orange
pair and fires `lap_completed = True` when the car physically
crosses the line.

State machine:

    BEFORE_START      The detector hasn't yet identified a stable
                      big-orange pair. We sit here until two
                      big-orange landmarks within 6 m of each other
                      and ±5 m of the spawn point exist in the map.
    APPROACHING       We have a start gate. The car is now driving
                      around the loop. We track cumulative travelled
                      distance and the sign of the line-crossing
                      projection.
    CROSSED           lap_completed has fired once; we stay here
                      until the user resets the detector for a new
                      lap (trackdrive uses the same instance with
                      reset_for_next_lap()).

Three guards prevent false positives:

  1. `min_lap_distance_m` (default 30 m) — cumulative travelled
     distance must exceed this before any crossing fires. Prevents
     the detector firing on the start line at t=0.
  2. Sign flip of the perpendicular-projection onto the gate axis —
     a crossing is detected only when the car transitions from
     "before the gate" to "past the gate" relative to the gate's
     line. Approaching the gate but not crossing it doesn't fire.
  3. Big-orange pair stability — the gate is snapshotted on first
     identification and not updated thereafter. Random big-orange
     cones elsewhere on the track (autocross sometimes plants them)
     don't drift the gate.

The detector is ROS-free and pose-source-agnostic. It receives a
`Pose2D` and a list of `Landmark` per step from the lifecycle node.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, Optional

import numpy as np

from cone_slam.landmark_db import Landmark
from cone_slam.phase1_mapper import Pose2D


class _State(Enum):
    BEFORE_START = auto()
    APPROACHING = auto()
    CROSSED = auto()


@dataclass(frozen=True)
class _StartGate:
    """Snapshotted big-orange pair at lap-detector arming time."""
    left_xy:  np.ndarray   # (2,) one of the big-orange cones
    right_xy: np.ndarray   # (2,) the other
    midpoint: np.ndarray   # (2,) midpoint of the pair
    # Unit normal pointing in the "forward" travel direction at the
    # gate. Initialised from the car's pose at gate detection time.
    forward_unit: np.ndarray  # (2,)


class LapDetector:
    """Detects lap completion via a big-orange gate crossing."""

    def __init__(
        self,
        *,
        max_gate_pair_spacing_m: float = 6.0,
        max_gate_search_radius_m: float = 5.0,
        min_lap_distance_m: float = 30.0,
    ) -> None:
        """
        Args:
            max_gate_pair_spacing_m: Two big-orange landmarks closer
                than this are considered candidate gate pairs.
                FS rules-2026 spec width is 3 m; 6 m gives slack for
                survey noise.
            max_gate_search_radius_m: When looking for the start gate,
                only consider big-orange landmarks within this
                distance of the SPAWN POINT (the pose at the very
                first observe call). Stops random orange cones
                elsewhere on the track from being mistaken for the
                gate.
            min_lap_distance_m: cumulative travelled distance gate
                (see module docstring).
        """
        self._max_gate_pair_spacing_m = float(max_gate_pair_spacing_m)
        self._max_gate_search_radius_m = float(max_gate_search_radius_m)
        self._min_lap_distance_m = float(min_lap_distance_m)

        self._state = _State.BEFORE_START
        self._gate: Optional[_StartGate] = None
        # Anchor for distance accumulation + start-gate search.
        self._spawn_xy: Optional[np.ndarray] = None
        self._prev_pose_xy: Optional[np.ndarray] = None
        self._cumulative_distance_m: float = 0.0
        # Sign of the perpendicular-projection from the previous tick;
        # a flip from + to - (or - to +) past the gate indicates a
        # crossing.
        self._prev_signed_proj: Optional[float] = None

    # ----- public API -----

    @property
    def state(self) -> str:
        return self._state.name

    @property
    def cumulative_distance_m(self) -> float:
        return self._cumulative_distance_m

    @property
    def gate_armed(self) -> bool:
        """True once the start-gate pair has been identified."""
        return self._gate is not None

    def observe(
        self,
        pose: Pose2D,
        landmarks: Iterable[Landmark],
    ) -> bool:
        """Process one tick. Returns True iff this tick triggers lap
        completion (the rising edge — subsequent calls return False
        until `reset_for_next_lap()`)."""
        pose_xy = np.array([pose.x, pose.y])

        # Anchor + cumulative-distance tracking.
        if self._spawn_xy is None:
            self._spawn_xy = pose_xy.copy()
        if self._prev_pose_xy is not None:
            self._cumulative_distance_m += float(
                np.linalg.norm(pose_xy - self._prev_pose_xy)
            )
        self._prev_pose_xy = pose_xy.copy()

        # Already crossed — quiescent until reset.
        if self._state == _State.CROSSED:
            return False

        # Try to identify the start gate while we don't have one.
        if self._gate is None:
            gate = self._try_identify_gate(landmarks)
            if gate is None:
                return False
            # Initialise the forward-unit using the car's CURRENT
            # heading (gate is detected while car is on the start
            # line facing into the track).
            forward_unit = np.array([math.cos(pose.yaw),
                                     math.sin(pose.yaw)])
            self._gate = _StartGate(
                left_xy=gate[0],
                right_xy=gate[1],
                midpoint=(gate[0] + gate[1]) * 0.5,
                forward_unit=forward_unit,
            )
            self._state = _State.APPROACHING
            return False

        # We have a gate. Check for a crossing.
        signed_proj = self._signed_projection(pose_xy)
        crossed = False
        if (self._prev_signed_proj is not None
                and self._cumulative_distance_m >= self._min_lap_distance_m
                and self._sign_changed(self._prev_signed_proj, signed_proj)):
            crossed = True
            self._state = _State.CROSSED
        self._prev_signed_proj = signed_proj
        return crossed

    def reset_for_next_lap(self) -> None:
        """Re-arm the detector for the next lap (trackdrive). Keeps
        the gate snapshot and spawn anchor; resets only the crossing
        state."""
        self._state = _State.APPROACHING
        self._prev_signed_proj = None

    # ----- internals -----

    def _try_identify_gate(
        self,
        landmarks: Iterable[Landmark],
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Find two big-orange landmarks within (a) max_gate_pair_
        spacing of each other AND (b) max_gate_search_radius of the
        spawn point. Returns their (left, right) XY pair, or None
        if no qualifying pair exists yet."""
        if self._spawn_xy is None:
            return None
        candidates: list[np.ndarray] = []
        for lm in landmarks:
            if not lm.is_big_orange:
                continue
            xy = lm.position[:2]
            if float(np.linalg.norm(xy - self._spawn_xy)) > self._max_gate_search_radius_m:
                continue
            candidates.append(xy.copy())
        if len(candidates) < 2:
            return None
        # Brute O(N²) — typical N is 2-4 big-orange cones near spawn.
        best_pair: Optional[tuple[np.ndarray, np.ndarray]] = None
        best_spacing = float("inf")
        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                d = float(np.linalg.norm(a - b))
                if d > self._max_gate_pair_spacing_m:
                    continue
                if d < best_spacing:
                    best_spacing = d
                    best_pair = (a, b)
        return best_pair

    def _signed_projection(self, pose_xy: np.ndarray) -> float:
        """Signed scalar distance of `pose_xy` from the gate's line,
        positive in the gate's forward_unit direction.

        Crossing the gate is a sign change of this value.
        """
        assert self._gate is not None
        delta = pose_xy - self._gate.midpoint
        return float(np.dot(delta, self._gate.forward_unit))

    @staticmethod
    def _sign_changed(prev: float, curr: float) -> bool:
        # Strictly across zero. Exact-zero is not a crossing on its
        # own — we wait for the next tick to confirm direction.
        return (prev < 0.0 and curr > 0.0) or (prev > 0.0 and curr < 0.0)

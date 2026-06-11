"""Persistent cone landmark database.

The keystone gap audit identified at the start of the pivot was:
*existing cone detection has no per-landmark identity across frames*.
This module fills that gap.

Every cone is assigned a stable integer ID at first sighting. The ID
is the GTSAM landmark key — gtsam.symbol('l', id) — so the optimizer
can attach BearingRangeFactor3D constraints to it from any pose node
that observes it again in future scans. That's the "loop closure for
free" that GraphSLAM gives you over filter-based SLAM.

Position-only — no per-landmark colour. The body_y classifier that
used to set this was a heuristic with no real colour signal behind it,
and removing it from the pipeline removes the colour-flicker class of
DA failures we used to band-aid. Visualization renders all landmarks
the same colour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class Landmark:
    """One persistent cone in the world map."""

    id: int                                # gtsam landmark key index
    position: np.ndarray = field(           # (3,) world-frame x, y, z. Updated
        default_factory=lambda: np.zeros(3))  # by iSAM2 every scan.
    n_observations: int = 0                # how many factors point at it
    last_seen_step: int = -1               # graph step index of last association

    def update_estimate(self, position: np.ndarray) -> None:
        """Refresh the working position estimate from iSAM2."""
        self.position = position.copy()


class LandmarkDb:
    """Owner of all cone landmarks with persistent IDs."""

    def __init__(self) -> None:
        self._landmarks: Dict[int, Landmark] = {}
        self._next_id: int = 0

    # ----- create ------------------------------------------------------------

    def create(self, initial_position: np.ndarray, step: int) -> Landmark:
        """Allocate a new landmark with a fresh ID."""
        lm = Landmark(
            id=self._next_id,
            position=np.asarray(initial_position, dtype=float).copy(),
            n_observations=1,
            last_seen_step=step,
        )
        self._landmarks[lm.id] = lm
        self._next_id += 1
        return lm

    # ----- lookups -----------------------------------------------------------

    def get(self, lid: int) -> Landmark:
        return self._landmarks[lid]

    def __len__(self) -> int:
        return len(self._landmarks)

    def __iter__(self):
        return iter(self._landmarks.values())

    # ----- mutate from optimizer -------------------------------------------

    def update_from_estimate(self, get_position) -> None:
        """Walk every landmark and refresh its working position.

        Args:
            get_position: callable taking a landmark id and returning the
                latest np.ndarray position from iSAM2.
        """
        for lm in self._landmarks.values():
            try:
                lm.update_estimate(get_position(lm.id))
            except Exception:
                # Landmark not yet in the optimizer (just-added factor
                # hasn't been merged into iSAM2 yet). Leave the working
                # estimate in place; the next call will catch up.
                pass

    def mark_observed(self, lid: int, step: int) -> None:
        lm = self._landmarks[lid]
        lm.n_observations += 1
        lm.last_seen_step = step

    # ----- proximity query --------------------------------------------------

    def nearest_xy_distance_m(self, xy: np.ndarray) -> float:
        """Euclidean XY distance from `xy` to the closest existing
        landmark. Returns +inf when the db is empty. Z is ignored —
        cone landmarks are 2D for FSD tracks."""
        if not self._landmarks:
            return float("inf")
        x, y = float(xy[0]), float(xy[1])
        best = float("inf")
        for lm in self._landmarks.values():
            dx = x - lm.position[0]
            dy = y - lm.position[1]
            d = math.hypot(dx, dy)
            if d < best:
                best = d
        return best

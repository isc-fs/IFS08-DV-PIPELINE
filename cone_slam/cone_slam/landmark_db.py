"""Persistent cone landmark database.

The keystone gap audit identified at the start of the pivot was:
*existing cone detection has no per-landmark identity across frames*.
This module fills that gap.

Every cone is assigned a stable integer ID at first sighting. The ID
is the GTSAM landmark key — gtsam.symbol('l', id) — so the optimizer
can attach BearingRangeFactor3D constraints to it from any pose node
that observes it again in future scans. That's the "loop closure for
free" that GraphSLAM gives you over filter-based SLAM.

Landmark color is locked at FIRST observation. The audited spatial
classifier already has a hidden assumption that color shouldn't flip
once cached, and we lean into that — re-classifying on every scan
defeats the point of having stable IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from cone_slam.color_classifier import ConeColor


@dataclass
class Landmark:
    """One persistent cone in the world map."""

    id: int                                # gtsam landmark key index
    color: ConeColor                       # locked at first observation
    position: np.ndarray = field(           # (3,) world-frame x, y, z. Updated
        default_factory=lambda: np.zeros(3))  # by iSAM2 every scan.
    n_observations: int = 0                # how many factors point at it
    last_seen_step: int = -1               # graph step index of last association

    def update_estimate(self, position: np.ndarray) -> None:
        """Refresh the working position estimate from iSAM2."""
        self.position = position.copy()


class LandmarkDb:
    """Owner of all cone landmarks with persistent IDs.

    Lookups are by ID for everything except the data-association code,
    which scans by color. Both are cheap because cone counts stay
    small (FS tracks have ~100-300 cones total).
    """

    def __init__(self) -> None:
        self._landmarks: Dict[int, Landmark] = {}
        self._next_id: int = 0

    # ----- create ------------------------------------------------------------

    def create(
        self, color: ConeColor, initial_position: np.ndarray, step: int
    ) -> Landmark:
        """Allocate a new landmark with a fresh ID."""
        lm = Landmark(
            id=self._next_id,
            color=color,
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

    def all_by_color(self, color: ConeColor) -> List[Landmark]:
        """Return every landmark of the given color. Used by data
        association for the color-gate filter."""
        return [lm for lm in self._landmarks.values() if lm.color == color]

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

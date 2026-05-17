"""Immutable cone map for Phase 2 localisation.

At lap completion Phase 1 hands the live `LandmarkDb` over to
`Phase2Localiser` as a `FrozenMap`: positions and sigmas snapshotted,
no more inserts, no more updates. The frozen contract is what makes
Phase 2 structurally robust — there's no map state for DA cascades to
corrupt.

KD-tree backed for fast nearest-neighbour and radius queries; cone
counts are typically ~100-200 per FSD track, so the build cost is
trivial and amortised across hundreds of localisation ticks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from scipy.spatial import cKDTree
    _SCIPY_OK = True
except ImportError:
    # scipy is in dv_pipeline_stack's image but the regression test
    # may run on a thinner host environment. Fall back to a tiny
    # brute-force implementation so the rest of the API still works
    # for tests.
    cKDTree = None  # type: ignore[assignment]
    _SCIPY_OK = False

from cone_slam.landmark_db import Landmark


@dataclass(frozen=True)
class FrozenMap:
    """Immutable snapshot of a Phase 1 LandmarkDb.

    Attributes are public arrays so the EKF measurement Jacobian can
    take direct slices without re-copying. Treat as read-only after
    construction — mutating the arrays is undefined behaviour.

    `kdtree` is built lazily on first query (see methods below); we
    keep it off the dataclass so the dataclass stays
    `frozen=True`-compatible.
    """
    positions: np.ndarray       # (N, 2) world-frame x, y
    sigmas:    np.ndarray       # (N,)   per-landmark sigma_xy
    is_big_orange: np.ndarray   # (N,)   bool

    # ----- factories -----

    @classmethod
    def from_landmarks(cls, landmarks: Iterable[Landmark]) -> "FrozenMap":
        """Build a FrozenMap from a Phase 1 mapper's snapshot.

        Skips landmarks with `n_observations < 2` — those have only
        been seen once and are noisy single-shot detections that
        Phase 2 shouldn't trust. Tune the threshold if Phase 1's
        DA gate ever gets so tight that legitimate cones routinely
        sit at n=1.
        """
        kept = [lm for lm in landmarks if lm.n_observations >= 2]
        n = len(kept)
        positions = np.zeros((n, 2), dtype=float)
        sigmas    = np.zeros((n,),   dtype=float)
        is_big_orange = np.zeros((n,), dtype=bool)
        for i, lm in enumerate(kept):
            positions[i] = lm.position[:2]
            sigmas[i] = lm.sigma_xy
            is_big_orange[i] = lm.is_big_orange
        return cls(
            positions=positions,
            sigmas=sigmas,
            is_big_orange=is_big_orange,
        )

    # ----- query helpers -----

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def _kdtree(self):
        """Lazy KD-tree. Stored on the instance via object.__setattr__
        because the dataclass is frozen; we keep mutability for this
        derived structure only."""
        cached = getattr(self, "_kdtree_cache", None)
        if cached is not None:
            return cached
        if not _SCIPY_OK or len(self) == 0:
            return None
        tree = cKDTree(self.positions)
        object.__setattr__(self, "_kdtree_cache", tree)
        return tree

    def query_nearest(
        self,
        world_xy: np.ndarray,
        *,
        max_distance: float = float("inf"),
    ) -> tuple[int, float]:
        """Return (landmark_index, distance) of the closest landmark
        to `world_xy`. Returns (-1, +inf) when the map is empty or
        no landmark is within `max_distance`."""
        if len(self) == 0:
            return -1, float("inf")
        tree = self._kdtree()
        if tree is not None:
            d, i = tree.query(world_xy, k=1, distance_upper_bound=max_distance)
            if not np.isfinite(d) or i >= len(self):
                return -1, float("inf")
            return int(i), float(d)
        # Brute fallback when scipy isn't importable.
        diff = self.positions - np.asarray(world_xy)[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        idx = int(np.argmin(d2))
        d = float(np.sqrt(d2[idx]))
        if d > max_distance:
            return -1, float("inf")
        return idx, d

    def query_within_radius(
        self,
        world_xy: np.ndarray,
        radius_m: float,
    ) -> list[int]:
        """Return indices of all landmarks within `radius_m` of
        `world_xy`. Order undefined."""
        if len(self) == 0:
            return []
        tree = self._kdtree()
        if tree is not None:
            return list(map(int, tree.query_ball_point(world_xy, r=radius_m)))
        # Brute fallback.
        diff = self.positions - np.asarray(world_xy)[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        return [int(i) for i, dist2 in enumerate(d2) if dist2 <= radius_m * radius_m]

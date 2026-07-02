"""Two-phase parametric cone fit used by the LiDAR cone-detection pipeline.

Phase 1: scipy.optimize.minimize searches the apex (a, b) at fixed
(c=5.5, d=0.35). Phase 2: linear least-squares solves for (c, d) given
the apex from phase 1. Splitting the four-parameter fit into two stages
makes the optimiser well-behaved even on the small (3–10 point) clusters
that mid-range cones produce.
"""
from scipy.optimize import minimize
import numba
import numpy as np


@numba.njit(cache=True)
def cone_model(params, x, y):
    """Generalised cone surface: z(x, y) = d - c·sqrt((x-a)² + (y-b)²)."""
    a, b, c, d = params
    return d - c * np.sqrt((x - a) ** 2 + (y - b) ** 2)


@numba.njit(cache=True)
def objective_function_v2(params, x, y, z):
    """Phase-1 cost. Searches over (a, b) at fixed c, d."""
    a, b = params
    # Fixed c chosen empirically. Tried c=2.78 (IFS-08 real-cone slope)
    # on 2026-04-29 — counter-intuitively worse: the cost surface is
    # flatter when c matches the data, so phase 1 didn't move from the
    # (mean_x, mean_y) init, and phase 2 then recovered shallow c on
    # noisy data. c=5.5 forces phase 1 to walk the apex toward the
    # cluster's peak (steep cone → sharper minimum). The downstream
    # validation in cone_detection.py catches the slightly-too-shallow
    # phase-2 c via the centroid-fallback path.
    c = 5.5
    d = 0.35
    z_pred = cone_model([a, b, c, d], x, y)
    return np.mean((z - z_pred) ** 2)


@numba.njit(cache=True)
def get_gammas(x, y, a, b):
    return np.sqrt((x - a) ** 2 + (y - b) ** 2)


@numba.njit(cache=True)
def lst_sqrs_fit(X, y):
    # X.T is a non-contiguous transpose view of a C-contiguous X; using
    # it directly in `@` triggers Numba's slow path and emits a
    # NumbaPerformanceWarning. Materialise once with ascontiguousarray.
    Xt = np.ascontiguousarray(X.T)
    return np.linalg.inv(Xt @ X) @ Xt @ y


def cone_fit_2params(data, solver="L-BFGS-B"):
    """Two-phase cone fit. Returns (a, b, c, d) — apex (a, b, d), slope c."""
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    # Initialise apex at the cluster centroid.
    a = np.mean(x)
    b = np.mean(y)
    try:
        result = minimize(
            objective_function_v2, [a, b], args=(x, y, z), method=solver,
        )
        a, b = result.x
        gamma = get_gammas(x, y, a, b)
        M = np.vstack((np.ones(len(data)), gamma)).T
        d, c = lst_sqrs_fit(M, z.T)
        return a, b, -c, d
    except Exception as e:
        print(e)
        return 0, 0, 0, 0

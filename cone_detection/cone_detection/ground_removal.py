"""Cone-equation helpers used by the cone-fitting solvers.

The file used to be named for its original responsibility (RANSAC-based
ground-plane removal) but the live ground-removal path is now in
`ransac.py:ransac2`, called from `cone_detection.py:clustering_separation_rt`.
What survives here is just the parametric cone equation that
`two_phase_cone_detection.cone_fit_2params` and the solver-warmup path
in `cone_detection.warmup_numba_functions` need.

PR-B (chore/pipeline-extract-benchmarks) deleted ~480 LOC of dev-only
helpers: alternative cone fits (`ls_cone_fit`, `cone_fit`), benchmark
harnesses (`benchmark`, `performance_test`, `clustering_benchmark`),
visualization helpers (`plot_cone_vs_data`, `generate_cone_plot_points`,
`plot_3d_points`, `plot_clustering`), file-based offline pipelines
(`final_cone_result_file`, `ground_removal`, `read_lidar_data`), and
their now-unused imports (matplotlib, timeit, several sklearn cluster
backends, scipy.optimize.minimize/least_squares). None were called
from any production node, the launch files, or the runtime cone
detection path. Git history preserves them if they're ever wanted back.
"""
import numpy as np
import numba


@numba.njit
def cone_model(params, x, y):
    """Generalized cone surface z(x, y).

    Args:
        params: [a, b, c, d] — apex (a, b, d), slope c.
        x, y:   point coordinates.

    Returns:
        Predicted z at each (x, y).
    """
    a, b, c, d = params
    return d - c * np.sqrt((x - a) ** 2 + (y - b) ** 2)


@numba.njit
def objective_function(params, x, y, z):
    """Mean squared residual between observed z and `cone_model`(params, x, y).

    Used by scipy.optimize.minimize in `cone_fit_2params`.
    """
    z_pred = cone_model(params, x, y)
    return np.mean((z - z_pred) ** 2)

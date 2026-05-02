"""
========================
cone_detection.py (v1.1)
========================

Elaborado por Sergio Jiménez para el ISC
Permite detectar las posiciones de los conos dados los
datos de un LIDAR. Este archivo tiene distintos benchmarks, 
ademas de permitir visualizar la comparacion entre lo predicho 
y los datos

Novedades 1.1:
- Añadida rect2polars
"""

from slam.ransac import ransac, ransac2
from slam.ground_removal import cone_fit, read_lidar_data, clustering_separation
from slam.rotaciones import vectors2matrix
from slam.two_phase_cone_detection import *
import numpy as np
from sklearn.cluster import DBSCAN
import cProfile
import time
import matplotlib.pyplot as plt
import timeit

# import numba
import math


def final_cone_result_rt(data, model=DBSCAN, debug_counters=None):
    """
    Permite sacar los conos a partir de los datos del lidar. Version
    para Real Time. Aplica RANSAC, Clustering y el ajuste de cono. Esta hecha
    para solo hacer una medida del lidar de una vez

    Args:
        data (np.ndarray): los datos del lidar
        model (sklearn.model, optional): modelo de clustering empleado. Sirve una implementacion
                                         propia si es una clase con metodo fit_transform. Defaults to DBSCAN.
        debug_counters (dict, optional): if provided, gets populated with
            per-filter-stage cluster counts for diagnostics. Keys:
            n_input_points, n_clusters, after_min_pts, after_height_gate,
            after_fit_or_centroid, accepted, far_dropped, fit_used, centroid_used.

    Returns:
        float, float: las posiciones x e y de la punta del cono (a y b de los parametros del cono)
    """

    if debug_counters is not None:
        debug_counters["n_input_points"] = len(data)
    if len(data) == 0:
        return []
    labels, clean_data, def_coefs = clustering_separation_rt(data, model)
    if len(labels) == 0:
        return []
    separated_data = [
        np.array(clean_data[labels == label]) for label in np.unique(labels)
    ]
    if debug_counters is not None:
        debug_counters["n_clusters"] = len(separated_data)
        debug_counters["after_min_pts"] = 0
        debug_counters["after_height_gate"] = 0
        debug_counters["after_fit_or_centroid"] = 0
        debug_counters["accepted"] = 0
        debug_counters["far_dropped"] = 0
        debug_counters["fit_used"] = 0
        debug_counters["centroid_used"] = 0
    cone_positions = []
    for cone in separated_data:
        # Accept clusters with ≥ 2 points (was > 3). The previous floor
        # was rejecting half of all DBSCAN clusters per scan because far
        # cones (15–25 m) only return 2–3 points on the Hesai 128 ch:
        # at 20 m a small FS cone subtends ~0.7° H × 1.8° V and at our
        # 0.77° H step / 0.14° V step that's ~2–3 LiDAR returns total.
        # The downstream height + fit/centroid gates already do real
        # validation; this min-points floor was double-filtering.
        if len(cone) > 2:
            if debug_counters is not None:
                debug_counters["after_min_pts"] += 1
            v = np.array([0, 0, -1 * def_coefs[0]])
            w = np.array(def_coefs[1:])
            lidar_distance_to_floor = np.dot(v, w) / np.linalg.norm(w)
            clean_cone = cone[cone[:, 2] > 0.04 + lidar_distance_to_floor]
            if len(clean_cone) == 0:
                continue

            # Quick cluster-shape sanity gate BEFORE the L-BFGS-B fit.
            # Real cones have height in 0.10–0.55 m and a non-trivial
            # vertical span. This catches near-flat ground patches,
            # thin reflective stripes etc. so we don't spend solver
            # time on them and don't fall back to a meaningless centroid.
            cluster_height = float(clean_cone[:, 2].max() - clean_cone[:, 2].min())
            if cluster_height < 0.08 or cluster_height > 0.70:
                continue
            if debug_counters is not None:
                debug_counters["after_height_gate"] += 1

            params = cone_fit_2params(clean_cone, solver="SLSQP")

            # PRIMARY: parametric fit converged in the IFS-08 cone-
            # shape envelope (c ≈ 2.78 small, 2.86 big-orange).
            # FALLBACK: cluster has cone-like geometry (height,
            # n_points, range-aware point-count) but the fit landed
            # outside the envelope — common with 3 cm RangeNoiseStd
            # which fuzzes the L-BFGS-B gradient enough that phase 2
            # recovers a shallow c (1.5–2.5) on real cones. The
            # cluster centroid is a less-precise but unbiased estimate
            # in that case; SLAM weights it ~6× lower via 2.5× σ.
            fit_ok = (np.isfinite(params).all()
                      and 2.5 <  params[2] < 9.0
                      and 0.10 < params[3] < 0.60)

            # Strong geometric pre-validation for centroid fallback.
            # Conservative — better to lose a borderline cluster than
            # feed a ground-patch false positive (which we measured
            # earlier today: bare centroid fallback triggered earlier
            # cascade because of biased ground-patch positions).
            n_pts_pre = len(clean_cone)
            range_m_pre = float(np.hypot(
                float(clean_cone[:, 0].mean()),
                float(clean_cone[:, 1].mean())))
            # AMZ §3.2 / MUR rule: expected_points(d) ∝ 1/d² for a
            # spinning LiDAR. At our 200k pts/s × 10 Hz × 128 ch the
            # constant is ~250: small cone at 5 m → 10 pts, at 25 m →
            # 0.4 pts. Require ≥ max(2, 0.5×expected) so close-range
            # clusters get a stricter test than far-range ones; the
            # floor of 2 (was 6) lets sparse far-range clusters
            # through the centroid fallback. The 6-point floor was
            # blocking every cone past ~10 m from passing centroid_ok,
            # leaving fit_ok as the only path — and the L-BFGS-B fit
            # is unreliable on 2–3 noisy points, so most far cones
            # got dropped entirely.
            expected_pts = 250.0 / max(range_m_pre ** 2, 1.0)
            min_pts_required = max(2, int(0.5 * expected_pts))
            centroid_ok = (
                n_pts_pre >= min_pts_required
                and 0.20 <= cluster_height <= 0.55
                and range_m_pre <= 20.0    # only fall back at close
                                           # range where centroid bias
                                           # is small
            )

            if fit_ok:
                a_xy = float(params[0])
                b_xy = float(params[1])
                fallback_used = False
                if debug_counters is not None:
                    debug_counters["fit_used"] += 1
            elif centroid_ok:
                a_xy = float(clean_cone[:, 0].mean())
                b_xy = float(clean_cone[:, 1].mean())
                fallback_used = True
                if debug_counters is not None:
                    debug_counters["centroid_used"] += 1
            else:
                if debug_counters is not None:
                    if range_m_pre > 20.0:
                        debug_counters["far_dropped"] += 1
                continue
            if debug_counters is not None:
                debug_counters["after_fit_or_centroid"] += 1
                debug_counters["accepted"] += 1

            # The original bounds are still the trust signal — fitted
            # cones get the precise apex position; centroid-fallback
            # cones get the cluster mean which has more variance. We
            # report this to SLAM via σ_xy below.
            if True:
                # Cluster height (z-span, metres) — used downstream to separate
                # big-orange cones (datasheet 505 mm) from small blue/yellow/
                # small-orange cones (325 mm). Computed on the ground-removed
                # cluster so the RANSAC-plane clipping doesn't truncate it.
                height = (
                    float(clean_cone[:, 2].max() - clean_cone[:, 2].min())
                    if len(clean_cone) > 0
                    else 0.0
                )
                # Per-cone position uncertainty (σ_xy, metres). Centroid
                # variance scales as σ_ray / sqrt(N) and σ_ray itself
                # grows with range (bigger angular footprint per beam).
                # The 1.5× inflation factor was added on 2026-04-28
                # after a full bag diagnostic on trackA_manual_001602
                # measured per-observation residuals to the CSV cone
                # map and found the un-inflated formula was ~1.5×
                # tighter than the actual cone-detection error (median
                # 0.15 m residual at 5–25 m vs ~0.10 m predicted).
                # Reporting σ tighter than truth makes the SLAM graph
                # over-trust each observation and amplifies pose
                # corrections that are not justified by the data.
                # Reference cluster size is ~10 points (a healthy 8 m
                # cone with the Hesai ATX 128 ch); fewer points loosen
                # σ, more tighten it.
                range_m = float(np.hypot(a_xy, b_xy))
                n_pts = len(clean_cone)
                base_sigma = 0.05 + 0.005 * range_m
                sigma_xy = 1.5 * base_sigma / math.sqrt(max(1, n_pts) / 10.0)
                # Centroid fallback gets 2.5× σ — gives SLAM ~16% the
                # weight of a fitted observation per cone (1/2.5²),
                # which lets fitted cones dominate the optimizer's
                # corrections while still letting fallback cones
                # contribute when nothing else is in view.
                if fallback_used:
                    sigma_xy *= 2.5
                cone_positions.append((a_xy, b_xy, cluster_height, sigma_xy))
    return cone_positions


# @numba.njit
def rect2polars(x, y):

    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


def clustering_separation_rt(data, model):
    """
    Realiza la separación en clusters mediante un modelo de clustering. Primero hace ransac
    (se podría cambiar por cualquiera de las otras implementaciones), deshace las posibles rotaciones
    de los datos por el movimiento del coche y despues hace el clustering. Pensada para RT

    Args:
        data (str): ruta a los datos
        model (sklearn_model): el modelo de clustering de sklearn. Podría usarse una implementación propia si
                               se implementa como una clase con un metodo fit_predict que hace el ajuste y devuelve
                               las predicciones


    Returns:
        np.ndarray, np.ndarray, np.ndarray: las etiquetas para saber a que cluster pertenece cada cono,
                                            los datos con la correccion de rotacion y los coeficientes
                                            del plano sacado por ransac
    """
    A = np.c_[np.ones(data.shape[0]), data]
    inliers, def_coefs = ransac2(A, prob=0.9999, threshold=0.05)
    # COrreccion de rotacion (transformamos el vector normal al plano en un vector vertical, solo componente z)
    k = np.zeros(data.shape[1])
    k[-1] = 1

    outliers = np.ones(data.shape[0], dtype=bool)
    outliers[inliers] = False
    data = (data @ vectors2matrix(k, def_coefs[1:] / np.linalg.norm(def_coefs[1:])))[
        outliers
    ]
    # OTRAS OPCIONES DE MODELOS (por ahora he puesto DBSCAN)
    # clust_model = model()
    # clust_model = AgglomerativeClustering(
    #     n_clusters=None,
    #     linkage="ward",
    #     compute_full_tree=True,
    #     distance_threshold=0.5,
    # )
    if len(data) == 0:
        return np.array([]), data, def_coefs
    clust_model = DBSCAN(eps=0.3, min_samples=2)
    labels = clust_model.fit_predict(data)

    return labels, data, def_coefs


def profiling_cone_detection():
    """
    Hace profiling a la funcion de sacar los conos para saber que funciones
    son las que tardan mas
    """
    cProfile.run("benchmark_cone_detection(data)")


def solvers_benchmark(data, solver, model=DBSCAN):
    """
    Hace el benchmark para 1 solver en concreto

    Args:
        data (np.ndarray): los datos
        solver (str): el solver que se debe usar
        model (sklearn.model, optional): Modelo de clustering. Defaults to DBSCAN.
    """
    labels, clean_data, def_coefs = clustering_separation_rt(data, model)
    separated_data = [
        np.array(clean_data[labels == label]) for label in np.unique(labels)
    ]
    for cone in separated_data:

        if len(cone) > 3:
            v = np.array([0, 0, -1 * def_coefs[0]])
            w = np.array(def_coefs[1:])
            lidar_distance_to_floor = np.dot(v, w) / np.linalg.norm(w)
            # cone_positions.append((np.mean(cone[:, 0]), np.mean(cone[:, 1])))
            clean_cone = cone[cone[:, 2] > 0.05 + lidar_distance_to_floor]
            if len(clean_cone) > 3:
                cone_fit(clean_cone, solver)


def warmup_numba_functions():
    import os
    import numpy as np

    if os.environ.get("NUMBA_DISABLE_JIT", "0") == "1":
        return

    # Create dummy inputs representative of your actual data.
    dummy_x = np.linspace(0, 1, 10)
    dummy_y = np.linspace(0, 1, 10)
    dummy_z = np.linspace(0, 1, 10)

    # Warm up objective_function_v2
    dummy_params = np.array([0.0, 0.0])
    from slam.two_phase_cone_detection import (
        objective_function_v2,
        cone_model,
        get_gammas,
        lst_sqrs_fit,
    )
    from numba import float64

    signature_obj = (float64[:], float64[:], float64[:], float64[:])
    objective_function_v2.compile(signature_obj)
    objective_function_v2(dummy_params, dummy_x, dummy_y, dummy_z)

    # Warm up cone_model
    dummy_params_full = np.array([0.0, 0.0, 5.5, 0.35])
    cone_model(dummy_params_full, dummy_x, dummy_y)

    # Warm up get_gammas
    get_gammas(dummy_x, dummy_y, 0.0, 0.0)

    # Warm up lst_sqrs_fit. Create a dummy design matrix M.
    M = np.vstack((np.ones(10), np.sqrt((dummy_x - 0) ** 2 + (dummy_y - 0) ** 2))).T
    lst_sqrs_fit(M, dummy_z)


def best_solver(data):
    """
    Para benchmarkear los solvers de scipy.minimize

    Args:
        data (np.ndarray): los datos
    """
    solvers = [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        # "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        # "dogleg",
        "trust-constr",
        # "trust-ncg",
        # "trust-exact",
        # "trust-krylov",
    ]
    for solver in solvers:
        execution_time = timeit.timeit(
            lambda: solvers_benchmark(data, solver),
            globals=globals(),
            number=1,
        )

        print(f"Execution time {solver}: {execution_time} seconds")


def benchmark_cone_detection(data):
    """
    Ejecuta el cone detection en todos los datos

    Args:
        data (np.ndarray): los datos
    """
    for d in data:
        final_cone_result_rt(d)


def benchmark_process(data):
    """
    Calcula cuanto tiempo se tarda en todo el dataset

    Args:
        data (np.ndarray): los datos
    """
    execution_time = timeit.timeit(
        lambda: benchmark_cone_detection(data),
        globals=globals(),
        number=1,
    )

    print(f"Execution time: {execution_time} seconds")
    print(f"{len(data)} data points")
    print(f"{execution_time/len(data)} mean time")
    print(f"{len(data)/execution_time} mean fps")


def compare_data_to_processed(data):
    """
    Permite ver (en 2D) las posiciones predichas para los conos
    y los clusters separados por el lidar para unos datos concretos

    Args:
        data (np.ndarray): los datos
    """
    for d in data:
        # final_cone_result_rt now returns (x, y, height); discard height here.
        results = final_cone_result_rt(d)
        if not results:
            continue
        x = [r[0] for r in results]
        y = [r[1] for r in results]
        plt.scatter(x, y, marker="*")
        clustering_separation(d, plot=True)


if __name__ == "__main__":
    data = read_lidar_data("puntos_lidar.txt")

    final_cone_result_rt(data[0])
    profiling_cone_detection()
    benchmark_process(data)
    # compare_data_to_processed(data)

    # best_solver(data)

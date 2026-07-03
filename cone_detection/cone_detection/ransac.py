"""Numba-compiled RANSAC plane fit, used for ground-plane removal.

The live caller is `cone_detection.clustering_separation_rt`, which fits
the LiDAR ground plane on every scan before clustering the outliers
into cones. Tested for 2D and 3D data; extending to higher dimensions
would need changes to the cross-product step in the iteration loop.
"""
import numpy as np
import numba


@numba.jit(nopython=True)
def ransac2(
    data,
    m=3,
    prob=0.999,
    threshold=None,
    max_iter=200,
    iter_subsample_max=5000,
):
    """RANSAC plane fit. Returns (inlier_mask, plane coefficients).

    Args:
        data: (N, dim+1) — already augmented with a leading column of 1s
            for the bias term; numba doesn't play nicely with the
            np.c_/np.stack we'd otherwise use to add it on the fly.
        m: number of points that define a candidate plane.
        prob: target probability that the chosen subset is outlier-free
            (refines the iteration budget after each improving sample).
        threshold: inlier residual threshold. If None, set to the
            median absolute deviation of the last column — same heuristic
            sklearn's RANSACRegressor uses, with absolute (not squared)
            error so it's robust when the last dim isn't the largest-
            variance one.
        max_iter: hard upper bound on iterations. Degenerate scans
            with no consensus fall through cleanly at this cap.
        iter_subsample_max: cap on points used in the RANSAC iteration
            loop (#247). Consensus scoring runs on at most this many
            points; the final inlier mask always uses the full cloud.
            Set to 0 or negative to disable subsampling (profiling only).

    Returns:
        np.where-style index of inlier points in the full input cloud,
        and the plane coefficients [bias, n_x, n_y, ...] (last component
        is always positive — flipped at the end if necessary).
    """
    if not threshold:
        threshold = np.median(np.abs(data[:, -1] - np.median(data[:, -1])))
    A = data
    # Hoist the per-iteration loop-invariant slices into contiguous
    # arrays. `A[:, :-1]` is a strided (Fortran-flavoured) view of a
    # C-contiguous A; passing it directly to `.dot(...)` inside Numba
    # falls back to a slow path with a NumbaPerformanceWarning.
    # At 174 k pts/scan the slow path costs ~30-60 ms, enough to push
    # Cone_Detection CPU-bound. Materialising the slice once removes
    # the per-iteration overhead.
    A_xy_full = np.ascontiguousarray(A[:, :-1])
    A_z_full = np.ascontiguousarray(A[:, -1])
    data_size = len(data)
    # Subsample for the iteration loop (#247). RANSAC's consensus
    # probability depends on the inlier *ratio*, not absolute count —
    # running max_iter iterations against 5 000 random points finds
    # the same ground plane as the full ~95 k LiDAR cloud at ~19× the
    # per-iteration cost. Each iteration's `A_xy.dot(...)` is BLAS-
    # parallel and on the full cloud was sustaining 8-10 cores at 90%.
    # The final inlier set is computed against the FULL cloud in the
    # return statement so callers see identical outlier indices.
    n_sub = int(iter_subsample_max)
    if n_sub <= 0:
        n_sub = data_size
    if data_size > n_sub:
        sub_idx = np.random.choice(data_size, n_sub, replace=False)
        A_xy = np.ascontiguousarray(A_xy_full[sub_idx])
        A_z = np.ascontiguousarray(A_z_full[sub_idx])
        iter_data_size = n_sub
    else:
        A_xy = A_xy_full
        A_z = A_z_full
        iter_data_size = data_size
    k = float(max_iter)
    support = 0
    iters = 0

    # Sentinel z-up plane so def_coefs is never None even on a
    # degenerate cloud where no random sample beats support=0. The
    # previous version initialised def_coefs to None and bailed at
    # iters>50 to avoid hanging — but if the loop exited that way, the
    # subsequent `def_coefs[-1] < 0` check crashed.
    def_coefs = np.array([0.0, 0.0, 0.0, 1.0])

    while iters < k and iters < max_iter:
        # Sample m anchors from the FULL cloud — the iteration subsample
        # is only used to score consensus cheaply, not to constrain
        # which points can define a candidate plane.
        points_ind = np.random.choice(data_size, m, replace=False)
        points = A[points_ind]
        # Build the candidate plane via cross product (numba doesn't
        # JIT np.linalg.svd cleanly).
        vectors = points[1:][:, 1:] - points[0][1:]
        if m > 2:
            normal_vector = np.cross(vectors[0], vectors[1])
        else:
            normal_vector = np.array([-1 * vectors[0][1], vectors[0][0]])
        normal_vector /= np.linalg.norm(normal_vector)
        bias = -np.dot(normal_vector, points[0][1:])
        coefs = np.array([bias] + list(normal_vector))

        support_aux = 0
        for i in A_xy.dot((coefs[:-1] / (-1 * coefs[-1]))) - A_z:
            if abs(i) < threshold:
                support_aux += 1
        support_aux -= m
        if support_aux > support:
            support = support_aux
            def_coefs = coefs
            inlier_ratio = support / iter_data_size
            # Refine the iteration budget toward the standard RANSAC
            # bound. Guard the perfect-inlier case (log(0) = -inf) and
            # never let k climb above max_iter — degenerate scans must
            # exit cleanly instead of running for thousands of iters.
            if 0.0 < inlier_ratio < 1.0:
                k_new = np.log(1 - prob) / np.log(1 - inlier_ratio ** m)
                if k_new < k:
                    k = k_new
        iters += 1
    if def_coefs[-1] < 0:
        def_coefs = -1 * def_coefs
    # Final inlier mask over the FULL cloud — callers expect indices
    # into `data`, not the iteration subsample.
    return (
        np.where(
            np.abs(A_xy_full.dot(def_coefs[:-1] / (-1 * def_coefs[-1])) - A_z_full)
            < threshold
        ),
        def_coefs,
    )

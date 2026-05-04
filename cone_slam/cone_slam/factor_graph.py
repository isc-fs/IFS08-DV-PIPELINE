"""GTSAM iSAM2 wrapper for the cone-graph SLAM node.

PR A scope: pose nodes (X) + velocity nodes (V) + bias nodes (B), with
PriorFactors anchoring x_0/v_0/b_0 and ImuFactors between consecutive
poses. No cone landmarks yet — those land in PR B.

Symbols:
  X(k) = gtsam.symbol('x', k)   # Pose3 at scan k
  V(k) = gtsam.symbol('v', k)   # Vector3 velocity in nav frame at scan k
  B(k) = gtsam.symbol('b', k)   # imuBias.ConstantBias at scan k
  L(id) = gtsam.symbol('l', id) # Point3 cone landmark — PR B+

Why Pose3 internally even though FS cars don't fly: gtsam.ImuFactor is
defined for Pose3 + Vector3 velocity (per Forster preintegration).
Forcing Pose2 means hand-integrating IMU and losing efficient covariance
propagation. We accept the extra DOF and project to 2D when publishing
TF (zero out z, roll, pitch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import gtsam
from gtsam.symbol_shorthand import X, V, B, L  # type: ignore[attr-defined]


# Anchor prior — the very first pose is locked at world origin with
# tight covariance. After initial calibration, the bias prior is loose
# enough to let the optimizer refine on subsequent measurements.
PRIOR_POSE_SIGMAS = np.array([0.001, 0.001, 0.001,  # roll/pitch/yaw rad
                              0.001, 0.001, 0.001])  # x/y/z m
PRIOR_VEL_SIGMAS = np.array([0.01, 0.01, 0.01])     # m/s — car is stationary at init
PRIOR_BIAS_SIGMAS = np.concatenate([
    np.full(3, 0.05),    # accel bias m/s² — generous; calibration mean is approximate
    np.full(3, 0.001),   # gyro bias rad/s
])

# Bias-evolution noise for BetweenFactor<imuBias>. Allow biases to
# random-walk between scan nodes — small but non-zero so iSAM2 can
# evolve them as more LiDAR data constrains the trajectory.
#
# Tightened 10× on 2026-04-28 (1e-3→1e-4 accel, 1e-4→1e-5 gyro) after
# a per-50-step bias dump on trackA_manual_001602 showed the optimizer
# was absorbing pose-prediction error INTO the bias estimate as the car
# got far from spawn — gyro_z drifted to +0.011 rad/s by step 700,
# accel components drifted 50 % from the calibration value, and the
# resulting wrong-IMU integration was the actual cascade trigger. Real
# BMI088 bias varies over minutes, not seconds; the looser sigmas were
# letting the graph treat genuine pose-prediction error as bias
# correction. The 10× tightening preserves the optimizer's ability to
# track real slow drift while choking off this feedback loop.
BIAS_RW_SIGMAS = np.concatenate([
    np.full(3, 1e-4),    # accel bias m/s² per scan
    np.full(3, 1e-5),    # gyro bias rad/s per scan
])


@dataclass
class ScanResult:
    """What the node gets back after each iSAM2 update."""
    pose: gtsam.Pose3
    velocity: np.ndarray   # (3,) m/s in nav frame
    bias: gtsam.imuBias.ConstantBias


class FactorGraph:
    """Thin owner of an iSAM2 instance + symbol counter + the rolling
    'new' pieces that get handed to update() each scan.
    """

    def __init__(self, isam2_relinearize_threshold: float = 0.1) -> None:
        params = gtsam.ISAM2Params()
        # GTSAM 4.2.x Python bindings are inconsistent: setRelinearizeThreshold
        # exists as a method but relinearizeSkip is exposed as a property
        # (with no setter). Use whichever style each one supports.
        params.setRelinearizeThreshold(isam2_relinearize_threshold)
        # Re-linearize at most once every 10 update calls. Too aggressive
        # (1) lets iSAM2 freely rotate the entire trajectory each scan
        # to chase the latest observations, which is what was driving
        # the cone-only late-drive divergence. The 10× value is a
        # standard "less aggressive" GTSAM tuning that lets earlier
        # graph state act as a soft anchor on later updates.
        #
        # We tried skip=1 on 2026-04-28 after the bias-RW tightening,
        # hypothesizing the bias-absorption feedback loop had been the
        # only thing relying on stale linearizations. Wrong: skip=1
        # cascaded earlier (t=90s vs 92s) AND made the bias jump to
        # (-0.30, -0.76, +0.03) at step 700 (vs iter 7's tight
        # (-0.08, -0.04, -0.01)). The relinearizeSkip=10 anchor is
        # load-bearing on cone-only scenes regardless of priors. Stay
        # at 10.
        params.relinearizeSkip = 10
        self._isam = gtsam.ISAM2(params)

        # Accumulators between update() calls. Cleared after each update.
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_values = gtsam.Values()

        self._k: int = 0  # scan index. 0 is the anchor.

    # ----- INIT (called once) ------------------------------------------------

    def initialize_anchor(
        self,
        initial_pose: gtsam.Pose3,
        initial_velocity: np.ndarray,
        initial_bias: gtsam.imuBias.ConstantBias,
    ) -> None:
        """Add x_0, v_0, b_0 with priors and run the first update.

        Called once after IMU calibration completes. Subsequent scan
        callbacks use add_imu_step().
        """
        if self._k != 0:
            raise RuntimeError("initialize_anchor() called more than once")

        self._new_values.insert(X(0), initial_pose)
        self._new_values.insert(V(0), initial_velocity)
        self._new_values.insert(B(0), initial_bias)

        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_POSE_SIGMAS)
        vel_noise = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_VEL_SIGMAS)
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_BIAS_SIGMAS)

        self._new_factors.add(
            gtsam.PriorFactorPose3(X(0), initial_pose, pose_noise))
        self._new_factors.add(
            gtsam.PriorFactorVector(V(0), initial_velocity, vel_noise))
        self._new_factors.add(
            gtsam.PriorFactorConstantBias(B(0), initial_bias, bias_noise))

        self._flush_update()

    # ----- pre-update accumulators (called by node before _flush_update) ---

    def stage_imu_factor(
        self,
        pim: gtsam.PreintegratedImuMeasurements,
        prev_result: ScanResult,
    ) -> None:
        """Append the (X, V, B) triple + ImuFactor for this scan to the
        accumulator. Doesn't run iSAM2 yet — caller batches multiple
        factor types (IMU + cone observations) before flush_update().
        """
        prev_k = self._k
        self._k += 1
        new_k = self._k

        nav_state = gtsam.NavState(prev_result.pose, prev_result.velocity)
        predicted = pim.predict(nav_state, prev_result.bias)

        self._new_values.insert(X(new_k), predicted.pose())
        self._new_values.insert(V(new_k), predicted.velocity())
        self._new_values.insert(B(new_k), prev_result.bias)

        self._new_factors.add(gtsam.ImuFactor(
            X(prev_k), V(prev_k),
            X(new_k), V(new_k),
            B(prev_k),
            pim,
        ))
        bias_rw_noise = gtsam.noiseModel.Diagonal.Sigmas(BIAS_RW_SIGMAS)
        self._new_factors.add(gtsam.BetweenFactorConstantBias(
            B(prev_k), B(new_k),
            gtsam.imuBias.ConstantBias(),
            bias_rw_noise,
        ))

    def stage_velocity_prior(
        self,
        v_body_long: float,
        predicted_yaw: float,
        sigma_long: float = 0.30,
        sigma_lat: float = 0.30,
        sigma_vert: float = 0.05,
    ) -> None:
        """Add a unary prior on V(self._k) constraining the velocity to
        a body-frame longitudinal value (e.g. derived from /motor_rpm)
        with looser lateral uncertainty (the non-holonomic constraint
        for a non-slipping wheeled vehicle) and tight vertical (flat
        track).

        The factor lives in the world frame, so we rotate both the mean
        and the covariance through the predicted yaw:

            v_world  = R_z(yaw) · (v_long, 0, 0)
            Σ_world  = R_z(yaw) · diag(σ_long², σ_lat², σ_vert²) · R_z(yaw)ᵀ

        Why this matters on cone-only SLAM: when a scan window has 0–1
        cones, the optimizer runs on IMU + bias factors only and is
        free to rotate the global frame to find a cheaper minimum.
        That's the cascade. Adding a velocity prior every scan keeps
        the IMU-integrated trajectory honest in scale and direction —
        catastrophic global rotations get expensive even when cones
        disappear, because they violate the velocity prior.

        Defaults (re-loosened on 2026-04-28 from the
        previous σ_long=0.15 / σ_lat=0.07 settings after a full bag
        diagnostic measured the actual residuals on trackA_manual_001602):
          σ_long 0.30 m/s — measured std of (rpm-derived speed) vs
                             (GT body-frame v_x) is 0.61 m/s before the
                             RPM_TO_MS scale fix; we expect roughly
                             half of that variance to come from the
                             scale error (now corrected) and half
                             from genuine wheel-slip + motion noise.
                             0.30 m/s is a safe upper bound.
          σ_lat  0.30 m/s — measured std of GT body-frame v_y during
                             motion is 0.26 m/s (the car DOES drift
                             laterally during turns even though it's
                             "non-holonomic"). The earlier 0.07 m/s
                             was 4× tighter than reality and was
                             penalizing legitimate turn-induced lateral
                             motion. 0.30 matches the data.
          σ_vert 0.05 m/s — flat track; vertical motion is sensor noise.
        """
        cy, sy = np.cos(predicted_yaw), np.sin(predicted_yaw)
        # World-frame mean: project body-frame (v_long, 0, 0) through yaw.
        v_world = np.array([v_body_long * cy, v_body_long * sy, 0.0])
        # World-frame covariance: rotate body-frame diag(σ²) by yaw.
        R = np.array([[cy, -sy, 0.0],
                      [sy,  cy, 0.0],
                      [0.0, 0.0, 1.0]])
        sigma_body = np.diag(np.array([sigma_long, sigma_lat, sigma_vert]) ** 2)
        cov_world = R @ sigma_body @ R.T
        noise = gtsam.noiseModel.Gaussian.Covariance(cov_world)
        self._new_factors.add(gtsam.PriorFactorVector(
            V(self._k), v_world, noise))

    def stage_new_landmark(
        self,
        landmark_id: int,
        initial_world_xyz: np.ndarray,
    ) -> None:
        """Insert a brand-new landmark variable.

        Called once per cone, the first time it's observed. Subsequent
        observations of the same cone use stage_cone_observation() with
        its existing id and add another factor between the current pose
        and that landmark.
        """
        self._new_values.insert(L(landmark_id),
                                gtsam.Point3(*initial_world_xyz))

    def stage_cone_observation(
        self,
        landmark_id: int,
        body_x: float,
        body_y: float,
        sigma_xy: float = -1.0,
    ) -> None:
        """Add a BearingRange observation between the current pose
        (X(self._k)) and the landmark.

        Uses BearingRangeFactor3D internally (Pose3 + Point3) with
        z-component pinned to 0 — the FS car drives on flat ground so
        the cone's z stays at the LiDAR mount height in the world
        frame, and the bearing's vertical component is uninformative.

        Wrapped in a Huber robust loss: a SINGLE bad data association
        in a cone-only environment can drag the entire optimizer to a
        wrong global rotation (the catastrophic-divergence mode we
        kept hitting on 2026-04-27). Huber caps the influence of
        residuals beyond k σ, so outliers stop contributing rather
        than dominating.

        Observation noise scales linearly with range: cluster point
        count ∝ 1/d² so the centroid's variance grows with distance
        (MUR's `num_expected_points(d)` rule, AMZ §3.2). Reporting a
        constant 0.20 m for every cone lies to iSAM2 — a 25 m cone
        with true ±1 m error and reported ±0.2 m gets ~25× the weight
        it deserves, which is what was driving the late-drive yaw
        snap on the cone_slam runs (2026-04-27).
        """
        bearing = gtsam.Unit3(np.array([body_x, body_y, 0.0]))
        range_m = float(np.hypot(body_x, body_y))

        # Prefer the detector's per-cone σ_xy (Hesai 128 ch centroid SE
        # ≈ σ_ray / sqrt(N), so cluster size matters as much as range).
        # When the detector reports a positive σ, treat it as the
        # radial-equivalent uncertainty and convert to a bearing sigma
        # via small-angle ≈ lateral_m / range_m. When the sentinel
        # (≤0) is passed, fall back to the legacy linear-in-range
        # formula — same behavior as before this wiring.
        if sigma_xy > 0.0:
            range_sigma = sigma_xy
            # Avoid div-by-zero on cones at ~0 m (shouldn't happen but
            # the math demands a guard); fall back to a generous bound.
            bearing_sigma = sigma_xy / range_m if range_m > 0.5 else 0.2
        else:
            range_sigma = 0.05 + 0.005 * range_m
            bearing_sigma = 0.02 + 0.001 * range_m

        # 3D bearing is parameterized as Unit3 (2 DoF tangent space) +
        # 1 DoF range = 3-vector residual. Sigmas are (bearing_x,
        # bearing_y, range) in tangent space units.
        sigmas = np.array([bearing_sigma, bearing_sigma, range_sigma])
        gaussian = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        # k=1.345 is the classic Huber tuning that gives 95% efficiency
        # under a Gaussian model — i.e. you pay almost nothing in the
        # nominal case but stop a wild outlier at 1.345σ.
        huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
        robust = gtsam.noiseModel.Robust.Create(huber, gaussian)

        # GTSAM 4.2 Python: BearingRangeFactor3D takes bearing and range
        # as separate args (not a wrapped BearingRange3D).
        self._new_factors.add(gtsam.BearingRangeFactor3D(
            X(self._k), L(landmark_id),
            bearing, range_m, robust,
        ))

    def commit(self) -> ScanResult:
        """Run iSAM2 with everything staged for this scan and return
        the latest pose/velocity/bias estimate at X(self._k).
        """
        return self._flush_update()

    def commit_with_pose_sanity_check(
        self,
        predicted_pose: gtsam.Pose3,
        max_pos_dev_m: float,
        max_yaw_dev_rad: float,
    ) -> "tuple[ScanResult, bool]":
        """Run iSAM2 with everything staged, then sanity-check the
        optimized pose at X(self._k) against an IMU-predicted pose.

        If the optimized pose deviates more than (`max_pos_dev_m`,
        `max_yaw_dev_rad`) from `predicted_pose`, a strong PriorFactor
        on X(self._k) is added at `predicted_pose` and iSAM2 is run
        again. The strong prior over-constrains pose to the IMU-predicted
        value, neutralizing the bad cone factor(s) that caused the
        snap. The bad factors stay in the graph; future iterations
        will relinearize against the corrected pose without snapping.

        Why a corrective prior instead of a true rollback (#273): iSAM2
        does support factor removal by index, but doing so cleanly
        across our staging mechanism would require tracking factor
        indices through every stage_* call. The strong prior achieves
        the same observable behavior (pose at this step matches IMU
        prediction) with one extra factor and one extra update call.

        Returns (result, was_corrected). `was_corrected=True` means the
        sanity check fired and the corrective prior was applied.
        """
        result = self._flush_update()

        pos_dev = float(np.linalg.norm(
            result.pose.translation() - predicted_pose.translation()))
        # Rotation deviation as the yaw of the relative rotation —
        # naturally wrapped to (-π, π].
        delta_rot = predicted_pose.rotation().between(result.pose.rotation())
        yaw_dev = abs(float(delta_rot.yaw()))

        if pos_dev <= max_pos_dev_m and yaw_dev <= max_yaw_dev_rad:
            return result, False

        # Excessive jump — apply strong prior at predicted pose.
        # Tight noise: 5 mm position, 0.3° rotation. Stronger than any
        # cone bearing-range factor we publish, so iSAM2 will pull the
        # pose to predicted_pose on this update.
        strong_sigmas = np.array([
            0.005, 0.005, 0.005,   # rotation rxx/ryy/rzz (rad)
            0.005, 0.005, 0.005,   # translation x/y/z (m)
        ])
        strong_noise = gtsam.noiseModel.Diagonal.Sigmas(strong_sigmas)
        self._new_factors.add(gtsam.PriorFactorPose3(
            X(self._k), predicted_pose, strong_noise))

        corrected = self._flush_update()
        return corrected, True

    def discard_staged(self) -> None:
        """Drop everything staged since the last commit and rewind
        self._k. Used when a pre-commit sanity check determines this
        scan should be skipped — for example when data association
        flips from steady to mostly-new (the cascade signature on
        cone-poor scenes after a sharp turn). Discarding leaves the
        graph in the same state it was in before stage_imu_factor was
        called, so the next scan retries from the previous committed
        pose estimate.

        NOTE: this only undoes the GRAPH side. The caller is
        responsible for not having mutated LandmarkDb yet — i.e.
        the DA-failure check must run BEFORE creating new landmark
        entries, otherwise discard_staged() leaves orphan landmark
        rows in the db that were never linked to a graph node.
        """
        self._new_factors.resize(0)
        self._new_values.clear()
        # Undo the stage_imu_factor increment so the next scan's
        # stage_imu_factor produces the same prev_k → new_k pair.
        self._k -= 1

    def landmark_position(self, landmark_id: int) -> Optional[np.ndarray]:
        """Return the latest world-frame Point3 estimate for a
        landmark, or None if iSAM2 hasn't merged it yet."""
        try:
            estimate = self._isam.calculateEstimate()
            return np.array(estimate.atPoint3(L(landmark_id)))
        except Exception:
            return None

    def pose_covariance(
        self, k: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Return the 6×6 marginal covariance of X(k) in Pose3 tangent
        space. Defaults to k = self._k - 1 — i.e. the most recently
        committed pose. The current step's pose hasn't been committed
        yet at DA time (we stage X(k) but call associate() before
        commit), so its marginal isn't available; the previous step is
        the closest valid query and a useful approximation for DA.

        Tangent ordering (per GTSAM convention): [ω_x, ω_y, ω_z, v_x,
        v_y, v_z] — rotation first, translation second. The 2D
        (yaw, x, y) sub-covariance is at indices [2, 3, 4].
        """
        target_k = (self._k - 1) if k is None else k
        if target_k < 0:
            return None
        try:
            return np.array(self._isam.marginalCovariance(X(target_k)))
        except Exception:
            return None

    def landmark_covariance(self, landmark_id: int) -> Optional[np.ndarray]:
        """Return the 3×3 marginal covariance of a landmark's world-
        frame position, or None if iSAM2 hasn't merged it yet (e.g. on
        the same scan it was first inserted) or the call throws.

        Used by Mahalanobis DA gating in data_association.py — combined
        with the observation σ to compute an information-aware match
        gate. Brand-new landmarks have very loose covariance, so the
        Mahalanobis gate is wide initially; as iSAM2 accumulates
        observations the covariance tightens and DA gets more
        discriminating automatically.
        """
        try:
            return np.array(self._isam.marginalCovariance(L(landmark_id)))
        except Exception:
            return None

    # ----- per-scan: add IMU factor, optimize -------------------------------

    def add_imu_step(
        self,
        pim: gtsam.PreintegratedImuMeasurements,
        prev_result: ScanResult,
    ) -> ScanResult:
        """Append a new (X, V, B) triple and the IMU factor connecting
        it to the previous step. Returns the updated estimate at the
        new step.
        """
        prev_k = self._k
        self._k += 1
        new_k = self._k

        # Predict the new state from the previous estimate + integrated
        # IMU. iSAM2 will refine this once we call update().
        nav_state = gtsam.NavState(prev_result.pose, prev_result.velocity)
        predicted = pim.predict(nav_state, prev_result.bias)

        self._new_values.insert(X(new_k), predicted.pose())
        self._new_values.insert(V(new_k), predicted.velocity())
        self._new_values.insert(B(new_k), prev_result.bias)

        # ImuFactor: x_prev, v_prev → x_new, v_new, with a single bias
        # node shared. We attach the same bias to both ends and add a
        # tiny BetweenFactor<imuBias> so the bias is allowed to evolve.
        self._new_factors.add(gtsam.ImuFactor(
            X(prev_k), V(prev_k),
            X(new_k), V(new_k),
            B(prev_k),
            pim,
        ))
        bias_rw_noise = gtsam.noiseModel.Diagonal.Sigmas(BIAS_RW_SIGMAS)
        self._new_factors.add(gtsam.BetweenFactorConstantBias(
            B(prev_k), B(new_k),
            gtsam.imuBias.ConstantBias(),  # zero-mean — bias drifts, doesn't jump
            bias_rw_noise,
        ))

        return self._flush_update()

    # ----- internals ---------------------------------------------------------

    def _flush_update(self) -> ScanResult:
        """Hand new_factors + new_values to iSAM2, read back the latest
        estimate at the current step.
        """
        self._isam.update(self._new_factors, self._new_values)
        # Optional second update for tighter convergence on big residuals.
        # GTSAM convention: 1-2 extra calls per step is normal.
        self._isam.update()

        # Reset accumulators.
        self._new_factors.resize(0)
        self._new_values.clear()

        estimate = self._isam.calculateEstimate()
        return ScanResult(
            pose=estimate.atPose3(X(self._k)),
            velocity=estimate.atVector(V(self._k)),
            bias=estimate.atConstantBias(B(self._k)),
        )

    @property
    def step(self) -> int:
        return self._k

    def latest(self) -> Optional[ScanResult]:
        if self._k < 0:
            return None
        estimate = self._isam.calculateEstimate()
        return ScanResult(
            pose=estimate.atPose3(X(self._k)),
            velocity=estimate.atVector(V(self._k)),
            bias=estimate.atConstantBias(B(self._k)),
        )

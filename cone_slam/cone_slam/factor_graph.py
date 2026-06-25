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

import time
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


def _cone_bearing_range(body_x: float, body_y: float, sigma_xy: float):
    """Build the (bearing, range, robust-noise) triple for one cone
    BearingRange observation.

    Shared by the incremental SLAM path (stage_cone_observation) and the
    frozen-map localization path (localize) so both use a byte-identical
    measurement model. Observation noise scales linearly with range
    (cluster point count ∝ 1/d², MUR §3.2) and is wrapped in a Huber
    robust loss (k=1.345, 95 % Gaussian efficiency) so a single bad data
    association can't drag the optimizer to a wrong global rotation. When
    the detector reports a positive σ_xy it's treated as the radial-
    equivalent uncertainty and converted to a bearing sigma via small-
    angle ≈ lateral_m / range_m; the ≤0 sentinel falls back to the legacy
    linear-in-range formula.
    """
    bearing = gtsam.Unit3(np.array([body_x, body_y, 0.0]))
    range_m = float(np.hypot(body_x, body_y))
    if sigma_xy > 0.0:
        range_sigma = sigma_xy
        bearing_sigma = sigma_xy / range_m if range_m > 0.5 else 0.2
    else:
        range_sigma = 0.05 + 0.005 * range_m
        bearing_sigma = 0.02 + 0.001 * range_m
    # Tangent-space sigmas: (bearing_x, bearing_y, range).
    gaussian = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([bearing_sigma, bearing_sigma, range_sigma]))
    huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
    robust = gtsam.noiseModel.Robust.Create(huber, gaussian)
    return bearing, range_m, robust


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

        # Full iSAM2 estimate, recomputed once per commit in
        # _flush_update and reused for every read this scan (tip pose/vel/
        # bias AND every landmark_position query). calculateEstimate() is
        # a full back-substitution over the whole graph; reading the tip
        # plus refreshing ~N landmarks used to trigger ~N+1 of them per
        # scan (O(landmarks × graph-size) — the proc-time blowup). One
        # solve/scan instead. None until the first commit.
        self._latest_estimate: Optional[gtsam.Values] = None

        # --- per-scan profiling (SLAM_PROF diagnostic) ----------------------
        # Wall time (ms) spent inside the two hot iSAM2 primitives, summed
        # over however many _flush_update calls a scan makes (1 normally, 2
        # when the pose-jump sanity check re-solves). Reset by reset_prof()
        # at scan entry; read by the node when it emits SLAM_PROF.
        self.prof_update_ms: float = 0.0    # both isam.update() calls
        self.prof_calcest_ms: float = 0.0   # calculateEstimate() back-sub
        self.prof_flushes: int = 0          # # of _flush_update this scan

    def reset_prof(self) -> None:
        """Zero the per-scan profiling accumulators. Call at scan entry."""
        self.prof_update_ms = 0.0
        self.prof_calcest_ms = 0.0
        self.prof_flushes = 0

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

    def stage_odom_between(
        self,
        between_pose: gtsam.Pose3,
        sigma_yaw_rad: float = 0.03,
        sigma_xy_m: float = 0.10,
        sigma_rp_rad: float = 0.05,
        sigma_z_m: float = 0.05,
    ) -> None:
        """Stage a BetweenFactor on X(k-1) → X(k) sourced from /odom's
        delta-pose over the scan window.

        /odom is the 9-state EKF's output. It already fuses IMU + RPM +
        steering with Coriolis-correct prediction, low-vx-gated steering
        correction, and online gyro-bias tracking. Each /odom message
        is a higher-quality pose estimate than what cone_graph_slam can
        compute internally from IMU + RPM alone (its only fallback when
        cone factors are skipped during cascade).

        Wiring /odom as a BetweenFactor turns the EKF into a soft
        constraint on SLAM's pose-to-pose motion. The advantage shows
        up exactly when SLAM needs it most: during cascade-skip windows
        in a curve, IMU-only predict produces 10 m / 94° pose jumps
        (bag autocross_track_20260404_013721_20260518_095215); /odom's
        per-scan drift is ~3 cm and ~0.1° over the same window.

        Default sigmas reflect the post-#534/#539 EKF's measured per-
        scan residuals to GT (yaw ~3.7° per 10 s = 0.4° per scan; xy
        ~5-10 cm per scan). σ_yaw=0.03 rad (~1.7°) is loose enough to
        not over-constrain SLAM relative to cone observations, tight
        enough to anchor when cones go away.

        The between_pose is in SE(3) and expresses the pose change
        from X(k-1) to X(k) in X(k-1)'s body frame — exactly what
        `prev_odom_pose.inverse().compose(current_odom_pose)` produces.
        """
        sigmas = np.array([
            sigma_rp_rad,   # roll
            sigma_rp_rad,   # pitch
            sigma_yaw_rad,  # yaw — THE constraint
            sigma_xy_m,     # tx
            sigma_xy_m,     # ty
            sigma_z_m,      # tz
        ])
        noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        self._new_factors.add(gtsam.BetweenFactorPose3(
            X(self._k - 1), X(self._k), between_pose, noise))

    def stage_odom_motion_step(
        self,
        between_pose: gtsam.Pose3,
        prev_result: ScanResult,
        world_velocity: np.ndarray,
        sigma_yaw_rad: float = 0.02,
        sigma_xy_m: float = 0.05,
        sigma_rp_rad: float = 0.05,
        sigma_z_m: float = 0.05,
        vel_sigma: float = 0.5,
    ) -> gtsam.Pose3:
        """EKF-as-motion-model: advance the graph by one scan using the
        /odom delta-pose as the PRIMARY motion constraint, with no IMU
        preintegration factor.

        This is the counterpart to stage_imu_factor for the
        ``motion_model='odom'`` configuration. Where stage_imu_factor
        adds an ImuFactor (and relies on the looser stage_odom_between
        as a soft backup), this method makes the 9-state EKF's per-scan
        delta the dominant pose-to-pose constraint and drops the IMU
        factor entirely — the EKF already fuses IMU + RPM + steering
        with Coriolis correction and online gyro-bias tracking, so a
        separate ImuFactor would double-count the IMU (the accelerometer
        and gyro enter the graph twice, through /odom and through the
        raw IMU factor, as if they were independent measurements) and
        re-introduce the degenerate-timestamp preintegration error that
        the EKF is immune to.

        What gets staged for X(k), V(k), B(k):
          * X(k) initialised at ``prev_pose ∘ between_pose`` and
            constrained by a TIGHT BetweenFactor on X(k-1)→X(k). Sigmas
            default tighter than stage_odom_between (yaw 0.02 rad ≈ 1.1°,
            xy 5 cm) because here /odom is the only motion source rather
            than a backstop — it must out-weight a noisy cone factor in
            cone-poor windows. Cone BearingRange factors still correct
            accumulated global drift; the EKF only owns short-horizon
            motion.
          * V(k) seeded from the EKF twist (``world_velocity``) and held
            by a loose unary prior. Without an IMU factor V(k) would
            otherwise be unconstrained (indeterminate). The prior keeps
            it well-posed and gives /slam/pose a sensible twist.
          * B(k) carried forward unchanged with a bias random-walk
            BetweenFactor. The bias is unused for motion in this mode
            but the B chain must stay connected back to the B(0) prior
            so iSAM2 doesn't see an indeterminate variable, and so the
            ScanResult bias field stays readable.

        Returns the predicted (pre-optimization) pose at X(k) so the
        caller can use it for data association and the pose-jump sanity
        check — the EKF prediction replaces pim.predict() in this mode.
        """
        prev_k = self._k
        self._k += 1
        new_k = self._k

        predicted_pose = prev_result.pose.compose(between_pose)
        self._new_values.insert(X(new_k), predicted_pose)
        self._new_values.insert(V(new_k), world_velocity)
        self._new_values.insert(B(new_k), prev_result.bias)

        sigmas = np.array([
            sigma_rp_rad,   # roll
            sigma_rp_rad,   # pitch
            sigma_yaw_rad,  # yaw — THE constraint
            sigma_xy_m,     # tx
            sigma_xy_m,     # ty
            sigma_z_m,      # tz
        ])
        noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        self._new_factors.add(gtsam.BetweenFactorPose3(
            X(prev_k), X(new_k), between_pose, noise))

        vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, vel_sigma)
        self._new_factors.add(gtsam.PriorFactorVector(
            V(new_k), world_velocity, vel_noise))

        bias_rw_noise = gtsam.noiseModel.Diagonal.Sigmas(BIAS_RW_SIGMAS)
        self._new_factors.add(gtsam.BetweenFactorConstantBias(
            B(prev_k), B(new_k),
            gtsam.imuBias.ConstantBias(),
            bias_rw_noise,
        ))
        return predicted_pose

    def stage_steering_between(
        self,
        dt: float,
        v_body_long: float,
        steering_rad: float,
        prev_pose: gtsam.Pose3,
        imu_predicted_dyaw_rad: float,
        wheelbase_m: float = 1.570,
        slip_threshold_rad_s: float = 0.30,
        sigma_yaw_rad: float = 0.05,
        sigma_xy_m: float = 1.0,
    ) -> str:
        """Stage a kinematic-bicycle BetweenFactor between X(k-1) and X(k).

        This is the steering-sensor channel into the factor graph. The
        kinematic-bicycle model says ω = (v_x / L) · tan(δ). Integrated
        over the scan window dt, this predicts an expected relative pose
        change with the chassis turning at that rate while moving v_x
        body-x. A BetweenFactor encodes that expectation and constrains
        the optimizer to follow steering+RPM during cone-poor windows
        (the exact moment the gyro-only IMU predict was producing 10 m /
        94° pose jumps in bag autocross_track_20260404_013721_20260518_095215).

        The factor is slip-gated: if the IMU's measured Δyaw over dt
        disagrees with the steering prediction by more than
        slip_threshold_rad_s, the chassis is sliding (the kinematic-
        bicycle model breaks down) and the factor is NOT staged. Caller
        gets back the gate decision in the return string for logging.

        Returned strings:
          "staged"     — factor added.
          "stopped"    — vehicle essentially stationary (|v| < 0.1).
          "slip"       — slip-gate fired; factor skipped.

        Sigmas are deliberately loose on translation (~1 m) so this
        factor only meaningfully constrains *yaw rate*. Translation
        comes from RPM + IMU; over-tightening here would conflict.
        Yaw σ 0.05 rad ≈ 2.9° per scan window — about 2× the kinematic
        bicycle's measured residual on FS-DV at speed.
        """
        if abs(v_body_long) < 0.1:
            return "stopped"

        # NOTE on sign: we tried negating omega_pred in commit 7037294
        # (bag _104623 showed 80 % slip rate during cornering with same-
        # magnitude opposite-sign mismatch between δ and IMU dyaw). It
        # DID get the factor staging on more samples, but bag _105724
        # showed the car turning the wrong direction at a curve — the
        # "opposite sign" was misdiagnosed. What's actually happening:
        # the kinematic-bicycle model OVER-PREDICTS yaw rate at FS-DV
        # speeds during transients (slip equilibrium not built up), and
        # the IMU's gyro reads small because gyro-bias drift suppresses
        # the rotation. The slip-gate firing on cornering samples is
        # CORRECT behaviour — the prediction is genuinely untrustworthy
        # in that regime. Negating it just doubled the magnitude error
        # and started corrupting yaw in the wrong direction.
        omega_pred = (v_body_long / wheelbase_m) * np.tan(steering_rad)
        omega_imu  = imu_predicted_dyaw_rad / dt if dt > 1e-6 else 0.0
        if abs(omega_pred - omega_imu) > slip_threshold_rad_s:
            return "slip"

        dyaw_pred = omega_pred * dt
        # BetweenFactor's relative transform is in prev_pose's body
        # frame: forward motion = (v·dt, 0, 0), rotation = R_z(dyaw).
        between = gtsam.Pose3(
            gtsam.Rot3.Rz(dyaw_pred),
            np.array([v_body_long * dt, 0.0, 0.0]),
        )
        sigmas = np.array([
            0.1,           # roll  rad (unused — flat ground)
            0.1,           # pitch rad (unused)
            sigma_yaw_rad, # yaw   rad ← THE constraint we care about
            sigma_xy_m,    # tx m — loose; IMU+RPM handle this
            sigma_xy_m,    # ty m
            0.1,           # tz m (flat ground)
        ])
        noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        self._new_factors.add(gtsam.BetweenFactorPose3(
            X(self._k - 1), X(self._k), between, noise))
        return "staged"

    def stage_new_landmark(
        self,
        landmark_id: int,
        initial_world_xyz: np.ndarray,
    ) -> None:
        """Insert a brand-new landmark variable plus a z-only anchor
        prior.

        Called once per cone, the first time it's observed. Subsequent
        observations of the same cone use stage_cone_observation() with
        its existing id and add another factor between the current pose
        and that landmark.

        Why the z-anchor: stage_cone_observation builds the bearing
        from `Unit3([body_x, body_y, 0.0])` and the range from
        `hypot(body_x, body_y)` — both horizontal-only. With the
        landmark and pose both at z ≈ 0 (anchor + flat-ground
        BetweenFactor) the Jacobian columns of every cone factor
        against landmark.z linearise to zero. That's a rank-1
        deficiency on every landmark. Today the larger graph (V(k),
        B(k), and a long pose chain) supplies enough off-axis
        Hessian structure for iSAM2 to route around it via Bayes-tree
        elimination order, but any change that shrinks the variable
        footprint can expose it as IndeterminantLinearSystem (we hit
        this on every landmark of a position-only graph experiment,
        2026-05-04). Anchoring z near the initial estimate (5 cm 1σ)
        while leaving xy effectively free (10 m 1σ — much wider than
        any cone-factor range residual, so the bearing-range factor
        stays the dominant xy constraint) costs one factor per
        landmark and removes the deficiency.
        """
        initial_point = gtsam.Point3(*initial_world_xyz)
        self._new_values.insert(L(landmark_id), initial_point)
        z_anchor_sigmas = np.array([10.0, 10.0, 0.05])
        z_anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(z_anchor_sigmas)
        self._new_factors.add(gtsam.PriorFactorPoint3(
            L(landmark_id), initial_point, z_anchor_noise))

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
        # Detector's per-cone σ_xy → range-scaled, Huber-robust noise
        # (Hesai 128 ch centroid SE ≈ σ_ray / sqrt(N), so cluster size
        # matters as much as range). Shared with the localization path —
        # see _cone_bearing_range for the full rationale.
        bearing, range_m, robust = _cone_bearing_range(body_x, body_y, sigma_xy)

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

    def localize(
        self,
        predicted_pose: gtsam.Pose3,
        cone_obs: "list[tuple[float, float, float, np.ndarray]]",
        velocity: np.ndarray,
        bias: gtsam.imuBias.ConstantBias,
        prior_sigmas: np.ndarray,
        max_iterations: int = 10,
    ) -> ScanResult:
        """Frozen-map localization: solve for the current pose with a
        small, fixed-size graph that NEVER touches the incremental iSAM2
        instance.

        Once the lap is closed and mapping is frozen (see the node's
        _update_loop_closure), the landmark map is final, so we no longer
        need full SLAM — we need localization against a known map. Growing
        the smoothed graph by X/V/B every scan would keep forcing iSAM2 to
        relinearize an ever-lengthening trajectory; re-observing the
        start/finish cones connects today's pose back to step-0 variables
        and re-eliminates the whole loop (the 150–250 ms loop-closure
        latency burst). This method sidesteps that entirely: a one-shot
        Levenberg–Marquardt solve over a single pose, anchored by a prior
        at the motion-model prediction and corrected by cone BearingRange
        factors to the FROZEN landmark positions. Cost is O(observed
        cones) and constant — no relinearization, no graph growth, ever.

        Args:
            predicted_pose: motion-model prediction (prev pose ∘ odom
                delta) — the prior anchor and linearization point.
            cone_obs: matched observations as
                (body_x, body_y, sigma_xy, landmark_world_xyz) tuples.
                landmark_world_xyz is the frozen iSAM2 estimate.
            velocity, bias: carried through to the ScanResult unchanged
                (the EKF owns velocity in odom mode; bias is inert here).
            prior_sigmas: 6-vector (r, p, yaw, x, y, z) for the pose
                prior. Loose enough that cones correct accumulated drift,
                tight enough to dead-reckon on the prediction when cones
                are sparse or rejected.
            max_iterations: LM iteration cap for bounded latency.

        The landmarks are inserted as variables pinned by a tight 3-axis
        prior (rather than baked into a custom fixed-point factor) so the
        cone factor is byte-identical to the SLAM path's
        BearingRangeFactor3D + Huber model. The tight prior also removes
        the z-rank deficiency that a position-only graph would otherwise
        hit (see stage_new_landmark).
        """
        _t = time.perf_counter()
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        # Local single-pose key, independent of the frozen iSAM2 graph's
        # X(self._k). Landmarks get local L(i) keys in observation order.
        pose_key = X(0)
        values.insert(pose_key, predicted_pose)
        graph.add(gtsam.PriorFactorPose3(
            pose_key, predicted_pose,
            gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)))

        pin_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-4, 1e-4, 1e-4]))
        for i, (body_x, body_y, sigma_xy, lm_xyz) in enumerate(cone_obs):
            lkey = L(i)
            pt = gtsam.Point3(
                float(lm_xyz[0]), float(lm_xyz[1]), float(lm_xyz[2]))
            values.insert(lkey, pt)
            graph.add(gtsam.PriorFactorPoint3(lkey, pt, pin_noise))
            bearing, range_m, robust = _cone_bearing_range(
                body_x, body_y, sigma_xy)
            graph.add(gtsam.BearingRangeFactor3D(
                pose_key, lkey, bearing, range_m, robust))

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iterations)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        solved = optimizer.optimize()
        pose = solved.atPose3(pose_key)

        # Bill the solve to the same accumulator the SLAM path uses for
        # isam.update() so SLAM_PROF's commit/upd column keeps meaning
        # "time spent solving for this scan's pose".
        self.prof_update_ms += (time.perf_counter() - _t) * 1e3
        self.prof_flushes += 1
        return ScanResult(pose=pose, velocity=velocity, bias=bias)

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
        landmark, or None if iSAM2 hasn't merged it yet.

        Reads from the estimate cached by the last _flush_update rather
        than recomputing calculateEstimate() per call. update_from_estimate
        invokes this once per landmark, so a per-call full back-substitution
        meant ~N full solves of the whole graph per scan — the dominant
        cost behind the proc-time blowup. The cache is refreshed on every
        commit and is only ever read in the same scan right after one, so
        it's always fresh.
        """
        if self._latest_estimate is None:
            return None
        try:
            return np.array(self._latest_estimate.atPoint3(L(landmark_id)))
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
        _t = time.perf_counter()
        self._isam.update(self._new_factors, self._new_values)
        # Optional second update for tighter convergence on big residuals.
        # GTSAM convention: 1-2 extra calls per step is normal.
        self._isam.update()
        self.prof_update_ms += (time.perf_counter() - _t) * 1e3

        # Reset accumulators.
        self._new_factors.resize(0)
        self._new_values.clear()

        # Compute the full estimate ONCE per commit and cache it. The tip
        # reads below and every landmark_position() query this scan reuse
        # this single back-substitution instead of each triggering a fresh
        # full solve (was ~N+1 solves/scan of the whole graph). Same
        # numbers as before — only the cost changes.
        _t = time.perf_counter()
        self._latest_estimate = self._isam.calculateEstimate()
        self.prof_calcest_ms += (time.perf_counter() - _t) * 1e3
        self.prof_flushes += 1
        estimate = self._latest_estimate
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

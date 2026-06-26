// Copyright 2026 IFSSIM contributors.
//
// 9-state EKF implementation. See odometry_filter.hpp for the
// algorithm derivation, state vector, and rationale.

#include "odometry_filter/odometry_filter.hpp"

#include <cmath>

namespace odometry_filter {

namespace {
constexpr double kTwoPi = 2.0 * M_PI;

// Wrap an angle to (-π, π].
double wrap_pi(double a) {
  while (a > M_PI) {a -= kTwoPi;}
  while (a < -M_PI) {a += kTwoPi;}
  return a;
}
}  // namespace


OdometryFilter::OdometryFilter(const EkfParams & params) : params_(params) {
  reset();
}

namespace {
// Initial covariance diagonal. All off-diagonals zero; diagonal mixes
// "we know it's zero at activate" (small) with "this state can drift
// quickly" (bigger margin). Pulled out so both reset() and the
// calibration-completion path use the same P0.
void set_initial_covariance(Eigen::Matrix<double, kStateDim, kStateDim> & P) {
  P.setZero();
  P(X, X)         = 0.01 * 0.01;     // (1 cm)²
  P(Y, Y)         = 0.01 * 0.01;
  P(THETA, THETA) = 0.01 * 0.01;     // (0.01 rad)² ≈ (0.6°)²
  P(VX, VX)       = 0.5 * 0.5;       // (0.5 m/s)² — we ARE stationary but allow margin
  P(VY, VY)       = 0.5 * 0.5;
  P(OMEGA, OMEGA) = 0.05 * 0.05;
  P(BA_X, BA_X)   = 0.20 * 0.20;     // BMI088 bias spec (conservative)
  P(BA_Y, BA_Y)   = 0.20 * 0.20;
  P(BG_Z, BG_Z)   = 0.02 * 0.02;
}
}  // namespace


void OdometryFilter::reset() {
  state_ = OdometryState{};
  diag_ = FilterDiagnostics{};
  calib_ = Calibration{};
  x_.setZero();
  set_initial_covariance(P_);

  t_imu_last_.reset();
  latest_steering_rad_ = 0.0;
  have_steering_ = false;
  latest_rpm_ms_ = 0.0;
  have_rpm_ = false;
}

Eigen::Matrix<double, kStateDim, kStateDim> OdometryFilter::covariance() const {
  return P_;
}

Eigen::Matrix<double, kStateDim, 1> OdometryFilter::state_vector() const {
  return x_;
}


void OdometryFilter::push_imu(
  double t,
  const Eigen::Vector3d & accel,
  const Eigen::Vector3d & gyro)
{
  if (!calib_.completed) {
    accumulate_calibration(t, accel, gyro);
    return;
  }

  // First sample after activate: just record the time stamp; we need
  // two samples to form a dt.
  if (!t_imu_last_.has_value()) {
    t_imu_last_ = t;
    return;
  }

  const double dt = t - *t_imu_last_;
  t_imu_last_ = t;
  if (dt < params_.dt_min || dt > params_.dt_max) {
    return;
  }

  predict_step(dt, accel, gyro);

  // Steering kinematic cross-check fires here (with vx and ω fresh
  // from predict). The function gates internally; also sets slip_flag
  // which the NHC step below reads.
  if (have_steering_) {
    correct_steering();
  }

  // Non-holonomic constraint: rolling cars don't slide sideways. This
  // is a pseudo-measurement (z = 0, h = vy) applied every IMU tick.
  // It's the only thing in the filter that observes vy directly.
  //
  // 2026-05-24 (fix/nhc-loose-during-slip): NHC fires every tick now,
  // with the sigma adapted to slip state. Previously NHC was GATED
  // OFF when slip_flag fired — which was "correct" in the sense that
  // real tire slip means vy ≠ 0 — but bag analysis (#447 step C
  // equivalent) on _20260524_123201 showed slip_flag fires for
  // ~half of an autocross lap. During those windows vy ran to
  // ±1 m/s and integrated into 56 m of /odom XY drift mid-lap. SLAM
  // tracks /odom via the #545 BetweenFactor and inherited that
  // drift → cone-DA cascade in the late corner.
  //
  // Slip-aware NHC (sigma_vy_nhc_slip = 0.5 m/s during slip vs the
  // tight 0.10 m/s otherwise) keeps the constraint informative
  // throughout the lap without forcing vy=0 during legitimate
  // sideslip. correct_nhc() reads diag_.slip_flag to pick the sigma.
  correct_nhc();

  publish_state_view();
}


void OdometryFilter::push_rpm(double /*t*/, double rpm) {
  // Track wheel speed even before calibration completes so the bias
  // calibration can gate on a genuine standstill (rpm ≈ 0).
  latest_rpm_ms_ = rpm * params_.rpm_to_ms;
  have_rpm_ = true;
  if (!calib_.completed) {
    return;
  }
  correct_rpm(latest_rpm_ms_);
  publish_state_view();
}


void OdometryFilter::push_steering(double /*t*/, double angle_rad) {
  // Cached for the next push_imu — keeps δ, vx, ω time-aligned at the
  // same tick rather than racing the IMU integration.
  latest_steering_rad_ = angle_rad;
  have_steering_ = true;
}


// ---------------------------------------------------------------------
// Predict step
// ---------------------------------------------------------------------
void OdometryFilter::predict_step(
  double dt,
  const Eigen::Vector3d & accel,
  const Eigen::Vector3d & gyro)
{
  // Bias-correct the raw IMU sample.
  const double ax  = accel(0) - x_(BA_X);
  const double ay  = accel(1) - x_(BA_Y);
  const double wz  = gyro(2)  - x_(BG_Z);

  // Read the previous state into named locals (cheaper than indexing
  // into x_ everywhere; also makes the math line up 1:1 with the
  // header comment block).
  const double theta = x_(THETA);
  const double vx    = x_(VX);
  const double vy    = x_(VY);

  const double c = std::cos(theta);
  const double s = std::sin(theta);

  // Forward Euler. ω is read directly from the gyro each tick (not
  // integrated — F[OMEGA, OMEGA] = 0; see header).
  x_(X)     += (vx * c - vy * s) * dt;
  x_(Y)     += (vx * s + vy * c) * dt;
  x_(THETA)  = wrap_pi(theta + wz * dt);
  x_(VX)    += (ax + wz * vy) * dt;
  x_(VY)    += (ay - wz * vx) * dt;
  x_(OMEGA)  = wz;
  // Biases random-walk via Q; mean dynamics are identity (no update).

  // Build the Jacobian F = ∂f/∂x evaluated at the PRE-update state.
  // Note: ω_{k+1} = wz = (ω̃_z − bg_z) is an *assignment*, not a
  // propagation — it depends on the gyro input and bg_z, but NOT on
  // state.OMEGA. So:
  //   * F[*, OMEGA] = 0 for every row (state.OMEGA does not enter f).
  //   * The Coriolis terms vx ← + wz·vy and vy ← − wz·vx pull their
  //     ω from the input (wz), so their dependence on bg_z is via
  //     ∂wz/∂bg_z = −1, giving F[VX, BG_Z] = −vy·dt and
  //     F[VY, BG_Z] = +vx·dt.
  //   * F[OMEGA, BG_Z] = −1 (the entire ω row is rewritten each tick).
  Eigen::Matrix<double, kStateDim, kStateDim> F =
    Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
  F(X,     THETA) = (-vx * s - vy * c) * dt;
  F(X,     VX)    =  c * dt;
  F(X,     VY)    = -s * dt;
  F(Y,     THETA) = ( vx * c - vy * s) * dt;
  F(Y,     VX)    =  s * dt;
  F(Y,     VY)    =  c * dt;
  F(THETA, BG_Z)  = -dt;
  F(VX,    VY)    =  wz * dt;
  F(VX,    BA_X)  = -dt;
  F(VX,    BG_Z)  = -vy * dt;
  F(VY,    VX)    = -wz * dt;
  F(VY,    BA_Y)  = -dt;
  F(VY,    BG_Z)  =  vx * dt;
  F(OMEGA, OMEGA) = 0.0;       // assignment row — no propagation
  F(OMEGA, BG_Z)  = -1.0;

  // Process noise Q (diagonal, continuous densities × dt).
  Eigen::Matrix<double, kStateDim, kStateDim> Q =
    Eigen::Matrix<double, kStateDim, kStateDim>::Zero();
  // X, Y get noise indirectly through VX/VY.
  Q(THETA, THETA) = (0.005 * 0.005);                            // small
  Q(VX,    VX)    = params_.sigma_ax * params_.sigma_ax;
  Q(VY,    VY)    = params_.sigma_ay * params_.sigma_ay;
  Q(OMEGA, OMEGA) = params_.sigma_gz * params_.sigma_gz;
  Q(BA_X,  BA_X)  = params_.sigma_ba_walk * params_.sigma_ba_walk;
  Q(BA_Y,  BA_Y)  = params_.sigma_ba_walk * params_.sigma_ba_walk;
  Q(BG_Z,  BG_Z)  = params_.sigma_bg_walk * params_.sigma_bg_walk;

  P_ = F * P_ * F.transpose() + Q * dt;
}


// ---------------------------------------------------------------------
// Measurement updates
// ---------------------------------------------------------------------
void OdometryFilter::correct_rpm(double z_vx) {
  // h(x) = vx → H = e_VX.
  Eigen::Matrix<double, 1, kStateDim> H = Eigen::Matrix<double, 1, kStateDim>::Zero();
  H(0, VX) = 1.0;

  const double y       = z_vx - x_(VX);              // innovation
  const double R_rpm   = params_.sigma_rpm * params_.sigma_rpm;
  const double S       = (H * P_ * H.transpose())(0, 0) + R_rpm;  // 1×1
  Eigen::Matrix<double, kStateDim, 1> K = P_ * H.transpose() / S;

  // Schmidt-Kalman partition: this observation reveals VX (and via
  // accel-integration mismatch, BA_X/BA_Y). It reveals NOTHING about
  // pose (X, Y, THETA), yaw rate (OMEGA), or gyro bias (BG_Z).
  // Cross-correlations through P would otherwise pull all of them
  // every tick. On bag _210206 the leakage through P[THETA, *]
  // produced a +59° integrated yaw error mid-lap, even though gyro
  // and yaw-rate matched truth — translated to 25-30 m of /odom
  // position drift via vx·sin(θ_err) integration. Same bug class
  // as the BG_Z leak; same fix recipe (zero the gain, keep Joseph
  // form for symmetry).
  K(X)     = 0.0;
  K(Y)     = 0.0;
  K(THETA) = 0.0;
  K(OMEGA) = 0.0;
  K(BG_Z)  = 0.0;
  // Keep K[BA_X], K[BA_Y], K[VY] — RPM-vs-state-vx residual
  // legitimately informs accel bias (integration mismatch) and
  // through Coriolis P[VY, VX] couplings (rotational vx error
  // manifests in vy too).

  x_ += K * y;
  x_(THETA) = wrap_pi(x_(THETA));

  // Joseph-form covariance update — mandatory whenever K is
  // artificially modified away from the optimal Kalman gain.
  // Standard form P = (I-K·H)·P assumes K = P·Hᵀ/S exactly; with
  // hand-zeroed K components the matrix becomes asymmetric and
  // over many ticks can drift non-PSD, corrupting downstream
  // consumers via the /odom covariance fields. Joseph form
  //   P = (I-K·H)·P·(I-K·H)ᵀ + K·R·Kᵀ
  // is symmetric + PSD for ANY K. Previous attempt (#550, closed)
  // used standard form and broke path_planning via NaN-poisoned
  // TF covariance — see #550 postmortem.
  const Eigen::Matrix<double, kStateDim, kStateDim> IKH =
      Eigen::Matrix<double, kStateDim, kStateDim>::Identity() - K * H;
  P_ = IKH * P_ * IKH.transpose() + R_rpm * (K * K.transpose());
}


void OdometryFilter::correct_nhc() {
  // Non-holonomic constraint as a pseudo-measurement: z = 0, h = vy.
  // Standard practice for wheeled-vehicle odometry filters; bounds the
  // unobservable-vy drift without coupling to any sensor.
  Eigen::Matrix<double, 1, kStateDim> H = Eigen::Matrix<double, 1, kStateDim>::Zero();
  H(0, VY) = 1.0;

  // Slip-aware sigma: tighter when the kinematic-bicycle model is
  // consistent with the gyro (the car is rolling normally), looser
  // when slip_flag is on (real lateral motion is plausible up to
  // ~0.5 m/s 1-σ). See call site in push_imu for the rationale.
  const double sigma_vy = diag_.slip_flag
      ? params_.sigma_vy_nhc_slip
      : params_.sigma_vy_nhc;

  const double y       = 0.0 - x_(VY);                              // innovation
  const double R_nhc   = sigma_vy * sigma_vy;
  const double S       = (H * P_ * H.transpose())(0, 0) + R_nhc;
  Eigen::Matrix<double, kStateDim, 1> K = P_ * H.transpose() / S;

  // Schmidt-Kalman partition: NHC observes VY only (and via accel
  // integration mismatch, the LATERAL bias BA_Y). Pose (X, Y, THETA),
  // yaw rate (OMEGA), forward velocity (VX), longitudinal bias (BA_X),
  // and gyro bias (BG_Z) are unobserved by this constraint. See
  // correct_rpm for the full reasoning.
  //
  // BA_X is zeroed here on purpose, even though P[VY, BA_X] is non-zero:
  // that cross-covariance is a spurious by-product of the Coriolis
  // coupling (BA_X → VX → −ω·vx → VY). Letting the vy pseudo-measurement
  // pull BA_X through it is not a real observation of longitudinal bias.
  // While /odom always has RPM, correct_rpm anchors BA_X every tick and
  // the leak is masked; but in any IMU-only window (RPM dropout) the leak
  // drives BA_X to ~−0.4 m/s², which feeds a phantom +0.4 m/s² into
  // ax = accel_x − BA_X and runs vx away. Same partition reasoning as the
  // BG_Z zeroing. Joseph form below keeps P symmetric/PSD for this
  // hand-modified K.
  K(X)     = 0.0;
  K(Y)     = 0.0;
  K(THETA) = 0.0;
  K(OMEGA) = 0.0;
  K(VX)    = 0.0;
  K(BA_X)  = 0.0;
  K(BG_Z)  = 0.0;
  // K[VY], K[BA_Y] preserved.

  x_ += K * y;
  x_(THETA) = wrap_pi(x_(THETA));

  // Joseph-form covariance update (see correct_rpm).
  const Eigen::Matrix<double, kStateDim, kStateDim> IKH =
      Eigen::Matrix<double, kStateDim, kStateDim>::Identity() - K * H;
  P_ = IKH * P_ * IKH.transpose() + R_nhc * (K * K.transpose());
}


void OdometryFilter::correct_steering() {
  // Kinematic-bicycle prediction of ω from current vx + δ.
  const double vx = x_(VX);
  const double L  = params_.wheelbase_m;
  if (L <= 1.0e-3) {
    return;
  }

  // Low-vx gate: the kinematic-bicycle model assumes steady-state
  // tire slip equilibrium. During launch (vx < ~3 m/s) the front
  // wheels turn but the chassis hasn't built up the lateral force
  // to actually rotate yet, so omega_pred over-predicts. Folding
  // that in pulls the EKF's ω state away from the (correct) gyro
  // reading and bakes spurious yaw into θ for the lifetime of the
  // run. Below the threshold, lean on the gyro alone — bg_z is
  // already calibrated by this point and the BMI088 zero-rate
  // accuracy is well under any drift we care about.
  diag_.low_vx_gate_on = false;
  if (vx < params_.min_vx_for_steering_correct) {
    diag_.low_vx_gate_on = true;
    return;
  }

  const double tan_d = std::tan(latest_steering_rad_);
  const double omega_pred = (vx / L) * tan_d;

  // Diagnostics: residual = pred − measured (consistent with previous
  // /odom_diag/yaw_residual_rad_s contract).
  const double residual = omega_pred - x_(OMEGA);
  diag_.yaw_residual_rad_s = residual;
  diag_.slip_flag = std::abs(residual) > params_.slip_yaw_residual_threshold;

  // Gate the EKF update: above the threshold the kinematic model is
  // wrong (slip), so folding it in would corrupt ω.
  if (diag_.slip_flag) {
    return;
  }

  // h(x) = ω → H picks up ω directly. (vx is in the prediction term
  // h_pred = (vx/L)·tan(δ) only as a way to convert δ into an ω
  // measurement; once we treat this as a measurement OF ω, the
  // measurement equation is z_omega = ω, where z_omega = omega_pred.)
  Eigen::Matrix<double, 1, kStateDim> H = Eigen::Matrix<double, 1, kStateDim>::Zero();
  H(0, OMEGA) = 1.0;

  const double y       = omega_pred - x_(OMEGA);
  const double R_steer = params_.sigma_steer * params_.sigma_steer;
  const double S       = (H * P_ * H.transpose())(0, 0) + R_steer;
  Eigen::Matrix<double, kStateDim, 1> K = P_ * H.transpose() / S;

  // Schmidt-Kalman partition: kinematic-bicycle observation observes
  // OMEGA only. Everything else (pose, velocities, biases) is
  // unobserved by this measurement. The residual reflects model
  // error (transient slip, suspension dynamics) — not state error in
  // X/Y/THETA/VX/VY/biases. See correct_rpm for full reasoning.
  K(X)     = 0.0;
  K(Y)     = 0.0;
  K(THETA) = 0.0;
  K(VX)    = 0.0;
  K(VY)    = 0.0;
  K(BA_X)  = 0.0;
  K(BA_Y)  = 0.0;
  K(BG_Z)  = 0.0;
  // K[OMEGA] preserved — this IS the observed state.

  x_ += K * y;
  x_(THETA) = wrap_pi(x_(THETA));

  // Joseph-form covariance update (see correct_rpm).
  const Eigen::Matrix<double, kStateDim, kStateDim> IKH =
      Eigen::Matrix<double, kStateDim, kStateDim>::Identity() - K * H;
  P_ = IKH * P_ * IKH.transpose() + R_steer * (K * K.transpose());
}


// ---------------------------------------------------------------------
// Stationary calibration
// ---------------------------------------------------------------------
void OdometryFilter::accumulate_calibration(
  double t,
  const Eigen::Vector3d & accel,
  const Eigen::Vector3d & gyro)
{
  if (!calib_.t_first.has_value()) {
    calib_.t_first = t;
  }

  // Only fold samples taken at a genuine standstill (wheel speed ≈ 0). A
  // non-stationary window soaks real motion into the bias: a measured
  // +5.2°/s turn during the 3 s window became a +5.2°/s gyro-bias error
  // that drifted SLAM's heading until it lost lock (the true gyro bias is
  // ~0). Require at least one RPM sample so a not-yet-seen rpm
  // (latest_rpm_ms_ still 0) can't masquerade as a standstill.
  if (have_rpm_ && std::abs(latest_rpm_ms_) <= params_.stationary_speed_ms) {
    calib_.accel_sum += accel;
    calib_.gyro_sum  += gyro;
    calib_.n_samples += 1;
  }

  if ((t - *calib_.t_first) < params_.calibration_seconds) {
    return;
  }

  if (calib_.n_samples > 0) {
    const Eigen::Vector3d accel_mean =
      calib_.accel_sum / static_cast<double>(calib_.n_samples);
    const Eigen::Vector3d gyro_mean =
      calib_.gyro_sum / static_cast<double>(calib_.n_samples);

    // Accel bias: subtract gravity (assumed body-z because the car
    // is stationary on a level surface at spawn). x/y biases are
    // whatever's left.
    calib_.accel_bias = accel_mean - Eigen::Vector3d(0.0, 0.0, kG);
    calib_.gyro_bias  = gyro_mean;
  } else {
    // No standstill captured in the window (recorder opened late / bag
    // starts mid-motion). Don't fabricate a bias from moving data —
    // default to zero bias and level gravity. The gyro is accurate, so 0
    // is far closer to truth than a contaminated mean, and RPM aiding
    // anchors vx regardless of the accel bias.
    calib_.accel_bias = Eigen::Vector3d::Zero();
    calib_.gyro_bias  = Eigen::Vector3d::Zero();
  }
  calib_.completed  = true;

  // Anchor pose at origin once calibrated; biases land in the state.
  // Also re-seed P to P0 — during the calibration accumulation window
  // the post-bias-known predict ticks (if any straggle past the
  // transition threshold but inside the user's outer loop) inflate
  // P off-diagonals between position and velocity. Those off-diagonals
  // wouldn't be physically meaningful: we KNOW the car is stationary
  // at the origin at this instant. Leaving them in causes the very
  // first RPM correction to bump state.X by ~K[X]·z_vx, since
  // K[X] = P[X, VX]/S is large when P[X, VX] has cumulated.
  x_.setZero();
  x_(BA_X) = calib_.accel_bias(0);
  x_(BA_Y) = calib_.accel_bias(1);
  x_(BG_Z) = calib_.gyro_bias(2);
  set_initial_covariance(P_);
  publish_state_view();

  t_imu_last_ = t;
}


void OdometryFilter::seed_forward_velocity(double vx) {
  if (!calib_.completed) {
    return;
  }
  x_(VX) = vx;
  publish_state_view();
}


void OdometryFilter::publish_state_view() {
  state_.x        = x_(X);
  state_.y        = x_(Y);
  state_.yaw      = x_(THETA);
  state_.vx       = x_(VX);
  state_.vy       = x_(VY);
  state_.yaw_rate = x_(OMEGA);
}

}  // namespace odometry_filter

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
  // It's the only thing in the filter that observes vy directly:
  //   * The Coriolis-correct predict cancels centripetal accel during
  //     cornering (v̇y = ay − ω·vx ≈ 0 in steady-state turns), but it
  //     does not OBSERVE vy. Any small residual ay from bias error or
  //     sensor noise integrates into vy with nothing pulling it back.
  //     Live-sim regression: a stationary car with calibrated biases
  //     drifted to vy ≈ +0.96 m/s within ~10 s of activation, which
  //     fed PurePursuit a false "the car is sliding" signal and sent
  //     it off-track before the first turn.
  //   * Gated on slip_flag (raised by correct_steering when the
  //     kinematic-bicycle prediction disagrees with the gyro). Real
  //     lateral motion (tire sideslip during hard cornering) violates
  //     vy = 0; disabling NHC there lets the EKF carry the true
  //     non-zero vy.
  //   * sigma_vy_nhc is loose enough (0.10 m/s) that the EKF doesn't
  //     fight legitimate body-y dynamics within rolling tolerance, but
  //     tight enough to pull bias-noise integration back to zero on
  //     ~1 s timescales.
  // Prior memory note ("NHC failed 3 times — fix Coriolis instead")
  // applied to NHC layered on top of the broken complementary predict,
  // where NHC was masking the Coriolis-missing symptom and fighting
  // real centripetal accel. With the predict now Coriolis-correct, NHC
  // and predict don't conflict — NHC just bounds the unobservable-vy
  // drift.
  if (!diag_.slip_flag) {
    correct_nhc();
  }

  publish_state_view();
}


void OdometryFilter::push_rpm(double /*t*/, double rpm) {
  if (!calib_.completed) {
    return;
  }
  const double z_vx = rpm * params_.rpm_to_ms;
  correct_rpm(z_vx);
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
  const Eigen::Matrix<double, kStateDim, 1> K = P_ * H.transpose() / S;

  x_ += K * y;
  x_(THETA) = wrap_pi(x_(THETA));

  P_ = (Eigen::Matrix<double, kStateDim, kStateDim>::Identity() - K * H) * P_;
}


void OdometryFilter::correct_nhc() {
  // Non-holonomic constraint as a pseudo-measurement: z = 0, h = vy.
  // Standard practice for wheeled-vehicle odometry filters; bounds the
  // unobservable-vy drift without coupling to any sensor.
  Eigen::Matrix<double, 1, kStateDim> H = Eigen::Matrix<double, 1, kStateDim>::Zero();
  H(0, VY) = 1.0;

  const double y       = 0.0 - x_(VY);                              // innovation
  const double R_nhc   = params_.sigma_vy_nhc * params_.sigma_vy_nhc;
  const double S       = (H * P_ * H.transpose())(0, 0) + R_nhc;
  const Eigen::Matrix<double, kStateDim, 1> K = P_ * H.transpose() / S;

  x_ += K * y;
  x_(THETA) = wrap_pi(x_(THETA));

  P_ = (Eigen::Matrix<double, kStateDim, kStateDim>::Identity() - K * H) * P_;
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
  const Eigen::Matrix<double, kStateDim, 1> K = P_ * H.transpose() / S;

  x_ += K * y;
  x_(THETA) = wrap_pi(x_(THETA));

  P_ = (Eigen::Matrix<double, kStateDim, kStateDim>::Identity() - K * H) * P_;
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

  calib_.accel_sum += accel;
  calib_.gyro_sum  += gyro;
  calib_.n_samples += 1;

  if ((t - *calib_.t_first) < params_.calibration_seconds) {
    return;
  }
  if (calib_.n_samples == 0) {
    return;
  }

  const Eigen::Vector3d accel_mean =
    calib_.accel_sum / static_cast<double>(calib_.n_samples);
  const Eigen::Vector3d gyro_mean =
    calib_.gyro_sum / static_cast<double>(calib_.n_samples);

  // Accel bias: subtract gravity (assumed body-z because the car
  // is stationary on a level surface at spawn). x/y biases are
  // whatever's left.
  calib_.accel_bias = accel_mean - Eigen::Vector3d(0.0, 0.0, kG);
  calib_.gyro_bias  = gyro_mean;
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


void OdometryFilter::publish_state_view() {
  state_.x        = x_(X);
  state_.y        = x_(Y);
  state_.yaw      = x_(THETA);
  state_.vx       = x_(VX);
  state_.vy       = x_(VY);
  state_.yaw_rate = x_(OMEGA);
}

}  // namespace odometry_filter

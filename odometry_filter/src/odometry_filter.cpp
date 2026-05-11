// Copyright 2026 IFSSIM contributors.
//
// Implementation of OdometryFilter. The math here is intentionally
// a line-for-line port of the Python reference in
// pipeline/sim_supervisor/sim_supervisor/odometry.py — same constants,
// same operator order, same wrapping. Numeric equivalence is verified
// by gtest fixtures in test/test_odometry_filter.cpp that mirror the
// 21 pytest cases in pipeline/sim_supervisor/test/test_odometry.py.

#include "odometry_filter/odometry_filter.hpp"

#include <algorithm>
#include <cmath>

namespace odometry_filter {

namespace {
constexpr double kTwoPi = 2.0 * M_PI;
}  // namespace

OdometryFilter::OdometryFilter(const Params & params) : params_(params) {
  diag_.effective_alpha_vx = params_.alpha_vx;
}

void OdometryFilter::reset() {
  state_ = OdometryState{};
  calib_ = Calibration{};
  diag_ = FilterDiagnostics{};
  diag_.effective_alpha_vx = params_.alpha_vx;
  t_imu_last_.reset();
  t_rpm_last_.reset();
  latest_rpm_vx_.reset();
  latest_steering_rad_ = 0.0;
  latest_brake_ = 0.0;
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

  // Integration dt — clamp to a sane bound so a clock glitch
  // (e.g. sim time discontinuity at refresh-bridge) doesn't
  // produce a one-tick velocity spike.
  if (!t_imu_last_.has_value()) {
    t_imu_last_ = t;
    return;
  }
  const double dt = t - *t_imu_last_;
  t_imu_last_ = t;
  if (dt <= 0.0 || dt > 0.1) {
    return;
  }

  // Bias-corrected readings
  const Eigen::Vector3d a_body = accel - calib_.accel_bias;
  const Eigen::Vector3d w_body = gyro - calib_.gyro_bias;

  // Yaw rate is gyro-z directly (no integration step needed —
  // we read instantaneous angular velocity from the gyro).
  state_.yaw_rate = w_body(2);

  // Integrate yaw, wrap to [-pi, pi].
  state_.yaw += state_.yaw_rate * dt;
  if (state_.yaw > M_PI) {
    state_.yaw -= kTwoPi;
  } else if (state_.yaw < -M_PI) {
    state_.yaw += kTwoPi;
  }

  // Predict vx, vy from accel. Body-frame: ax pushes vx, ay
  // pushes vy. We do NOT include accel-z (gravity removal with
  // an unrolled chassis is a kinematic-bicycle assumption that
  // holds for FSD courses).
  const double ax_body = a_body(0);
  const double ay_body = a_body(1);

  // vx prediction: pure integration. RPM correction lands in
  // push_rpm() asynchronously.
  state_.vx += ax_body * dt;

  // vy prediction with kinematic-bicycle decay toward zero.
  // Real lateral accel during a yaw maneuver still shows up;
  // the leak just keeps integration noise from accumulating.
  state_.vy =
    (1.0 - params_.beta_vy_leak) * state_.vy + ay_body * dt;

  // Integrate position (rotate body velocity into world frame).
  const double c = std::cos(state_.yaw);
  const double s = std::sin(state_.yaw);
  state_.x += (c * state_.vx - s * state_.vy) * dt;
  state_.y += (s * state_.vx + c * state_.vy) * dt;

  // Phase 3 (#383): kinematic-bicycle yaw-rate cross-check.
  // Predicted yaw rate from current vx + steering angle:
  //     ω_pred = (v_x / wheelbase) · tan(δ)
  // Residual = ω_pred - ω_measured (IMU gyro_z). Persistent
  // non-zero residual = the car is sliding (real ω diverges
  // from kinematic prediction).
  if (params_.wheelbase_m > 1e-3) {
    const double yaw_rate_pred =
      state_.vx / params_.wheelbase_m * std::tan(latest_steering_rad_);
    diag_.yaw_residual_rad_s = yaw_rate_pred - state_.yaw_rate;
    diag_.slip_flag =
      std::abs(diag_.yaw_residual_rad_s) > params_.slip_yaw_residual_threshold;
  }
}

void OdometryFilter::push_rpm(double t, double rpm) {
  t_rpm_last_ = t;
  latest_rpm_vx_ = rpm * params_.rpm_to_ms;

  if (!calib_.completed) {
    return;
  }

  // Phase 3 (#383): scale α_vx down when brake authority is high
  // (drive wheels potentially locked → RPM unreliable). The
  // filter falls back to IMU integration during the slip window;
  // vx re-converges to RPM after brake releases.
  const double alpha = (latest_brake_ > params_.brake_lockup_threshold)
                         ? params_.alpha_vx_brake
                         : params_.alpha_vx;
  diag_.effective_alpha_vx = alpha;

  // Apply the complementary-filter correction immediately on
  // arrival rather than waiting for the next IMU step. RPM is
  // the slower input; deferring would add latency.
  const double residual = *latest_rpm_vx_ - state_.vx;
  state_.vx += alpha * residual;
}

void OdometryFilter::push_steering(double /*t*/, double angle_rad) {
  // No state mutation here — the prediction lives in push_imu so it
  // stays time-aligned with the IMU gyro_z it's compared against.
  latest_steering_rad_ = angle_rad;
}

void OdometryFilter::push_brake(double /*t*/, double brake) {
  // Defensive clamp to [0, 1] even though the bridge enforces the
  // normalisation.
  latest_brake_ = std::clamp(brake, 0.0, 1.0);
}

void OdometryFilter::accumulate_calibration(
  double t,
  const Eigen::Vector3d & accel,
  const Eigen::Vector3d & gyro)
{
  if (!calib_.t_first.has_value()) {
    calib_.t_first = t;
  }

  calib_.accel_sum += accel;
  calib_.gyro_sum += gyro;
  calib_.n_samples += 1;

  if ((t - *calib_.t_first) < params_.calibration_seconds) {
    return;
  }
  if (calib_.n_samples == 0) {
    return;
  }

  // Estimate biases as the mean of the stationary readings.
  // Accel bias: subtract gravity (assumed body-z, since the car
  // is stationary on a level surface). The simulator places the
  // car upright at spawn so this assumption holds at t=0.
  const Eigen::Vector3d accel_mean =
    calib_.accel_sum / static_cast<double>(calib_.n_samples);
  const Eigen::Vector3d gyro_mean =
    calib_.gyro_sum / static_cast<double>(calib_.n_samples);

  calib_.accel_bias = accel_mean - Eigen::Vector3d(0.0, 0.0, kG);
  calib_.gyro_bias = gyro_mean;
  calib_.completed = true;

  // Anchor pose at origin once calibration is done. Velocity is
  // zero (we just verified stationary).
  state_ = OdometryState{};
  t_imu_last_ = t;
}

}  // namespace odometry_filter

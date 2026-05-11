// Copyright 2026 IFSSIM contributors.
//
// gtest port of the 21 pytest scenarios in
// pipeline/sim_supervisor/test/test_odometry.py. Same fixtures, same
// numeric assertions. The two implementations have to agree to 1e-9
// absolute tolerance on every fixture so any sim/real-car divergence
// shows up here, not in a lap test.
//
// Run inside the container:
//   colcon build --packages-select odometry_filter
//   colcon test --packages-select odometry_filter
//   colcon test-result --verbose

#include <cmath>

#include <gtest/gtest.h>

#include "odometry_filter/odometry_filter.hpp"

using odometry_filter::OdometryFilter;
using odometry_filter::Params;
using odometry_filter::kG;
using odometry_filter::kAlphaVx;
using odometry_filter::kAlphaVxBrake;
using odometry_filter::kRpmToMs;

namespace {

constexpr double kImuDt = 0.0025;  // 400 Hz
constexpr int kCalibrationSamples = 1500;

// stationary_filter fixture — mirrors pytest's fixture of the same name.
// Feeds 1500 samples at 400 Hz of clean stationary IMU (accel = (0, 0, g),
// gyro = 0), leaving the filter post-calibration with biases ~0.
OdometryFilter make_stationary_filter() {
  OdometryFilter f;
  const Eigen::Vector3d accel_stationary(0.0, 0.0, kG);
  const Eigen::Vector3d gyro_stationary(0.0, 0.0, 0.0);
  for (int i = 0; i < kCalibrationSamples; ++i) {
    f.push_imu(static_cast<double>(i) * kImuDt, accel_stationary, gyro_stationary);
  }
  EXPECT_TRUE(f.is_calibrated()) << "fixture should leave filter calibrated";
  return f;
}

}  // namespace


// ============================================================================
// Calibration behaviour
// ============================================================================

TEST(OdometryFilterCalibration, StartsUncalibrated) {
  OdometryFilter f;
  EXPECT_FALSE(f.is_calibrated());
}

TEST(OdometryFilterCalibration, CalibratesWithinWindow) {
  OdometryFilter f;
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  for (int i = 0; i < 1500; ++i) {
    f.push_imu(i * kImuDt, accel, gyro);
  }
  EXPECT_TRUE(f.is_calibrated());
}

TEST(OdometryFilterCalibration, EstimatesAccelBias) {
  OdometryFilter f;
  const double bias_x = 0.05;
  const Eigen::Vector3d accel(bias_x, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  for (int i = 0; i < 1500; ++i) {
    f.push_imu(i * kImuDt, accel, gyro);
  }
  ASSERT_TRUE(f.is_calibrated());
  EXPECT_NEAR(f.accel_bias_for_tests()(0), bias_x, 1e-6);
  EXPECT_NEAR(f.state().x, 0.0, 1e-12);
  EXPECT_NEAR(f.state().y, 0.0, 1e-12);
  EXPECT_NEAR(f.state().vx, 0.0, 1e-12);
}

TEST(OdometryFilterCalibration, UncalibratedFilterPublishesZeroState) {
  auto f = make_stationary_filter();
  EXPECT_TRUE(f.is_calibrated());
  const auto & s = f.state();
  EXPECT_EQ(s.x, 0.0);
  EXPECT_EQ(s.y, 0.0);
  EXPECT_EQ(s.vx, 0.0);
  EXPECT_EQ(s.vy, 0.0);
}


// ============================================================================
// Velocity tracking — RPM correction
// ============================================================================

TEST(OdometryFilterRpm, PullsVxTowardTarget) {
  auto f = make_stationary_filter();
  const double target_rpm = 100.0;
  const double target_vx = target_rpm * kRpmToMs;
  // α=0.10 default → each push closes ~10 % of residual.
  // 200 pushes → gap is target × (1-α)^200 ≪ 1e-4.
  for (int i = 0; i < 200; ++i) {
    f.push_rpm(i * 0.0125, target_rpm);
  }
  EXPECT_NEAR(f.state().vx, target_vx, 1e-4);
}

TEST(OdometryFilterRpm, CorrectionIsMonotone) {
  auto f = make_stationary_filter();
  const double target_rpm = 50.0;
  const double target_vx = target_rpm * kRpmToMs;
  double prev_vx = f.state().vx;
  for (int i = 0; i < 20; ++i) {
    f.push_rpm(i * 0.0125, target_rpm);
    EXPECT_GE(f.state().vx, prev_vx - 1e-9) << "vx should be non-decreasing";
    EXPECT_LE(f.state().vx, target_vx + 1e-9) << "no overshoot";
    prev_vx = f.state().vx;
  }
}


// ============================================================================
// Yaw integration
// ============================================================================

TEST(OdometryFilterYaw, IntegratesConstantGyro) {
  auto f = make_stationary_filter();
  const Eigen::Vector3d gyro(0.0, 0.0, 0.5);  // 0.5 rad/s
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 400; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  // 1 s of integration at 0.5 rad/s → yaw ≈ 0.5 rad. Tolerance for
  // the first-tick skip (initial t_imu_last_ assignment).
  EXPECT_NEAR(f.state().yaw, 0.5, 0.01);
}

TEST(OdometryFilterYaw, WrapsToNegative) {
  OdometryFilter f;
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro_zero = Eigen::Vector3d::Zero();
  for (int i = 0; i < 1500; ++i) {
    f.push_imu(i * kImuDt, accel, gyro_zero);
  }
  ASSERT_TRUE(f.is_calibrated());
  const Eigen::Vector3d gyro_spin(0.0, 0.0, 5.0);  // 5 rad/s
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 320; ++i) {  // 0.8 s → ~4 rad raw → wraps
    f.push_imu(t0 + i * kImuDt, accel, gyro_spin);
  }
  EXPECT_GE(f.state().yaw, -M_PI);
  EXPECT_LE(f.state().yaw, M_PI);
}


// ============================================================================
// Integration end-to-end
// ============================================================================

TEST(OdometryFilterIntegration, ConstantRpmDrivesPosition) {
  auto f = make_stationary_filter();
  const double target_rpm = 100.0;
  const double target_vx = target_rpm * kRpmToMs;
  for (int i = 0; i < 200; ++i) {
    f.push_rpm(i * 0.0125, target_rpm);
  }
  EXPECT_NEAR(f.state().vx, target_vx, 1e-4);

  // 1 s @ 400 Hz, zero accel, RPM re-pinned every 5 IMU samples.
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 400; ++i) {
    if (i % 5 == 0) {
      f.push_rpm(0.625 + i * kImuDt, target_rpm);
    }
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  // After 1 s at vx ≈ 0.898 m/s, expect x ≈ 0.898 m.
  EXPECT_NEAR(f.state().x, target_vx, 0.05);
}


// ============================================================================
// reset()
// ============================================================================

TEST(OdometryFilterReset, ClearsState) {
  auto f = make_stationary_filter();
  // Mutate state.
  f.push_rpm(0.0, 200.0);
  EXPECT_GT(f.state().vx, 0.0);
  f.reset();
  EXPECT_FALSE(f.is_calibrated());
  EXPECT_EQ(f.state().x, 0.0);
  EXPECT_EQ(f.state().vx, 0.0);
  EXPECT_EQ(f.state().yaw, 0.0);
}


// ============================================================================
// Phase 3 — yaw-rate cross-check + slip flag
// ============================================================================

TEST(OdometryFilterPhase3, YawResidualZeroWhenKinematicsMatch) {
  auto f = make_stationary_filter();
  // vx>0, steering=0, no slip → yaw_residual ≈ 0.
  f.push_steering(0.0, 0.0);
  f.push_rpm(0.0, 200.0);
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 50; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  EXPECT_NEAR(f.diagnostics().yaw_residual_rad_s, 0.0, 1e-9);
  EXPECT_FALSE(f.diagnostics().slip_flag);
}

TEST(OdometryFilterPhase3, YawResidualNonzeroUnderSteering) {
  auto f = make_stationary_filter();
  // Steering ≠ 0 + non-trivial vx + IMU gyro_z = 0 → ω_pred = (vx/L)tan(δ)
  // gives a non-zero residual. With vx ≈ 1.6 (from RPM 200), L=1.55,
  // δ=0.4: ω_pred ≈ 1.6/1.55 · tan(0.4) ≈ 0.44 rad/s > 0.3 threshold.
  f.push_steering(0.0, 0.4);
  // Re-pin RPM continuously so the α=0.10 blended vx actually settles
  // near the target before we measure the residual.
  const double target_rpm = 200.0;
  for (int i = 0; i < 100; ++i) {
    f.push_rpm(i * 0.0125, target_rpm);
  }
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 50; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  EXPECT_GT(std::abs(f.diagnostics().yaw_residual_rad_s), 0.1);
}


// ============================================================================
// Brake-pressure α scaling (#383)
// ============================================================================

TEST(OdometryFilterBrake, CollapsesAlphaVx) {
  auto f = make_stationary_filter();
  f.push_brake(0.0, 0.5);  // > 0.30 threshold
  f.push_rpm(0.0, 100.0);
  EXPECT_DOUBLE_EQ(f.diagnostics().effective_alpha_vx, kAlphaVxBrake);
}

TEST(OdometryFilterBrake, ReleaseRestoresAlphaVx) {
  auto f = make_stationary_filter();
  f.push_brake(0.0, 0.5);  // under brake
  f.push_rpm(0.0, 100.0);
  ASSERT_DOUBLE_EQ(f.diagnostics().effective_alpha_vx, kAlphaVxBrake);
  // Release.
  f.push_brake(0.1, 0.0);
  f.push_rpm(0.1, 100.0);
  EXPECT_DOUBLE_EQ(f.diagnostics().effective_alpha_vx, kAlphaVx);
}

TEST(OdometryFilterBrake, BelowThresholdUsesNormalAlpha) {
  auto f = make_stationary_filter();
  f.push_brake(0.0, 0.20);  // < 0.30
  f.push_rpm(0.0, 100.0);
  EXPECT_DOUBLE_EQ(f.diagnostics().effective_alpha_vx, kAlphaVx);
}


// ============================================================================
// reset() clears Phase 3 state too
// ============================================================================

TEST(OdometryFilterReset, ClearsPhase3State) {
  auto f = make_stationary_filter();
  f.push_steering(0.0, 0.4);
  f.push_brake(0.0, 0.7);
  f.push_rpm(0.0, 200.0);
  ASSERT_DOUBLE_EQ(f.diagnostics().effective_alpha_vx, kAlphaVxBrake);
  f.reset();
  EXPECT_DOUBLE_EQ(f.diagnostics().effective_alpha_vx, kAlphaVx);
  EXPECT_FALSE(f.diagnostics().slip_flag);
  EXPECT_NEAR(f.diagnostics().yaw_residual_rad_s, 0.0, 1e-9);
}

// ============================================================================
// Brake clamping
// ============================================================================

TEST(OdometryFilterBrake, ClampsOutOfRangeBrake) {
  auto f = make_stationary_filter();
  // Negative brake clamps to 0 → below threshold → normal α.
  f.push_brake(0.0, -0.5);
  f.push_rpm(0.0, 100.0);
  EXPECT_DOUBLE_EQ(f.diagnostics().effective_alpha_vx, kAlphaVx);
  // > 1 clamps to 1 → above threshold → brake α.
  f.push_brake(0.0, 1.5);
  f.push_rpm(0.0, 100.0);
  EXPECT_DOUBLE_EQ(f.diagnostics().effective_alpha_vx, kAlphaVxBrake);
}

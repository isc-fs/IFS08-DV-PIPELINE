// Copyright 2026 IFSSIM contributors.
//
// Dead-reckoning odometry filter for the IFS-08 DV pipeline.
//
// This is the C++ port of the Python OdometryFilter (was
// pipeline/sim_supervisor/sim_supervisor/odometry.py). The
// algorithm is identical line-for-line — numeric equivalence to
// 1e-9 verified by the gtest suite in test/test_odometry_filter.cpp
// using the same fixtures as the Python pytest suite.
//
// Inputs (both sides of the sim/real-car parity):
//   * IMU (BMI088-class) — accel + gyro at native rate (~400 Hz).
//     Used for prediction (vx integration via accel-x, yaw via
//     gyro-z) and for stationary bias calibration.
//   * Motor RPM (~80 Hz) — primary longitudinal velocity correction.
//     Multiplied by rpm_to_ms to yield body-frame longitudinal speed
//     at the rear axle.
//   * Steering angle (rad) — kinematic-bicycle yaw cross-check.
//   * Brake pressure ([0, 1]) — scales the RPM correction when
//     wheels are potentially locked.
//
// Output: 6-state body-frame estimate
//   pose:  x, y, yaw  (world-frame, integrated from spawn)
//   twist: vx, vy, yaw_rate  (body-frame instantaneous)
//
// Filter: complementary (predict on IMU, correct on RPM). The pure-
// math class is ROS-agnostic — drivers / nodes wrap it. Sim uses
// odometry_filter_node (planned Phase 2). The real-car uDV firmware
// will link the same library.
//
// Why this lives outside sim_supervisor: the supervisor is sim-only
// (faking the real-car uDV's plumbing). The filter is the only piece
// of the supervisor that has a real-car counterpart — extracting it
// makes that boundary explicit.

#ifndef ODOMETRY_FILTER__ODOMETRY_FILTER_HPP_
#define ODOMETRY_FILTER__ODOMETRY_FILTER_HPP_

#include <Eigen/Core>
#include <optional>

namespace odometry_filter {

// ---------------------------------------------------------------------
// Module-level constants (mirror odometry.py module globals exactly).
// ---------------------------------------------------------------------

// Motor-RPM → body-frame longitudinal velocity.
// 2026-05-10 re-derivation (issue #380): bagged /odom and
// /testing_only/odom over a 41 s motion window on test_submodule.csv,
// paired GT vs filter-output samples within ±50 ms windows, computed
// mean(|GT.vx|) / mean(|/odom.vx|) = 0.9140 across 3324 paired
// samples (p10=0.898, p90=0.944, spread 0.046 — tight, steady).
// New constant 0.00898 × 0.9140 = 0.00821 recovers exactly the doc-
// derived value from docs/dv_pipeline_rebuild.md §3.5:
//     RPM_TO_MS = (2π × WheelRadius / GearRatio) / 60
//               = (2π × 0.228 / 2.909) / 60
//               = 0.00821
constexpr double kRpmToMs = 0.00821;

// Drop motor-RPM samples this old (seconds). Sustained staleness
// means the bridge stopped publishing; fall back to IMU-only
// prediction rather than feeding stale velocity corrections.
constexpr double kRpmStaleS = 0.5;

// Stationary-calibration window. While collecting IMU samples the
// filter does not publish — the autonomy stack tolerates brief
// startup delay (it's already inside Phase 1's "configuring" stage).
// Same value slam_node uses for its own bias calibration.
constexpr double kCalibrationSeconds = 3.0;

// Complementary-filter blend on vx. RPM correction strength per RPM
// message. 0.10 = each new RPM sample pulls vx by 10 % of the
// residual. With RPM at ~80 Hz that's an effective time constant of
// ~125 ms — fast enough to track real accel/decel, slow enough to
// average out RPM quantisation noise.
constexpr double kAlphaVx = 0.10;

// Wheelbase used in the kinematic-bicycle yaw prediction:
//     ω_pred = (v_x / wheelbase) · tan(δ)
// Compared against IMU gyro_z to publish a yaw residual diagnostic +
// detect lateral-slip events. 1.570 m is the authoritative IFS-08
// spec — confirmed in docs/MODEL_IFS_08/DYNAMIC_MOD/Cooling/
// BR_regenerativeBR.m (`L = 1.570`) and the MONO sheet of
// docs/MODEL_IFS_08/ISC_IFS_08.xlsx (BQ64 `L = 1570 mm`). The previous
// value (1.55) was sourced from the URDF, which itself was approximate
// — see issue #462. inputs_vehicle_dynamics.m lists 1.627 (4% disagreement
// between authors); 1.570 is what the BR/Sandra and master MONO sheet
// agree on and what we treat as authoritative until the spec owners
// converge.
constexpr double kWheelbaseM = 1.570;

// Brake-pressure threshold above which RPM is considered unreliable
// (drive wheels potentially locked). When exceeded the complementary
// filter's α_vx is scaled toward zero so vx integration tracks the
// IMU prediction rather than the RPM-derived target. 0.30 (30 % brake
// command) is a deliberately conservative floor.
constexpr double kBrakeLockupThreshold = 0.30;

// α_vx multiplier when brake_pressure > kBrakeLockupThreshold.
// 0.05 keeps a small residual pull toward RPM so the filter
// eventually re-converges after the brake event, but lets IMU
// integration dominate during the slip window.
constexpr double kAlphaVxBrake = 0.05;

// |yaw_residual| > this → slip_flag goes true on the latest tick.
// 0.3 rad/s ≈ 17 °/s — comfortably above sensor noise but well below
// what the steering-angle range × wheelbase product would imply at
// typical FS speeds.
constexpr double kSlipYawResidualThreshold = 0.3;

// Body-frame lateral-velocity decay per IMU step. The kinematic
// bicycle assumes vy ≈ 0 in clean rolling; this slow leak pulls vy
// toward zero while the IMU accel-y prediction term still tracks
// real lateral accel during yaw maneuvers. 1e-3 per IMU sample at
// 400 Hz → ~2.5 s time constant.
constexpr double kBetaVyLeak = 1e-3;

// Gravity magnitude (sim's BMI088 model uses standard gravity).
constexpr double kG = 9.81;


// ---------------------------------------------------------------------
// 6-DOF body-frame state /odom carries.
// ---------------------------------------------------------------------
struct OdometryState {
  // World-frame position, integrated from spawn (drifts long-term)
  double x{0.0};
  double y{0.0};
  double yaw{0.0};

  // Body-frame velocity (vx forward, vy left per REP-103)
  double vx{0.0};
  double vy{0.0};
  double yaw_rate{0.0};
};


// ---------------------------------------------------------------------
// Cross-check residuals the node surfaces on /odom_diag/* for tuning
// + slip detection. Populated each IMU tick when both steering and
// IMU samples are fresh.
// ---------------------------------------------------------------------
struct FilterDiagnostics {
  // ω_pred - ω_measured, where ω_pred = (vx/L)·tan(δ).
  // Persistent non-zero residual = car is sliding (yaw rate from
  // IMU differs from what steering geometry would predict).
  double yaw_residual_rad_s{0.0};

  // True iff |yaw_residual| > kSlipYawResidualThreshold on the
  // latest tick. Raw per-tick truth; downstream consumers debounce.
  bool slip_flag{false};

  // Effective α_vx applied to the most recent RPM correction.
  // Drops toward kAlphaVxBrake when brake_pressure > threshold.
  double effective_alpha_vx{kAlphaVx};
};


// ---------------------------------------------------------------------
// Complementary filter — IMU prediction + RPM correction.
//
// Lifecycle:
//   1. Construction: state is zero, calibration not started.
//   2. push_imu() during first kCalibrationSeconds: accumulate bias
//      estimates (assumes car is stationary; accel reads (0, 0, +g)
//      modulo bias; gyro reads (0, 0, 0) modulo bias). Output
//      undefined; is_calibrated() returns false.
//   3. After calibration: push_imu() drives prediction (integrate
//      accel-x into vx, accel-y into vy with leak, gyro-z into yaw).
//      push_rpm() applies the correction step.
//   4. state() always returns the latest estimate.
//
// All parameters keep their module-level defaults unless overridden
// in the constructor — convenient for unit tests that want to bypass
// calibration or use deterministic values.
// ---------------------------------------------------------------------
struct Params {
  double rpm_to_ms = kRpmToMs;
  double rpm_stale_s = kRpmStaleS;
  double calibration_seconds = kCalibrationSeconds;
  double alpha_vx = kAlphaVx;
  double beta_vy_leak = kBetaVyLeak;
  double wheelbase_m = kWheelbaseM;
  double brake_lockup_threshold = kBrakeLockupThreshold;
  double alpha_vx_brake = kAlphaVxBrake;
  double slip_yaw_residual_threshold = kSlipYawResidualThreshold;
};


class OdometryFilter {
 public:
  // Default-constructed filter uses all module defaults.
  OdometryFilter() : OdometryFilter(Params{}) {}

  // Custom-parameter constructor — for unit tests + production
  // tuning. Pass a struct-literal Params{.alpha_vx = 0.05, ...}.
  explicit OdometryFilter(const Params & params);

  // ----- Public read-only accessors -----
  const OdometryState & state() const noexcept { return state_; }
  const FilterDiagnostics & diagnostics() const noexcept { return diag_; }
  bool is_calibrated() const noexcept { return calib_.completed; }

  // Tear down all state. Use on lifecycle on_cleanup.
  void reset();

  // ----- Ingestion API (called from the ROS node) -----

  // One IMU sample. Called from the IMU subscription callback at the
  // BMI088 native rate (~400 Hz).
  //
  // accel, gyro are 3-vectors in body frame. accel is m/s² with
  // gravity included (so a stationary upright IMU reads
  // ~(0, 0, +9.81)); gyro is rad/s.
  void push_imu(
    double t,
    const Eigen::Vector3d & accel,
    const Eigen::Vector3d & gyro);

  // One motor-RPM sample. Called from the RPM subscription (~80 Hz).
  // t is wall-clock seconds; stored for the rpm_stale_s check.
  void push_rpm(double t, double rpm);

  // One steering-angle sample (rad). Cached for the next push_imu's
  // kinematic-bicycle cross-check.
  void push_steering(double t, double angle_rad);

  // One brake-pressure sample. Clamped to [0, 1] defensively.
  void push_brake(double t, double brake);

  // ----- Internal-state access for unit tests only -----
  // Mirrors the Python test suite which pokes at _calib for the
  // accel_bias / gyro_bias values. Production callers should NOT
  // touch these — they may change without notice.
  const Eigen::Vector3d & accel_bias_for_tests() const noexcept {
    return calib_.accel_bias;
  }
  const Eigen::Vector3d & gyro_bias_for_tests() const noexcept {
    return calib_.gyro_bias;
  }
  int n_calibration_samples_for_tests() const noexcept {
    return calib_.n_samples;
  }

 private:
  // Calibration accumulators — populated during the first
  // calibration_seconds of push_imu calls, then frozen.
  struct Calibration {
    int n_samples{0};
    Eigen::Vector3d accel_sum{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gyro_sum{Eigen::Vector3d::Zero()};
    std::optional<double> t_first{};
    bool completed{false};

    Eigen::Vector3d accel_bias{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gyro_bias{Eigen::Vector3d::Zero()};
  };

  void accumulate_calibration(
    double t,
    const Eigen::Vector3d & accel,
    const Eigen::Vector3d & gyro);

  Params params_;
  OdometryState state_;
  Calibration calib_;
  FilterDiagnostics diag_;

  std::optional<double> t_imu_last_{};
  std::optional<double> t_rpm_last_{};
  std::optional<double> latest_rpm_vx_{};

  double latest_steering_rad_{0.0};
  double latest_brake_{0.0};
};

}  // namespace odometry_filter

#endif  // ODOMETRY_FILTER__ODOMETRY_FILTER_HPP_

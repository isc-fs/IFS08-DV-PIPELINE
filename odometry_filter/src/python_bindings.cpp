// Copyright 2026 IFSSIM contributors.
//
// pybind11 bindings for the production 9-state odometry EKF.
//
// These exist so the offline SLAM benchmark (tools/sim_benchmark) drives
// the EXACT C++ filter the pipeline runs, instead of a hand-maintained
// Python re-implementation that can silently drift out of sync (a stale
// port once made a filter change look like a regression in the
// benchmark — the whole reason this module exists). The benchmark imports
// `odometry_filter_py` and wraps OdometryFilter directly; only the
// deliberately-degraded diagnostic variants (open-loop wheel DR) stay in
// Python, since they are NOT a port of any production code.
//
// The library itself stays ROS-free; this is an optional build artifact
// gated on pybind11 being available (see CMakeLists).

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "odometry_filter/odometry_filter.hpp"

namespace py = pybind11;
using odometry_filter::EkfParams;
using odometry_filter::FilterDiagnostics;
using odometry_filter::OdometryFilter;
using odometry_filter::OdometryState;

PYBIND11_MODULE(odometry_filter_py, m) {
  m.doc() =
      "Python bindings for the production C++ 9-state odometry EKF "
      "(odometry_filter::OdometryFilter). Used by the SLAM benchmark so "
      "offline replay runs the real filter, not a re-implementation.";

  py::class_<EkfParams>(m, "EkfParams")
      .def(py::init<>())
      .def_readwrite("wheelbase_m", &EkfParams::wheelbase_m)
      .def_readwrite("rpm_to_ms", &EkfParams::rpm_to_ms)
      .def_readwrite("calibration_seconds", &EkfParams::calibration_seconds)
      .def_readwrite("stationary_speed_ms", &EkfParams::stationary_speed_ms)
      .def_readwrite("sigma_ax", &EkfParams::sigma_ax)
      .def_readwrite("sigma_ay", &EkfParams::sigma_ay)
      .def_readwrite("sigma_gz", &EkfParams::sigma_gz)
      .def_readwrite("sigma_ba_walk", &EkfParams::sigma_ba_walk)
      .def_readwrite("sigma_bg_walk", &EkfParams::sigma_bg_walk)
      .def_readwrite("sigma_rpm", &EkfParams::sigma_rpm)
      .def_readwrite("sigma_steer", &EkfParams::sigma_steer)
      .def_readwrite("sigma_vy_nhc", &EkfParams::sigma_vy_nhc)
      .def_readwrite("sigma_vy_nhc_slip", &EkfParams::sigma_vy_nhc_slip)
      .def_readwrite(
          "slip_yaw_residual_threshold",
          &EkfParams::slip_yaw_residual_threshold)
      .def_readwrite(
          "min_vx_for_steering_correct",
          &EkfParams::min_vx_for_steering_correct)
      .def_readwrite("dt_min", &EkfParams::dt_min)
      .def_readwrite("dt_max", &EkfParams::dt_max);

  py::class_<OdometryState>(m, "OdometryState")
      .def_readonly("x", &OdometryState::x)
      .def_readonly("y", &OdometryState::y)
      .def_readonly("yaw", &OdometryState::yaw)
      .def_readonly("vx", &OdometryState::vx)
      .def_readonly("vy", &OdometryState::vy)
      .def_readonly("yaw_rate", &OdometryState::yaw_rate);

  py::class_<FilterDiagnostics>(m, "FilterDiagnostics")
      .def_readonly("yaw_residual_rad_s", &FilterDiagnostics::yaw_residual_rad_s)
      .def_readonly("slip_flag", &FilterDiagnostics::slip_flag)
      .def_readonly("low_vx_gate_on", &FilterDiagnostics::low_vx_gate_on);

  py::class_<OdometryFilter>(m, "OdometryFilter")
      .def(py::init<>())
      .def(py::init<const EkfParams &>(), py::arg("params"))
      .def("push_imu", &OdometryFilter::push_imu,
           py::arg("t"), py::arg("accel"), py::arg("gyro"))
      .def("push_rpm", &OdometryFilter::push_rpm,
           py::arg("t"), py::arg("rpm"))
      .def("push_steering", &OdometryFilter::push_steering,
           py::arg("t"), py::arg("angle_rad"))
      .def("seed_forward_velocity", &OdometryFilter::seed_forward_velocity,
           py::arg("vx"))
      .def("reset", &OdometryFilter::reset)
      .def("is_calibrated", &OdometryFilter::is_calibrated)
      .def("covariance", &OdometryFilter::covariance)
      .def("state_vector", &OdometryFilter::state_vector)
      .def_property_readonly(
          "state", &OdometryFilter::state,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "diagnostics", &OdometryFilter::diagnostics,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "params", &OdometryFilter::params,
          py::return_value_policy::reference_internal);
}

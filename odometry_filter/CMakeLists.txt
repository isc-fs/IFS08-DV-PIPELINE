cmake_minimum_required(VERSION 3.16)
project(odometry_filter CXX)

# -----------------------------------------------------------------------------
# C++ port of the Python OdometryFilter that lived in
# sim_supervisor.odometry. This package ships only the pure-math
# library + gtests; the ROS node that drives it (subscriptions,
# publishes, TF) is a separate package (odometry_filter_node, Phase 2).
#
# Compiled with -O2 -ffast-math: the filter is a hot loop at 400 Hz
# IMU rate and the math is plain FP arithmetic with no NaN/Inf
# expected. If those ever creep in (e.g. from a malformed IMU sample
# in replay) downstream consumers see them via /odom; we don't trap
# them here.
# -----------------------------------------------------------------------------

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2 -ffast-math)
endif()

find_package(ament_cmake REQUIRED)
# eigen3_cmake_module is the ROS-friendly way to find Eigen3 — it
# brings in the system Eigen3Config.cmake without depending on the
# host's pkg-config setup. Provided by ros-humble-eigen3-cmake-module.
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# -----------------------------------------------------------------------------
# Library — pure C++, no ROS deps.
# -----------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED
  src/odometry_filter.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

# -----------------------------------------------------------------------------
# Optional Python bindings (pybind11) — lets the offline SLAM benchmark
# drive the REAL filter instead of a hand-maintained re-implementation
# that can drift out of sync. Gated on pybind11 being available so the
# core library / firmware build never depends on it.
# -----------------------------------------------------------------------------
# pybind11's newer tooling (pybind11_add_module) calls python3_add_library
# from CMake's FindPython3, so the Development component must be found
# BEFORE pybind11 is loaded.
find_package(Python3 QUIET COMPONENTS Interpreter Development)
find_package(pybind11_vendor QUIET)
find_package(pybind11 QUIET)
if(pybind11_FOUND AND Python3_Development_FOUND)
  # Compile the filter source straight into the module (rather than
  # linking libodometry_filter.so) so the resulting .so is SELF-CONTAINED.
  # The benchmark stages just this one file into a container whose image
  # may carry a different libodometry_filter.so; a runtime dependency on
  # that lib would then fail to resolve the bindings' symbols.
  pybind11_add_module(odometry_filter_py
    src/python_bindings.cpp
    src/odometry_filter.cpp)
  target_include_directories(odometry_filter_py
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(odometry_filter_py PRIVATE Eigen3::Eigen)

  # Install the compiled module where a sourced workspace puts it on
  # PYTHONPATH (the distro's site-packages, relative to the install root).
  if(NOT DEFINED PYTHON_INSTALL_DIR)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import sysconfig, os; print(os.path.relpath(sysconfig.get_path('platlib'), sysconfig.get_path('data')))"
      OUTPUT_VARIABLE PYTHON_INSTALL_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  install(TARGETS odometry_filter_py DESTINATION "${PYTHON_INSTALL_DIR}")
else()
  message(STATUS
    "pybind11 not found — skipping odometry_filter_py bindings "
    "(SLAM benchmark falls back to its Python reference port).")
endif()

# -----------------------------------------------------------------------------
# Tests — gtest fixtures cover the EKF math (predict + correct +
# bias estimation + Coriolis cornering regression).
# -----------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_odometry_filter
    test/test_odometry_filter.cpp
  )
  target_link_libraries(test_odometry_filter ${PROJECT_NAME})
  # -ffast-math is fine for the filter hot loop but reorders FP enough
  # that EXPECT_NEAR with tight tolerances misbehaves. Disable it on
  # the test target only.
  target_compile_options(test_odometry_filter PRIVATE -fno-fast-math)
endif()

ament_package()

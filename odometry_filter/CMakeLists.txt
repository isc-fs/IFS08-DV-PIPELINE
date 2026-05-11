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
# Tests — gtest fixtures port the 21 Python scenarios in
# pipeline/sim_supervisor/test/test_odometry.py. Numeric equivalence is
# the contract: same input sequence in → same state.{x,y,yaw,vx,vy} out
# to 1e-9 absolute tolerance.
# -----------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_odometry_filter
    test/test_odometry_filter.cpp
  )
  target_link_libraries(test_odometry_filter ${PROJECT_NAME})

  # Numeric-equivalence harness — see test/validate_vs_python.py
  # for the full story. Built and installed only under BUILD_TESTING
  # so the production binary set stays minimal.
  add_executable(validate_vs_python_harness
    test/validate_vs_python_harness.cpp
  )
  target_link_libraries(validate_vs_python_harness ${PROJECT_NAME})
  install(
    TARGETS validate_vs_python_harness
    DESTINATION lib/${PROJECT_NAME}
  )
  install(
    PROGRAMS test/validate_vs_python.py
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()

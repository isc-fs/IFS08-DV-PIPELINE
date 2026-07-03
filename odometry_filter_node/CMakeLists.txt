cmake_minimum_required(VERSION 3.16)
project(odometry_filter_node CXX)

# rclcpp wrapper for the OdometryFilter library shipped in the
# `odometry_filter` package. See package.xml for the multi-phase
# rationale (Phase 1 = pure-math library, this = Phase 2).

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(odometry_filter REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(node_base_cpp REQUIRED)
find_package(dv_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(odometry_filter_node
  src/odometry_filter_node.cpp
)
target_link_libraries(odometry_filter_node
  Eigen3::Eigen
)
target_link_libraries(odometry_filter_node
  node_base_cpp::node_base_cpp
)
ament_target_dependencies(odometry_filter_node
  odometry_filter
  rclcpp
  rclcpp_lifecycle
  dv_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  geometry_msgs
  tf2_ros
)

install(
  TARGETS odometry_filter_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()

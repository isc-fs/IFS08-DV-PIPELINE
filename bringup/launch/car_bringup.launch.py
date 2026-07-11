"""
Car bringup — the dv-pipeline.service entry point: sensors + autonomy.

car_pipeline.launch.py is deliberately autonomy-only (mode_manager +
mission_control + lifecycle nodes on the real-vehicle topic surface).
This wrapper adds the on-vehicle sensor layer around it, replacing the
old isc_ws/isc_startup stub stack as the DVPC race-mode entry point:

  * Hesai ATX LiDAR driver → /lidar_points. The hesai_ros_driver package
    lives in the DVPC's ~/ros2_ws (not this repo); it is resolved lazily
    at launch time via FindPackageShare, so this file imports fine on
    machines without it as long as with_lidar:=false.
  * Static base_link → hesai_lidar / imu_link TFs. REP-103 (x forward,
    y left, z up), base_link z=0 at the CAR floor, origin at the rear-axle
    midpoint. Geometry measured 2026-06-29 (heights) + 2026-07-11 (x/y and
    orientation): LiDAR x=0.667 y=0.027 z=0.9127, pitch 3.323deg nose-down
    (0.0580 rad); IMU remounted aligned 2026-07-11 x=0.425 y=0.040, z=0.2962
    inherited (height unchanged), roll/pitch/yaw=0.
    NOTE: these TFs are consumed by tf2 (viz + any tf2-based transform) but
    NOT by the estimation data path — cone_detection emits cones already
    labelled base_link (lidar_xy hardcoded (0,0)) and the EKF/SLAM read /imu
    raw. So the LiDAR x/pitch is not yet applied to cone positions (bounded
    bias); the aligned IMU needs no correction beyond a minor translation
    lever-arm. See dvpc-pipeline-integration-tasks item 10.
  * Optional foxglove_bridge for umbilical/bench monitoring
    (foxglove:=true; default off — racing needs no visualisation).

Usage (dv-pipeline.service / bench):
  ros2 launch bringup car_bringup.launch.py
  ros2 launch bringup car_bringup.launch.py foxglove:=true      # bench
  ros2 launch bringup car_bringup.launch.py with_lidar:=false   # no ATX
"""
from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    with_lidar = LaunchConfiguration("with_lidar")
    foxglove = LaunchConfiguration("foxglove")

    return LaunchDescription([
        DeclareLaunchArgument(
            "with_lidar", default_value="true",
            description="Start the Hesai ATX driver (needs hesai_ros_driver "
                        "from the DVPC's ros2_ws sourced)."),
        DeclareLaunchArgument(
            "foxglove", default_value="false",
            description="Start foxglove_bridge (bench/umbilical monitoring)."),

        # --- Sensors ------------------------------------------------------
        # Hesai ATX driver → /lidar_points (topic + hesai_lidar frame are
        # configured in the driver's own config.yaml on the DVPC).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("hesai_ros_driver"), "launch", "start.py",
                ])
            ),
            condition=IfCondition(with_lidar),
        ),

        # --- Static TFs (car geometry) -------------------------------------
        # base_link → hesai_lidar (frame_id of the driver's PointCloud2).
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="tf_base_to_lidar",
            arguments=[
                "--x", "0.667", "--y", "0.027", "--z", "0.9127",
                # pitch 3.323deg nose-down (0.0580 rad); roll/yaw level.
                "--yaw", "0.0", "--pitch", "0.0580", "--roll", "0.0",
                "--frame-id", "base_link", "--child-frame-id", "hesai_lidar",
            ],
        ),
        # base_link → imu_link (must match the uDV firmware's
        # imu_msg.header.frame_id — see ros_task.c: "imu_link").
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="tf_base_to_imu",
            arguments=[
                # IMU remounted aligned 2026-07-11 (roll/pitch/yaw=0); z
                # inherited from the 2026-06-29 measurement (height unchanged).
                "--x", "0.425", "--y", "0.040", "--z", "0.2962",
                "--yaw", "0.0", "--pitch", "0.0", "--roll", "0.0",
                "--frame-id", "base_link", "--child-frame-id", "imu_link",
            ],
        ),

        # --- Bench monitoring (opt-in) --------------------------------------
        Node(
            package="foxglove_bridge", executable="foxglove_bridge",
            name="foxglove_bridge",
            condition=IfCondition(foxglove),
        ),

        # --- Autonomy (management + lifecycle nodes, car profile) ----------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("bringup"), "launch",
                    "car_pipeline.launch.py",
                ])
            ),
        ),
    ])

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
    y left, z up), base_link z=0 at the CAR floor. Vertical distances
    measured 2026-06-29: LiDAR base 912.7 mm above the car floor; IMU
    616.5 mm below the LiDAR (≈ LattePanda location) → z = 0.2962 m.
    ⚠️ x/y offsets of both sensors and the ATX mounting orientation
    (yaw/pitch/roll) are still UNMEASURED — 0.0 placeholders below.
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
            # prerun/ BRANCH DIVERGENCE — DO NOT MERGE TO dev (see PRERUN.md).
            # Default flipped true->false so the service boots WITHOUT the
            # Hesai driver: a replayed rosbag feeds /lidar_points (re-stamped
            # to now via deploy/prerun_restamp_relay.py) as the only source,
            # else the live driver and the bag collide. Pass with_lidar:=true
            # to run the real driver. This is the sole functional change here.
            "with_lidar", default_value="false",
            description="Start the Hesai ATX driver (needs hesai_ros_driver "
                        "from the DVPC's ros2_ws sourced). prerun/: default "
                        "false — the bag provides /lidar_points."),
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
                "--x", "0.0", "--y", "0.0", "--z", "0.9127",  # x,y TODO: measure
                "--yaw", "0.0", "--pitch", "0.0", "--roll", "0.0",  # TODO: ATX level?
                "--frame-id", "base_link", "--child-frame-id", "hesai_lidar",
            ],
        ),
        # base_link → imu_link (must match the uDV firmware's
        # imu_msg.header.frame_id — see ros_task.c: "imu_link").
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="tf_base_to_imu",
            arguments=[
                "--x", "0.0", "--y", "0.0", "--z", "0.2962",  # x,y TODO: measure
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

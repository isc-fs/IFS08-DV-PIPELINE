"""Car sensor layer — the on-vehicle sensor sources, WITHOUT any autonomy.

Split out of car_bringup.launch.py (#manual-record) so it can run on its own:

  * car_bringup.launch.py = this + car_pipeline.launch.py (sensors + autonomy),
    which is what dv-pipeline.service launches for a race / DV run.
  * dv-manual.service launches THIS ALONE, so manual driving can be recorded
    (full sensor graph incl. /lidar_points) without bringing up mission_control
    or the lifecycle nodes. See deploy/dv_manual.sh + docs/RECORDING.md.

What it starts:
  * Hesai ATX LiDAR driver → /lidar_points (hesai_ros_driver, from the DVPC's
    ~/ros2_ws, resolved lazily so this imports on machines without it when
    with_lidar:=false).
  * Static base_link → hesai_lidar / imu_link TFs (REP-103, base_link z=0 at
    the car floor).
  * Optional foxglove_bridge (foxglove:=true) for bench/umbilical monitoring.

It does NOT start the uDV micro-ROS agent — the uDV publishes /imu, /motor_rpm
and /steering_angle over microros-agent.service, which is independent of both
this and the pipeline. So in manual mode those topics are present iff that
service is up (it normally is).
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
    ])

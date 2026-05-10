"""
robot_state_publisher launch — coche_urdf.

Loads the IFS-08 placeholder URDF, runs robot_state_publisher at the
default 200 Hz TF rate, and runs joint_state_publisher with hard-coded
zero positions for the four wheel rotation joints + two steering
joints. The result is a static TF tree under base_link visible in
Foxglove / RViz.

When real joint actuation eventually lands (steer angle from the
control stack, wheel rotation from motor RPM), joint_state_publisher
gets replaced by a node that publishes /joint_states from the
autonomy-side state. For now zero-state is fine — the chassis is
still visible and the canonical TF root is in place.

Usage:
  ros2 launch coche_urdf robot_state_publisher.launch.py

Or from a parent launch file as IncludeLaunchDescription.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("coche_urdf")
    urdf_path = os.path.join(pkg_share, "urdf", "ifs_08.urdf")

    with open(urdf_path, "r") as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                # 200 Hz TF — diagram contract. RSP will publish at the
                # rate /joint_states arrives, so this only matters for
                # the static identity transforms; joint_state_publisher
                # below is what drives the dynamic ones.
                "publish_frequency": 200.0,
            }],
        ),
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            output="screen",
            parameters=[{
                # Default-state publisher — zero rad on all joints.
                # rate sets /joint_states publish frequency.
                "rate": 200,
                "use_gui": False,
            }],
        ),
    ])

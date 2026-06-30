"""Topic remap contract — pure data, no launch / ROS imports.

This module is the single source of truth for the topic-name surface
each autonomy node is wired onto, for both the simulator (`/fsds/*`
bridge) and the real car (uDV micro-ROS node + Hesai driver).

It is deliberately free of any `launch` / `launch_ros` / `rclpy`
import so the contract can be unit-tested in a plain pytest
environment with no ROS installed (see bringup/test/test_topic_contract.py).
`launch_common` imports these constants and `autonomy_remaps()` to build
the actual LifecycleNode actions.

Remap tuple convention (ROS 2 `from:=to`): the FIRST element is the
topic name the node uses *in its own source code*; the SECOND is the
real topic it is rewritten onto. The "from" therefore differs per node
depending on what that node hardcodes — e.g. cone_detection_node
subscribes to "/fsds/lidar/Lidar1" in code, so its car remap "from" is
"/fsds/lidar/Lidar1", whereas odometry_filter_node subscribes to "/imu"
in code, so its "from" is "/imu". Getting this direction wrong silently
breaks the car (the node subscribes to a topic nobody publishes), which
is exactly why this table is pinned by tests.
"""
from __future__ import annotations


# ---------------------------------------------------------------------
# Simulator surface (IFSSIM /fsds/* bridge → pipeline-side names).
# Mirrors the pre-refactor table from
# docker/dv_pipeline_stack/pipeline.launch.py.
# ---------------------------------------------------------------------
REMAP_LIDAR    = ("/fsds/lidar/Lidar1", "/lidar/Lidar1")
REMAP_GSS      = ("/fsds/gss",          "/gss")
REMAP_IMU      = ("/fsds/imu",          "/imu")
REMAP_GT       = ("/fsds/testing_only/odom", "/testing_only/odom")
# GT track layout — only consumed by slam's debug_gt_cones diagnostic.
REMAP_TRACK    = ("/fsds/testing_only/track", "/testing_only/track")
REMAP_RPM      = ("/fsds/motor_rpm",    "/motor_rpm")
REMAP_CMD      = ("/fsds/control_command", "/control_command")
REMAP_STEERING = ("/fsds/steering_angle", "/steering_angle")
REMAP_BRAKE    = ("/fsds/brake_pressure", "/brake_pressure")


# ---------------------------------------------------------------------
# Car surface (real-vehicle sources → pipeline-side names).
#
# Verified against the live firmware (IFS08-DV-uDV @ feat/14, node
# `cubemx_node`, empty namespace) and the Hesai driver:
#
#   uDV   /imu/data_raw   sensor_msgs/Imu          400 Hz  → pure remap
#   Hesai /lidar_points   sensor_msgs/PointCloud2  ~10 Hz  → pure remap
#
# Only IMU and LiDAR are pure remaps (type + units already match). The
# other two EKF inputs are published by the uDV directly on their
# canonical names (the unit conversions moved into firmware), so they are
# NOT in this table:
#
#   /steering_angle  uDV converts its steering sensor DEG→RAD on-board
#                    and publishes /steering_angle (RADIANS) directly.
#   /motor_rpm       uDV reads the inverter (CAN) and publishes
#                    /motor_rpm (motor-shaft RPM) directly.
#
# See docs/CAR_ADAPTATION.md for the full contract and the flagged gaps.
# ---------------------------------------------------------------------
REMAP_IMU_CAR   = ("/imu",               "/imu/data_raw")
REMAP_LIDAR_CAR = ("/fsds/lidar/Lidar1", "/lidar_points")

# NOTE: the *runtime* stock-typed uDV ↔ mission_control interface (the
# /assi/state, /ami/mission, /dv/status, /ctrl/cmd, /force_ebs byte
# contract + AMI→mission_id map) lives in
# `mission_control.interface_contract`, NOT here. It is consumed by
# mission_control (the reconciler) and sim_supervisor (the sim uDV
# emulator) at runtime; this launch-only module must NOT import it
# (bringup already depends on mission_control — importing back would be a
# circular package dependency). topic_contract stays purely the launch
# remap table.


# Node executables in mode_manager.AUTONOMY_NODE_ORDER order. Kept as a
# bare tuple here (rather than importing mode_registry) so this module
# has zero dependencies.
AUTONOMY_EXECUTABLES = (
    "odometry_filter_node",
    "cone_detection_node",
    "slam_node",
    "path_planning_node",
    "control_node",
)


def autonomy_remaps(profile: str = "sim") -> dict[str, list[tuple[str, str]]]:
    """Return ``{executable: [(from, to), ...]}`` for the given profile.

    Pure function (no launch import) so launch wiring and tests share
    exactly one definition of which node is remapped onto what.

    profile="sim": the historical IFSSIM bridge wiring.
    profile="car": the real-vehicle wiring — IMU+LiDAR pure remaps onto
        the uDV/Hesai topics; steering_angle + motor_rpm are published by
        the uDV on canonical names (so no remap entry); the sim-only
        ground-truth taps are dropped.
    """
    if profile not in ("sim", "car"):
        raise ValueError(
            f"autonomy_remaps: unknown profile {profile!r} "
            f"(expected 'sim' or 'car')")

    if profile == "car":
        return {
            "odometry_filter_node": [REMAP_IMU_CAR],
            "cone_detection_node": [REMAP_LIDAR_CAR],
            "slam_node": [REMAP_IMU_CAR],
            "path_planning_node": [],
            "control_node": [],
        }

    return {
        "odometry_filter_node": [REMAP_IMU, REMAP_RPM, REMAP_STEERING,
                                 REMAP_BRAKE],
        "cone_detection_node": [REMAP_LIDAR],
        "slam_node": [REMAP_IMU, REMAP_RPM, REMAP_GT, REMAP_TRACK],
        "path_planning_node": [],
        "control_node": [],
    }

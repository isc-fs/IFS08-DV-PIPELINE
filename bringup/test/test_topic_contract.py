"""Unit tests for the topic remap contract (sim + car profiles).

`bringup.topic_contract` is the single source of truth for which
autonomy node is wired onto which topic, for both the simulator and the
real car. The remap *direction* is subtle and silently breaks the car
when wrong (a node subscribes to a topic nobody publishes), so this test
pins the whole table.

Pure module — no launch / rclpy / DDS — runs in plain pytest with no ROS
install. Mirrors the import shim used by mission_control's tests.
"""
from __future__ import annotations

import os
import sys

import pytest

# Put the OUTER bringup/ dir (the one holding the `bringup` package) on
# the path so `import bringup.topic_contract` resolves without ROS.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from bringup.topic_contract import (  # noqa: E402
    AUTONOMY_EXECUTABLES,
    REMAP_IMU_CAR,
    REMAP_LIDAR_CAR,
    autonomy_remaps,
)


# --------------------------------------------------------------------
# Car remap direction — the bit that silently bricks the car if wrong.
# --------------------------------------------------------------------

def test_car_imu_remap_direction():
    # odometry_filter_node + slam_node subscribe to "/imu" IN CODE; on
    # the car that must be rewritten onto the uDV's /imu/data_raw.
    assert REMAP_IMU_CAR == ("/imu", "/imu/data_raw")


def test_car_lidar_remap_direction():
    # cone_detection_node subscribes to "/fsds/lidar/Lidar1" IN CODE;
    # on the car that must be rewritten onto the Hesai /lidar_points.
    assert REMAP_LIDAR_CAR == ("/fsds/lidar/Lidar1", "/lidar_points")


def test_car_remaps_never_target_isc_namespace():
    # The adaptation doc assumed an /isc/* surface that the firmware does
    # NOT publish. Guard against anyone re-introducing it.
    for remaps in autonomy_remaps("car").values():
        for _from, _to in remaps:
            assert not _to.startswith("/isc/"), (
                f"car remap targets non-existent /isc/* topic: {_to}")


# --------------------------------------------------------------------
# Car table shape — IMU + LiDAR pure remaps; steering/rpm come from the
# bridge on canonical names (so they must NOT appear as remaps).
# --------------------------------------------------------------------

def test_car_only_imu_and_lidar_are_remapped():
    car = autonomy_remaps("car")
    assert car["odometry_filter_node"] == [REMAP_IMU_CAR]
    assert car["cone_detection_node"] == [REMAP_LIDAR_CAR]
    assert car["slam_node"] == [REMAP_IMU_CAR]
    assert car["path_planning_node"] == []
    assert car["control_node"] == []


def test_car_does_not_remap_steering_or_rpm():
    # The uDV publishes /steering_angle (rad, converted on-board) and
    # /motor_rpm (from the inverter) directly, so neither may be remapped
    # — a remap here would mean the node subscribes to a name nobody
    # publishes.
    flat = [t for remaps in autonomy_remaps("car").values() for t in remaps]
    targets = {to for _from, to in flat}
    froms = {frm for frm, _to in flat}
    assert "/motor_rpm" not in targets and "/motor_rpm" not in froms
    assert "/steering_angle" not in targets and "/steering_angle" not in froms


def test_car_drops_sim_only_ground_truth_taps():
    # /fsds/testing_only/* has no publisher on the car.
    flat = [t for remaps in autonomy_remaps("car").values() for t in remaps]
    for frm, to in flat:
        assert "testing_only" not in frm and "testing_only" not in to


# --------------------------------------------------------------------
# Sim profile is unchanged (regression guard for the refactor).
# --------------------------------------------------------------------

def test_sim_profile_preserves_fsds_surface():
    sim = autonomy_remaps("sim")
    assert ("/fsds/imu", "/imu") in sim["odometry_filter_node"]
    assert ("/fsds/motor_rpm", "/motor_rpm") in sim["odometry_filter_node"]
    assert ("/fsds/steering_angle", "/steering_angle") in \
        sim["odometry_filter_node"]
    assert ("/fsds/lidar/Lidar1", "/lidar/Lidar1") in sim["cone_detection_node"]


# --------------------------------------------------------------------
# Both profiles cover exactly the five autonomy executables.
# --------------------------------------------------------------------

@pytest.mark.parametrize("profile", ["sim", "car"])
def test_profile_covers_every_autonomy_node(profile):
    remaps = autonomy_remaps(profile)
    assert set(remaps.keys()) == set(AUTONOMY_EXECUTABLES)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        autonomy_remaps("bench")

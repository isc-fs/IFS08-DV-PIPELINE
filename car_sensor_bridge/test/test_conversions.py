"""Unit tests for car_sensor_bridge.conversions.

Pure numeric transforms — no rclpy / DDS. Pins the steering deg→rad and
inverter→motor-RPM contracts that the on-car EKF depends on.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from car_sensor_bridge.conversions import (  # noqa: E402
    erpm_to_motor_rpm,
    inverter_value_to_motor_rpm,
    steering_sensor_deg_to_rad,
)


# ----------------------- steering deg → rad ---------------------------

def test_steering_basic_degrees_to_radians():
    assert steering_sensor_deg_to_rad(90.0) == pytest.approx(math.pi / 2)
    assert steering_sensor_deg_to_rad(0.0) == 0.0
    assert steering_sensor_deg_to_rad(-30.0) == pytest.approx(math.radians(-30))


def test_steering_matches_bench_anchor():
    # P1 bench: /steering/cmd 20.0 drives the wheel to +20°, sensor reads
    # ~+20°. With ratio 1.0 that is the road-wheel δ in radians.
    assert steering_sensor_deg_to_rad(20.0) == pytest.approx(0.349065, abs=1e-5)


def test_steering_ratio_scales_to_road_wheel():
    # A 10:1 steering ratio: 90° at the wheel ⇒ 9° at the road wheel.
    out = steering_sensor_deg_to_rad(90.0, steering_ratio=10.0)
    assert out == pytest.approx(math.radians(9.0))


def test_steering_sign_flips_direction():
    assert steering_sensor_deg_to_rad(15.0, sign=-1.0) == \
        pytest.approx(math.radians(-15.0))


def test_steering_offset_subtracted_before_scaling():
    # offset 5° zero-point: a raw 25° reads as 20° of true angle.
    out = steering_sensor_deg_to_rad(25.0, offset_deg=5.0)
    assert out == pytest.approx(math.radians(20.0))


def test_steering_offset_and_ratio_compose():
    out = steering_sensor_deg_to_rad(
        100.0, steering_ratio=10.0, offset_deg=10.0, sign=-1.0)
    # (-1) * (100 - 10) / 10 = -9° → radians
    assert out == pytest.approx(math.radians(-9.0))


def test_steering_zero_ratio_raises():
    with pytest.raises(ValueError):
        steering_sensor_deg_to_rad(10.0, steering_ratio=0.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_steering_non_finite_raises(bad):
    with pytest.raises(ValueError):
        steering_sensor_deg_to_rad(bad)
    with pytest.raises(ValueError):
        steering_sensor_deg_to_rad(10.0, steering_ratio=bad)


# ----------------------- inverter → motor RPM -------------------------

def test_erpm_divides_by_pole_pairs():
    assert erpm_to_motor_rpm(10000.0, 10) == pytest.approx(1000.0)
    assert erpm_to_motor_rpm(0.0, 5) == 0.0


def test_erpm_negative_pole_pairs_raises():
    with pytest.raises(ValueError):
        erpm_to_motor_rpm(1000.0, 0)
    with pytest.raises(ValueError):
        erpm_to_motor_rpm(1000.0, -3)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_erpm_non_finite_raises(bad):
    with pytest.raises(ValueError):
        erpm_to_motor_rpm(bad, 10)


def test_inverter_value_erpm_path():
    out = inverter_value_to_motor_rpm(
        12000.0, is_erpm=True, pole_pairs=12, scale=1.0)
    assert out == pytest.approx(1000.0)


def test_inverter_value_already_mechanical():
    # is_erpm False ⇒ pole_pairs ignored, value passes through (× scale).
    out = inverter_value_to_motor_rpm(
        1500.0, is_erpm=False, pole_pairs=10, scale=1.0)
    assert out == pytest.approx(1500.0)


def test_inverter_scale_applied_first():
    # scale folds in an LSB; here raw counts × 0.5 = eRPM, /pole_pairs.
    out = inverter_value_to_motor_rpm(
        20000.0, is_erpm=True, pole_pairs=10, scale=0.5)
    assert out == pytest.approx(1000.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_inverter_value_non_finite_raises(bad):
    with pytest.raises(ValueError):
        inverter_value_to_motor_rpm(bad, is_erpm=True, pole_pairs=10)
    with pytest.raises(ValueError):
        inverter_value_to_motor_rpm(
            100.0, is_erpm=False, pole_pairs=10, scale=bad)

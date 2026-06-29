"""Unit tests for car_supervisor.actuation — command scaling.

Pure numeric. Pins the steering [-1,1]→degrees scaling (incl. the
defence-in-depth safety clamp) and throttle clamping the relay depends
on.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from car_supervisor.actuation import (  # noqa: E402
    safe_stop_steering_deg,
    steering_norm_to_deg,
    throttle_norm_clamp,
)


# ----------------------- steering [-1,1] → deg ------------------------

def test_steering_full_lock_maps_to_max():
    assert steering_norm_to_deg(
        1.0, max_steering_deg=20.0, safety_limit_deg=25.0) == 20.0
    assert steering_norm_to_deg(
        -1.0, max_steering_deg=20.0, safety_limit_deg=25.0) == -20.0


def test_steering_centre_is_zero():
    assert steering_norm_to_deg(
        0.0, max_steering_deg=20.0, safety_limit_deg=25.0) == 0.0


def test_steering_half_input_is_half_max():
    assert steering_norm_to_deg(
        0.5, max_steering_deg=20.0, safety_limit_deg=25.0) == 10.0


def test_steering_input_overshoot_is_clamped_to_unit_first():
    # |input|>1 is clamped to 1 before scaling.
    assert steering_norm_to_deg(
        2.0, max_steering_deg=20.0, safety_limit_deg=25.0) == 20.0


def test_steering_safety_limit_caps_a_bad_scale():
    # A mis-set max (50°) must still be capped at the safety limit (25°).
    assert steering_norm_to_deg(
        1.0, max_steering_deg=50.0, safety_limit_deg=25.0) == 25.0
    assert steering_norm_to_deg(
        -1.0, max_steering_deg=50.0, safety_limit_deg=25.0) == -25.0


def test_steering_rejects_non_positive_limits():
    with pytest.raises(ValueError):
        steering_norm_to_deg(0.5, max_steering_deg=0.0, safety_limit_deg=25.0)
    with pytest.raises(ValueError):
        steering_norm_to_deg(0.5, max_steering_deg=20.0, safety_limit_deg=0.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_steering_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        steering_norm_to_deg(
            bad, max_steering_deg=20.0, safety_limit_deg=25.0)


# ----------------------- throttle clamp -------------------------------

@pytest.mark.parametrize("inp,out", [
    (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (-1.0, -1.0),
    (2.0, 1.0), (-3.0, -1.0),
])
def test_throttle_clamped_to_unit(inp, out):
    assert throttle_norm_clamp(inp) == out


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_throttle_non_finite_is_safe_coast(bad):
    # Non-finite must coast (0.0), never propagate or raise.
    assert throttle_norm_clamp(bad) == 0.0


def test_safe_stop_is_centred():
    assert safe_stop_steering_deg() == 0.0

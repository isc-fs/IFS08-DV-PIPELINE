"""Pure unit-conversion helpers for the car sensor bridge.

No rclpy / ROS imports — every function here is a plain numeric
transform so it can be exhaustively unit-tested in a bare pytest
environment (see test/test_conversions.py). The ROS node
(car_sensor_bridge_node) does nothing but I/O plumbing around these.

Why these conversions exist (verified against the firmware + EKF):

  * Steering. The uDV publishes /steering/angle_sensor in DEGREES
    (IFS08-DV-uDV: `steering_msg.data = can_c_get_steering_angle_deg()`),
    but the EKF's kinematic-bicycle yaw cross-check feeds δ through
    `tan(δ)` and is documented as RADIANS
    (odometry_filter::push_steering(t, angle_rad)). A raw topic remap
    can't convert units, hence this bridge.

  * Wheel speed. The uDV exposes NO motor-RPM / wheel-speed topic at
    all. The EKF consumes /motor_rpm (std_msgs/Float32) and scales it
    to m/s with its own `ekf.rpm_to_ms` parameter, so /motor_rpm must
    carry MOTOR shaft RPM. The car's source is the inverter, which
    typically reports ELECTRICAL RPM (eRPM); mechanical motor RPM =
    eRPM / pole_pairs. See docs/CAR_ADAPTATION.md for the open items.
"""
from __future__ import annotations

import math


def steering_sensor_deg_to_rad(
    sensor_deg: float,
    *,
    steering_ratio: float = 1.0,
    sign: float = 1.0,
    offset_deg: float = 0.0,
) -> float:
    """Convert the uDV steering sensor (degrees) to the EKF's δ (radians).

    Args:
        sensor_deg: raw value from /steering/angle_sensor, in degrees.
        steering_ratio: steering-wheel-to-road-wheel ratio. If the
            sensor measures the *road-wheel* angle directly, leave this
            at 1.0. If it measures the steering-wheel / column angle,
            set it to the mechanical ratio so the result is the
            road-wheel δ the bicycle model expects. MUST be confirmed
            against the steering geometry (flagged in CAR_ADAPTATION.md).
        sign: +1.0 or -1.0 to match the EKF sign convention (positive δ
            ⇒ positive yaw rate, i.e. left turn positive). Flip if the
            sensor's positive direction is opposite.
        offset_deg: mechanical zero offset (degrees) subtracted before
            scaling, for a sensor whose electrical zero isn't the
            mechanical straight-ahead.

    Returns:
        Road-wheel steering angle in radians.

    Raises:
        ValueError: if steering_ratio is zero (would divide by zero) or
            any argument is non-finite.
    """
    for name, val in (
        ("sensor_deg", sensor_deg),
        ("steering_ratio", steering_ratio),
        ("sign", sign),
        ("offset_deg", offset_deg),
    ):
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val!r}")
    if steering_ratio == 0.0:
        raise ValueError("steering_ratio must be non-zero")

    road_wheel_deg = sign * (sensor_deg - offset_deg) / steering_ratio
    return math.radians(road_wheel_deg)


def erpm_to_motor_rpm(erpm: float, pole_pairs: int) -> float:
    """Convert inverter electrical RPM to mechanical motor-shaft RPM.

    motor_rpm = erpm / pole_pairs.

    Args:
        erpm: electrical RPM as reported by the inverter.
        pole_pairs: motor pole-pair count (must be a positive integer).

    Raises:
        ValueError: if pole_pairs < 1 or erpm is non-finite.
    """
    if not math.isfinite(erpm):
        raise ValueError(f"erpm must be finite, got {erpm!r}")
    if pole_pairs < 1:
        raise ValueError(f"pole_pairs must be >= 1, got {pole_pairs!r}")
    return erpm / float(pole_pairs)


def inverter_value_to_motor_rpm(
    raw: float,
    *,
    is_erpm: bool,
    pole_pairs: int,
    scale: float = 1.0,
) -> float:
    """Map a raw inverter reading to motor-shaft RPM for /motor_rpm.

    The inverter's exact reporting convention is not yet pinned (no CAN
    on the LattePanda, no confirmed inverter topic), so this is kept
    deliberately general and parameter-driven:

        rpm = (raw * scale) [/ pole_pairs if is_erpm]

    Args:
        raw: value from the inverter feed.
        is_erpm: True if `raw` is electrical RPM (divide by pole_pairs);
            False if it is already mechanical motor RPM.
        pole_pairs: motor pole-pair count (used only when is_erpm).
        scale: linear units scale applied first (e.g. to fold in a
            fixed-point LSB or a gearbox ratio if /motor_rpm should
            carry wheel RPM instead — keep at 1.0 unless needed).

    Raises:
        ValueError: on non-finite input or pole_pairs < 1 when is_erpm.
    """
    if not math.isfinite(raw):
        raise ValueError(f"raw must be finite, got {raw!r}")
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale!r}")
    scaled = raw * scale
    if is_erpm:
        return erpm_to_motor_rpm(scaled, pole_pairs)
    return scaled

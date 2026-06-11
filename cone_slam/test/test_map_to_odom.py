"""Unit tests for cone_graph_slam_node.compute_map_to_odom — the SE(2)
math that produces the `map → odom` drift correction from SLAM's
absolute pose and sim_supervisor's dead-reckoning at the same instant.

Pure-Python (numpy only); no rclpy. Run with:
    cd pipeline/cone_slam && python3 -m pytest test/test_map_to_odom.py -v

The function is the math half of slam_node._publish_map_to_odom; pulling
it out at module level keeps it testable without standing up the whole
LifecycleNode + GTSAM stack.

Reference identity check used throughout:
    map → odom → base_link  should resolve to  slam_pose
    i.e. T_map_odom · T_odom_base == T_map_base
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cone_slam.tf_math import compute_map_to_odom


def compose_se2(a, b):
    """Compose two SE(2) poses (a · b) returning (x, y, yaw).
    Each input is (x, y, yaw).
    """
    ax, ay, ayaw = a
    bx, by, byaw = b
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (
        ax + c * bx - s * by,
        ay + s * bx + c * by,
        ((ayaw + byaw + math.pi) % (2 * math.pi)) - math.pi,
    )


# ---------------------------------------------------------------------
# Identity cases
# ---------------------------------------------------------------------
def test_identity_when_slam_matches_supervisor():
    """If SLAM and supervisor agree, map→odom is identity."""
    dx, dy, dyaw = compute_map_to_odom(
        slam_x=5.0, slam_y=3.0, slam_yaw=0.7,
        sup_x=5.0,  sup_y=3.0,  sup_yaw=0.7,
    )
    assert math.isclose(dx, 0.0, abs_tol=1e-9)
    assert math.isclose(dy, 0.0, abs_tol=1e-9)
    assert math.isclose(dyaw, 0.0, abs_tol=1e-9)


def test_identity_at_origin():
    dx, dy, dyaw = compute_map_to_odom(0, 0, 0, 0, 0, 0)
    assert math.isclose(dx, 0.0, abs_tol=1e-9)
    assert math.isclose(dy, 0.0, abs_tol=1e-9)
    assert math.isclose(dyaw, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------
# Pure-translation drift
# ---------------------------------------------------------------------
def test_pure_translation_drift():
    """SLAM at (5, 0, 0), supervisor at (10, 0, 0) — same heading. The
    odom frame's origin is at (-5, 0) in map.

    Verification: walking map → odom (-5,0) → base_link (+10,0,0) gives
    (5, 0) in map = slam_pose. ✓
    """
    dx, dy, dyaw = compute_map_to_odom(
        slam_x=5.0, slam_y=0.0, slam_yaw=0.0,
        sup_x=10.0, sup_y=0.0, sup_yaw=0.0,
    )
    assert math.isclose(dx, -5.0, abs_tol=1e-9)
    assert math.isclose(dy, 0.0, abs_tol=1e-9)
    assert math.isclose(dyaw, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------
# Round-trip property: map→odom · odom→base_link == map→base_link
# ---------------------------------------------------------------------
@pytest.mark.parametrize("slam_x,slam_y,slam_yaw,sup_x,sup_y,sup_yaw", [
    (10.0, 0.0, 0.0,      8.0, 0.0, 0.0),                       # x-axis drift
    (0.0,  5.0, math.pi/4, 0.0, 4.0, math.pi/4),                # diagonal drift, yaw
    (3.0, -2.0, -math.pi/6, 5.0, 1.0, 0.0),                     # mixed
    (-15.0, 20.0, math.pi - 0.1, -12.0, 18.0, -math.pi + 0.05),  # near-wrap yaw
    (100.0, -50.0, 1.2, 95.0, -48.0, 1.5),                      # far-field drift
])
def test_roundtrip_identity(slam_x, slam_y, slam_yaw, sup_x, sup_y, sup_yaw):
    """For any inputs, composing T_map_odom with T_odom_base must yield
    T_map_base (== slam pose). This is the defining identity."""
    dx, dy, dyaw = compute_map_to_odom(
        slam_x, slam_y, slam_yaw, sup_x, sup_y, sup_yaw,
    )
    T_map_odom = (dx, dy, dyaw)
    T_odom_base = (sup_x, sup_y, sup_yaw)
    T_map_base_computed = compose_se2(T_map_odom, T_odom_base)
    # Slam yaw also gets wrapped to (-π, π] for comparison
    slam_yaw_wrapped = ((slam_yaw + math.pi) % (2 * math.pi)) - math.pi
    assert math.isclose(T_map_base_computed[0], slam_x, abs_tol=1e-9)
    assert math.isclose(T_map_base_computed[1], slam_y, abs_tol=1e-9)
    assert math.isclose(T_map_base_computed[2], slam_yaw_wrapped, abs_tol=1e-9)


# ---------------------------------------------------------------------
# Yaw wrapping
# ---------------------------------------------------------------------
def test_yaw_wrap_positive():
    """dyaw computed > π should wrap into (-π, π]."""
    dx, dy, dyaw = compute_map_to_odom(
        slam_x=0.0, slam_y=0.0, slam_yaw=math.pi - 0.1,
        sup_x=0.0,  sup_y=0.0,  sup_yaw=-math.pi + 0.1,
    )
    # Raw difference = 2π - 0.2, should wrap to -0.2
    assert math.isclose(dyaw, -0.2, abs_tol=1e-9)


def test_yaw_wrap_negative():
    """dyaw computed < -π should wrap into (-π, π]."""
    dx, dy, dyaw = compute_map_to_odom(
        slam_x=0.0, slam_y=0.0, slam_yaw=-math.pi + 0.1,
        sup_x=0.0,  sup_y=0.0,  sup_yaw=math.pi - 0.1,
    )
    # Raw difference = -(2π - 0.2), should wrap to +0.2
    assert math.isclose(dyaw, 0.2, abs_tol=1e-9)


# ---------------------------------------------------------------------
# Realistic drift scenario from the #378 lap test
# ---------------------------------------------------------------------
def test_realistic_drift_scenario():
    """Reproduces the kind of drift /odom accumulates over a 41 s
    motion window (per #378's lap analysis): supervisor's odom drifted
    +21 % in path length; map→odom should compensate so map→base_link
    equals slam's absolute pose."""
    # Slam thinks we're at (50, 0) facing +x
    # Supervisor thinks we're at (60, 0) facing +x — +20 % overestimate
    dx, dy, dyaw = compute_map_to_odom(
        slam_x=50.0, slam_y=0.0, slam_yaw=0.0,
        sup_x=60.0,  sup_y=0.0,  sup_yaw=0.0,
    )
    # map→odom should put odom's origin at (-10, 0) in map
    assert math.isclose(dx, -10.0, abs_tol=1e-9)
    assert math.isclose(dy, 0.0, abs_tol=1e-9)
    assert math.isclose(dyaw, 0.0, abs_tol=1e-9)

    # Composition check: map→odom + odom→base = map→base = slam pose
    T_map_base = compose_se2((dx, dy, dyaw), (60.0, 0.0, 0.0))
    assert math.isclose(T_map_base[0], 50.0, abs_tol=1e-9)
    assert math.isclose(T_map_base[1], 0.0, abs_tol=1e-9)

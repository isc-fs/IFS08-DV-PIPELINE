#!/usr/bin/env python3
"""Cross-implementation numeric-equivalence check.

Runs the same input sequence through both the Python `OdometryFilter`
(sim_supervisor.odometry) and the C++ implementation (via a small CLI
harness — see the companion `validate_vs_python` binary built by this
package). Compares state.x, state.y, state.yaw, state.vx, state.vy
sample-by-sample.

This is the "sim = real-car" contract guard: any C++ change that
shifts a number more than 1e-9 from the Python reference shows up
here, not in a lap test. The Python implementation is the spec.

Run inside the container:

    cd /dv_pipeline_stack_ws/src/odometry_filter/test
    python3 validate_vs_python.py

The harness binary is the smaller-scope check we'll keep when Python
goes away — it embeds canned scenarios + exits non-zero on
divergence. This script is the bridge: it computes the Python
reference, calls the harness for the same input, diffs the two.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys


SCENARIOS = [
    # (name, [(t, kind, ...), ...])
    #   kind = "imu"   → (t, "imu", ax, ay, az, gx, gy, gz)
    #   kind = "rpm"   → (t, "rpm", value)
    #   kind = "steer" → (t, "steer", angle_rad)
    #   kind = "brake" → (t, "brake", brake_value)
    #
    # Each scenario starts with a stationary-calibration ramp.
]


def _calibration_prefix(n: int = 1500, dt: float = 0.0025):
    return [(i * dt, "imu", 0.0, 0.0, 9.81, 0.0, 0.0, 0.0) for i in range(n)]


def build_scenarios():
    # ----- Scenario 1: pure RPM ramp after calibration -----
    rpm_events = []
    for i in range(200):
        rpm_events.append((i * 0.0125, "rpm", 100.0))
    SCENARIOS.append(("rpm_ramp", _calibration_prefix() + rpm_events))

    # ----- Scenario 2: constant gyro yaw integration -----
    yaw_events = []
    t0 = 1500 * 0.0025
    for i in range(400):
        yaw_events.append((t0 + i * 0.0025, "imu", 0.0, 0.0, 9.81, 0.0, 0.0, 0.5))
    SCENARIOS.append(("yaw_integrate", _calibration_prefix() + yaw_events))

    # ----- Scenario 3: brake-on / brake-off RPM blend -----
    brake_events = [
        (0.0, "brake", 0.5),
        (0.0, "rpm", 100.0),
        (0.1, "brake", 0.0),
        (0.1, "rpm", 100.0),
    ]
    SCENARIOS.append(("brake_blend", _calibration_prefix() + brake_events))

    # ----- Scenario 4: full motion (RPM + position integration) -----
    motion_events = []
    t0 = 1500 * 0.0025
    for i in range(200):
        motion_events.append((i * 0.0125, "rpm", 100.0))
    for i in range(400):
        if i % 5 == 0:
            motion_events.append((0.625 + i * 0.0025, "rpm", 100.0))
        motion_events.append((t0 + i * 0.0025, "imu", 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
    SCENARIOS.append(("position_integration", _calibration_prefix() + motion_events))


def run_python_reference(events):
    import numpy as np
    from sim_supervisor.odometry import OdometryFilter

    f = OdometryFilter()
    for ev in events:
        kind = ev[1]
        if kind == "imu":
            t, _, ax, ay, az, gx, gy, gz = ev
            f.push_imu(t, np.array([ax, ay, az]), np.array([gx, gy, gz]))
        elif kind == "rpm":
            t, _, rpm = ev
            f.push_rpm(t, rpm)
        elif kind == "steer":
            t, _, angle = ev
            f.push_steering(t, angle)
        elif kind == "brake":
            t, _, brake = ev
            f.push_brake(t, brake)
    s = f.state
    return (s.x, s.y, s.yaw, s.vx, s.vy, s.yaw_rate)


def run_cpp_harness(events, harness_path):
    """Stream events on stdin, parse single-line output: x y yaw vx vy yaw_rate."""
    lines = []
    for ev in events:
        lines.append(" ".join(str(x) for x in ev))
    proc = subprocess.run(
        [harness_path],
        input="\n".join(lines).encode("utf-8"),
        capture_output=True,
        check=True,
    )
    parts = proc.stdout.decode().strip().split()
    return tuple(float(x) for x in parts)


def main():
    build_scenarios()
    harness = os.environ.get(
        "ODOMETRY_FILTER_HARNESS",
        "/dv_pipeline_stack_ws/install/odometry_filter/lib/odometry_filter/validate_vs_python_harness",
    )
    if not os.path.isfile(harness):
        print(f"harness not found: {harness}", file=sys.stderr)
        sys.exit(2)

    tol = 1e-9
    failed = 0
    for name, events in SCENARIOS:
        py = run_python_reference(events)
        cpp = run_cpp_harness(events, harness)
        diffs = [abs(p - c) for p, c in zip(py, cpp)]
        max_diff = max(diffs)
        ok = max_diff < tol
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name:25s}  max_diff={max_diff:.3e}")
        if not ok:
            print(f"      py : x={py[0]:+.9f} y={py[1]:+.9f} yaw={py[2]:+.9f}")
            print(f"          vx={py[3]:+.9f} vy={py[4]:+.9f} yaw_rate={py[5]:+.9f}")
            print(f"      cpp: x={cpp[0]:+.9f} y={cpp[1]:+.9f} yaw={cpp[2]:+.9f}")
            print(f"          vx={cpp[3]:+.9f} vy={cpp[4]:+.9f} yaw_rate={cpp[5]:+.9f}")
            failed += 1

    if failed:
        print(f"\n{failed} scenarios diverged (tol={tol}).")
        sys.exit(1)
    print(f"\nAll {len(SCENARIOS)} scenarios match to {tol}.")


if __name__ == "__main__":
    main()

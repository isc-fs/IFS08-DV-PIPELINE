"""Safety invariant: hard_stop_on_finish must default FALSE on the car.

`hard_stop_on_finish` enables DV_STOPPING — a full-pressure ASB stop at every
mission finish. It is safe to ENABLE only after the three-stage bench validation
(docs/HARD_STOP_BENCH.md) with byte-7 firmware flashed. Until then, and for every
normal run afterwards, it MUST be off.

Making it a launch argument (not a source default flipped by hand) is the whole
safety mechanism: enabling becomes a deliberate, per-run `hard_stop_on_finish:=true`
at launch, and the codebase's default stays safe. This test pins that the default
never silently drifts to true — a one-character regression there arms a
full-pressure stop on the next flashed run.

Source-level on purpose: the launch file imports `launch`/`launch_ros`, which need
ROS, but the invariant is a textual property of the DeclareLaunchArgument, so we
assert it directly and it runs in plain pytest with no ROS.
"""
from __future__ import annotations

import os
import re

_HERE = os.path.dirname(__file__)
LAUNCH = os.path.abspath(
    os.path.join(_HERE, os.pardir, "launch", "car_pipeline.launch.py"))
COMMON = os.path.abspath(
    os.path.join(_HERE, os.pardir, "bringup", "launch_common.py"))


def _src(path: str) -> str:
    with open(path) as fh:
        return fh.read()


def test_hard_stop_launch_arg_defaults_false():
    """The declared default must be exactly "false"."""
    src = _src(LAUNCH)
    m = re.search(
        r'DeclareLaunchArgument\(\s*["\']hard_stop_on_finish["\']\s*,\s*'
        r'default_value\s*=\s*["\']([^"\']+)["\']',
        src)
    assert m is not None, \
        "car_pipeline.launch.py must declare a hard_stop_on_finish launch arg"
    assert m.group(1) == "false", (
        f'hard_stop_on_finish launch default is "{m.group(1)}", MUST be "false" '
        f"— a true default arms a full-pressure ASB stop on the next flashed run")


def test_hard_stop_is_wired_through_to_mission_control():
    """The arg must actually reach management_actions, not just be declared."""
    launch = _src(LAUNCH)
    assert 'hard_stop_on_finish=LaunchConfiguration("hard_stop_on_finish")' \
        in launch, "the launch arg is declared but never passed to management_actions"

    common = _src(COMMON)
    assert "hard_stop_on_finish" in common, \
        "management_actions must accept and forward hard_stop_on_finish"
    # It must set the mission_control parameter of the same name.
    assert '"hard_stop_on_finish"' in common


def test_no_launch_file_hardcodes_hard_stop_true():
    """Belt-and-braces: no launch file may bake the flag on."""
    launch_dir = os.path.abspath(os.path.join(_HERE, os.pardir, "launch"))
    for fn in os.listdir(launch_dir):
        if not fn.endswith(".launch.py"):
            continue
        src = _src(os.path.join(launch_dir, fn))
        # Guard against default_value="true" and a direct =True pass-through.
        bad = re.search(
            r'hard_stop_on_finish["\']?\s*,\s*default_value\s*=\s*["\']true["\']',
            src)
        assert bad is None, f"{fn} defaults hard_stop_on_finish to true"
        assert "hard_stop_on_finish=True" not in src.replace(" ", ""), \
            f"{fn} hardcodes hard_stop_on_finish=True"

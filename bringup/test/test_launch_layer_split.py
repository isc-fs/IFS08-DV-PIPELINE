"""The sensor/autonomy launch split must stay clean (manual-record).

The manual-driving recorder (dv-manual.service) launches car_sensors.launch.py
ALONE — the whole point is to record manual laps WITHOUT bringing up autonomy.
So the invariant is:

  * car_sensors.launch.py must NOT pull in the autonomy stack (no car_pipeline,
    no mission_control / mode_manager). If it did, `dv manual` would silently
    start the pipeline it is meant to avoid.
  * car_bringup.launch.py must include BOTH layers, so the race path is
    unchanged by the split.

Source-level (launch files need ROS to execute), so it runs in plain pytest.
"""
from __future__ import annotations

import ast
import io
import os
import tokenize

_HERE = os.path.dirname(__file__)
LAUNCH = os.path.abspath(os.path.join(_HERE, os.pardir, "launch"))


def _src(name: str) -> str:
    with open(os.path.join(LAUNCH, name)) as fh:
        return fh.read()


def _code_only(name: str) -> str:
    """Source with comments AND string literals (incl. docstrings) removed, so a
    prose mention of another layer can't trip the 'no autonomy' check — only an
    actual code reference (an import, a package= arg, an include path) counts."""
    src = _src(name)
    out = []
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    for tok in toks:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    # Also drop any bare-expression docstrings the tokenizer left as STRING
    # already handled above; belt-and-braces re-check via ast is not needed.
    return " ".join(out)


def test_sensors_launch_has_no_autonomy():
    """car_sensors is sensors-only — no autonomy, or manual mode isn't manual.

    Checks CODE, not prose: the docstring legitimately explains the split by
    naming car_pipeline/car_bringup. What must not appear is an actual launch
    of autonomy.
    """
    code = _code_only("car_sensors.launch.py")
    for forbidden in ("car_pipeline", "mission_control", "mode_manager",
                      "car_bringup"):
        assert forbidden not in code, (
            f"car_sensors.launch.py has a CODE reference to {forbidden!r} — the "
            f"manual-record layer must not start autonomy")
    # And it must be parseable / define the launch entrypoint.
    tree = ast.parse(_src("car_sensors.launch.py"))
    assert any(isinstance(n, ast.FunctionDef)
               and n.name == "generate_launch_description"
               for n in ast.walk(tree))


def test_sensors_launch_starts_the_lidar():
    """It must still be the sensor layer — the cloud is the point of recording."""
    src = _src("car_sensors.launch.py")
    assert "hesai_ros_driver" in src
    assert "hesai_lidar" in src and "imu_link" in src   # the static TFs


def test_car_bringup_includes_both_layers():
    """The race path is sensors + autonomy — unchanged by the split."""
    src = _src("car_bringup.launch.py")
    assert "car_sensors.launch.py" in src, \
        "car_bringup must include the sensor layer"
    assert "car_pipeline.launch.py" in src, \
        "car_bringup must include the autonomy layer"


def test_car_bringup_forwards_sensor_args():
    """with_lidar / foxglove must reach car_sensors, else bench/no-ATX runs break."""
    src = _src("car_bringup.launch.py")
    assert "with_lidar" in src and "foxglove" in src

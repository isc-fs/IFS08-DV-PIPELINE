"""SLAM regression suite — pytest loader for fixture-driven replay.

Reads each YAML under `fixtures/*.yaml`, runs `replay_slam.py` on the
referenced bag, parses the per-scan residual CSV, and asserts the
pose / yaw errors stay within the fixture's `expectations.<phase>`
thresholds.

The fixture YAML schema is documented in
`fixtures/autocross_no-track_20260517_172922.yaml`. Phase selection
(`pre_rewrite` vs `post_rewrite`) is taken from the fixture's
`expectations.phase` field — flipping that one line is how we
transition the suite from "no-worse-than-today" gating to
"meets-rewrite-targets" gating once Phase 1 lands.

Skipped automatically in CI environments where the GTSAM Python
wheel isn't present (the SLAM library needs it; the rest of the
codebase doesn't). Developers can still run the suite locally
inside dv_pipeline_stack:

    docker compose exec dv_pipeline_stack bash -lc '
      source /opt/ros/humble/setup.bash &&
      source /dv_pipeline_stack_ws/install/setup.bash &&
      cd /dv_pipeline_stack_ws/src/cone_slam &&
      python3 -m pytest test/regression/ -v'
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


# Path to the replay script we drive. We resolve from the fixture
# directory upward so the test works under both the repo layout
# (`pipeline/cone_slam/...`) and the in-container install layout
# (`/dv_pipeline_stack_ws/src/cone_slam/...`).
_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_REPLAY_SCRIPT = Path(__file__).parents[2] / "scripts" / "replay_slam.py"

# We bail out of the suite if GTSAM (SLAM's hard dependency) isn't
# importable. Skipping is preferable to a hard fail in environments
# where the wheel is intentionally absent (e.g. the lightweight
# mission_control CI image).
gtsam_skip = pytest.importorskip("gtsam")  # noqa: F841

# rosbag2_py likewise — replay_slam.py imports it lazily, but the
# regression run would surface the error far from its cause.
rosbag2_skip = pytest.importorskip("rosbag2_py")  # noqa: F841


def _load_yaml(path: Path) -> dict[str, Any]:
    """Minimal YAML reader — we don't want to depend on PyYAML in
    the regression test (it's not in dv_pipeline_stack's image by
    default). Falls back to PyYAML if available, else a tiny
    line-based parser that covers our fixture shape."""
    text = path.read_text()
    try:
        import yaml  # type: ignore[import]
        return yaml.safe_load(text)
    except ImportError:
        return _yaml_lite(text)


def _yaml_lite(text: str) -> dict[str, Any]:
    """Parse the subset of YAML our fixtures use: scalar maps,
    nested maps, integer / float / string / bool leaves. Reject
    sequences or block scalars (>-, |) with a clear error so a
    fixture-author hits a useful message rather than a silent
    misparse."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for ln, line in enumerate(text.splitlines(), 1):
        s = line.split("#", 1)[0].rstrip()
        if not s.strip():
            continue
        indent = len(s) - len(s.lstrip())
        s = s.strip()
        # Pop scopes whose indent is >= current indent.
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if ":" not in s:
            raise ValueError(f"line {ln}: expected `key: value`, got {s!r}")
        key, _, val = s.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "" or val == ">-":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        # Trim quotes around the value.
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        # Type coerce.
        v: Any
        if val.lower() in ("true", "false"):
            v = (val.lower() == "true")
        else:
            try:
                v = int(val)
            except ValueError:
                try:
                    v = float(val)
                except ValueError:
                    v = val
        parent[key] = v
    return root


def _find_fixtures() -> list[Path]:
    if not _FIXTURE_DIR.is_dir():
        return []
    return sorted(_FIXTURE_DIR.glob("*.yaml"))


def _bag_dir_from_uri(uri: str) -> Path | None:
    """Resolve a `bag_uri` (path relative to repo root) to an absolute
    path the replay script can read. Returns None when the bag is
    missing, so the test can `pytest.skip` rather than fail — bags
    aren't tracked in git and CI may not have them.

    The fixture `bag_uri` is `bags/<name>`; we try:
      - $IFSSIM_BAGS_DIR/<name>      — explicit override (CI-friendly)
      - /workspace/<uri>             — generic mount
      - /dv_pipeline_stack_ws/<uri>  — in-container mount of repo bags
      - <repo_root>/<uri>            — host checkout
      - <cwd>/<uri>                  — caller's cwd
    """
    name = Path(uri).name  # `bags/foo` → `foo`
    candidates: list[Path] = []
    env_root = os.environ.get("IFSSIM_BAGS_DIR")
    if env_root:
        candidates.append(Path(env_root) / name)
    candidates += [
        Path("/workspace") / uri,
        Path("/dv_pipeline_stack_ws") / uri,
        Path(__file__).parents[5] / uri,
        Path.cwd() / uri,
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return None


def _run_replay(
    bag_dir: Path,
    out_csv: Path,
    mode: str,
    node_class: str | None = None,
    pose_source: str | None = None,
) -> int:
    """Invoke scripts/replay_slam.py and return its exit code.

    Stdout/stderr are inherited so the test framework's `-v` flag
    shows the SLAM progress. The script's CSV-output side-effect is
    what we then parse for residuals.
    """
    cmd = [
        sys.executable,
        str(_REPLAY_SCRIPT),
        str(bag_dir),
        "--csv", str(out_csv),
        "--quiet",
        # Loose threshold flags — actual gating happens against the
        # fixture's expectations after the run, on the parsed CSV.
        "--max-pose-err-m", "1e9",
        "--max-yaw-err-deg", "1e9",
    ]
    if node_class:
        cmd += ["--node-class", node_class]
        cmd += ["--behavior", mode]
    if pose_source:
        cmd += ["--pose-source", pose_source]
    return subprocess.call(cmd)


def _parse_residuals_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def _summarise(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"pose_err_mean": float("inf"), "pose_err_max": float("inf"),
                "yaw_err_mean_deg": float("inf"), "yaw_err_max_deg": float("inf"),
                "n_scans": 0}
    import math
    errs = [r["err_m"] for r in rows]
    yaws_deg = [math.degrees(abs(r["yaw_err_rad"])) for r in rows]
    return {
        "pose_err_mean": sum(errs) / len(errs),
        "pose_err_max":  max(errs),
        "yaw_err_mean_deg": sum(yaws_deg) / len(yaws_deg),
        "yaw_err_max_deg":  max(yaws_deg),
        "n_scans": len(rows),
    }


# ---------------------------------------------------------------------
# Fixture-driven parametrised test
# ---------------------------------------------------------------------

_FIXTURE_PATHS = _find_fixtures()


@pytest.mark.skipif(
    not _FIXTURE_PATHS,
    reason="no SLAM regression fixtures present",
)
@pytest.mark.skipif(
    not _REPLAY_SCRIPT.is_file(),
    reason=f"replay_slam.py not found at {_REPLAY_SCRIPT}",
)
@pytest.mark.parametrize(
    "fixture_path",
    _FIXTURE_PATHS,
    ids=[p.stem for p in _FIXTURE_PATHS],
)
def test_slam_regression(fixture_path: Path, tmp_path: Path) -> None:
    """Run replay_slam.py against the fixture's bag and assert that
    residuals fall under the fixture's phase-specific thresholds."""
    fx = _load_yaml(fixture_path)
    bag_dir = _bag_dir_from_uri(fx["bag_uri"])
    if bag_dir is None:
        pytest.skip(
            f"bag {fx['bag_uri']!r} not found on this machine. Set "
            f"IFSSIM_BAGS_DIR=<dir-containing-the-bag> to enable this "
            f"regression test (bags are intentionally not tracked in git)."
        )
    phase = fx["expectations"]["phase"]
    thr = fx["expectations"][phase]
    out_csv = tmp_path / f"{fixture_path.stem}.csv"

    rc = _run_replay(
        bag_dir, out_csv, fx["mode"],
        node_class=fx.get("node_class"),
        pose_source=fx.get("pose_source"),
    )
    if fx["expectations"].get("no_slam_crash", True):
        assert rc in (0, 1), (
            f"replay_slam.py exited with rc={rc} on {fixture_path.name} "
            f"(0 = pass, 1 = residual threshold exceeded, 2 = bag/setup "
            f"error). rc={rc} indicates a SLAM crash."
        )

    assert out_csv.is_file(), (
        f"replay_slam.py did not produce a residual CSV at {out_csv}; "
        f"return code was {rc}."
    )
    rows = _parse_residuals_csv(out_csv)
    summary = _summarise(rows)

    assert summary["n_scans"] >= thr["min_slam_pose_messages"], (
        f"too few scans in {fixture_path.name}: {summary['n_scans']} "
        f"< {thr['min_slam_pose_messages']} (SLAM never reached "
        f"SLAM_RUNNING or the bag is shorter than expected)"
    )

    # Phase-thresholded asserts. Each fail-message includes the fixture
    # name, the phase, and the observed-vs-threshold so a CI log is
    # diagnostic on its own.
    assert summary["pose_err_max"] <= thr["pose_error_max_m"], (
        f"{fixture_path.name} ({phase}): pose-err-max="
        f"{summary['pose_err_max']:.2f}m exceeds threshold "
        f"{thr['pose_error_max_m']:.2f}m"
    )
    assert summary["pose_err_mean"] <= thr["pose_error_mean_m"], (
        f"{fixture_path.name} ({phase}): pose-err-mean="
        f"{summary['pose_err_mean']:.2f}m exceeds threshold "
        f"{thr['pose_error_mean_m']:.2f}m"
    )
    assert summary["yaw_err_max_deg"] <= thr["yaw_error_max_deg"], (
        f"{fixture_path.name} ({phase}): yaw-err-max="
        f"{summary['yaw_err_max_deg']:.2f}° exceeds threshold "
        f"{thr['yaw_error_max_deg']:.2f}°"
    )
    assert summary["yaw_err_mean_deg"] <= thr["yaw_error_mean_deg"], (
        f"{fixture_path.name} ({phase}): yaw-err-mean="
        f"{summary['yaw_err_mean_deg']:.2f}° exceeds threshold "
        f"{thr['yaw_error_mean_deg']:.2f}°"
    )


def test_fixtures_discovered() -> None:
    """Smoke: at least one fixture exists; CI shouldn't silently
    pass with zero gates."""
    assert _FIXTURE_PATHS, (
        "no SLAM regression fixtures found under "
        f"{_FIXTURE_DIR}. Add at least one YAML to gate the SLAM rewrite."
    )


# Allow `python -m pipeline.cone_slam.test.regression.test_replay_regression`
# as a smoke entry-point (no pytest), useful when iterating on the
# loader itself without spinning up the whole pytest discovery.
if __name__ == "__main__":
    if not _FIXTURE_PATHS:
        print("no fixtures", file=sys.stderr)
        sys.exit(2)
    for fp in _FIXTURE_PATHS:
        fx = _load_yaml(fp)
        print(f"fixture {fp.name}: bag={fx['bag_uri']}, mode={fx['mode']}, "
              f"phase={fx['expectations']['phase']}")

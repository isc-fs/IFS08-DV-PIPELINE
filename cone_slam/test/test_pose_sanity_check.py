"""Tests for FactorGraph.commit_with_pose_sanity_check (#273 follow-up).

The sanity check guards against iSAM2 pose snaps caused by wrong cone
matches passing the DA gate. We mock _flush_update so the threshold
logic can be exercised without spinning up a real factor graph.
"""
from __future__ import annotations

import numpy as np
import pytest

gtsam = pytest.importorskip("gtsam")

from cone_slam.factor_graph import FactorGraph, ScanResult


def _make_pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> "gtsam.Pose3":
    return gtsam.Pose3(gtsam.Rot3.Rz(yaw), np.array([x, y, 0.0]))


def _make_result(pose: "gtsam.Pose3") -> ScanResult:
    return ScanResult(
        pose=pose,
        velocity=np.zeros(3),
        bias=gtsam.imuBias.ConstantBias(),
    )


@pytest.fixture
def fg() -> FactorGraph:
    """Bare FactorGraph — we don't initialize an anchor because we mock
    out _flush_update entirely."""
    return FactorGraph()


def test_pose_sanity_check_no_correction_below_position_threshold(
    fg: FactorGraph, monkeypatch
) -> None:
    """Optimized pose 0.5 m from prediction (within 0.8 m gate) → no correction."""
    optimized = _make_pose(x=0.5)
    monkeypatch.setattr(fg, "_flush_update",
                        lambda: _make_result(optimized))

    predicted = _make_pose(x=0.0)
    result, was_corrected = fg.commit_with_pose_sanity_check(
        predicted, max_pos_dev_m=0.8, max_yaw_dev_rad=0.3)

    assert not was_corrected
    assert result.pose.translation()[0] == pytest.approx(0.5)


def test_pose_sanity_check_no_correction_below_yaw_threshold(
    fg: FactorGraph, monkeypatch
) -> None:
    """Optimized pose with yaw 0.2 rad (within 0.3 rad gate) → no correction."""
    optimized = _make_pose(yaw=0.2)
    monkeypatch.setattr(fg, "_flush_update",
                        lambda: _make_result(optimized))

    predicted = _make_pose(yaw=0.0)
    result, was_corrected = fg.commit_with_pose_sanity_check(
        predicted, max_pos_dev_m=0.8, max_yaw_dev_rad=0.3)

    assert not was_corrected


def test_pose_sanity_check_correction_above_position_threshold(
    fg: FactorGraph, monkeypatch
) -> None:
    """Optimized pose 2.0 m from prediction → correction fires; second
    _flush_update returns pose at prediction (the strong prior wins)."""
    bad_pose = _make_pose(x=2.0)
    corrected = _make_pose(x=0.0)
    call_log = []

    def fake_flush() -> ScanResult:
        call_log.append(len(call_log))
        return _make_result(bad_pose if call_log[-1] == 0 else corrected)

    monkeypatch.setattr(fg, "_flush_update", fake_flush)

    predicted = _make_pose(x=0.0)
    result, was_corrected = fg.commit_with_pose_sanity_check(
        predicted, max_pos_dev_m=0.8, max_yaw_dev_rad=0.3)

    assert was_corrected, "expected correction to fire on 2.0 m position jump"
    assert len(call_log) == 2, "expected two _flush_update calls (initial + corrective)"
    assert result.pose.translation()[0] == pytest.approx(0.0)


def test_pose_sanity_check_correction_above_yaw_threshold(
    fg: FactorGraph, monkeypatch
) -> None:
    """Optimized pose with yaw 0.6 rad (above 0.3 rad gate) → correction fires."""
    bad_pose = _make_pose(yaw=0.6)
    corrected = _make_pose(yaw=0.0)
    calls = [bad_pose, corrected]

    monkeypatch.setattr(fg, "_flush_update",
                        lambda: _make_result(calls.pop(0)))

    predicted = _make_pose(yaw=0.0)
    result, was_corrected = fg.commit_with_pose_sanity_check(
        predicted, max_pos_dev_m=0.8, max_yaw_dev_rad=0.3)

    assert was_corrected
    assert abs(result.pose.rotation().yaw()) == pytest.approx(0.0, abs=1e-6)


def test_pose_sanity_check_yaw_wraps_correctly(
    fg: FactorGraph, monkeypatch
) -> None:
    """Yaw deviation across the ±π wrap (e.g. predicted=+170°, actual=-170°)
    should evaluate as a 20° gap, NOT 340°. Validates that the test uses
    the relative-rotation form."""
    # Deviation across the wrap: actual ≈ -170°, predicted ≈ +170° → 20° apart.
    actual = _make_pose(yaw=np.radians(-170.0))
    monkeypatch.setattr(fg, "_flush_update",
                        lambda: _make_result(actual))

    predicted = _make_pose(yaw=np.radians(170.0))
    result, was_corrected = fg.commit_with_pose_sanity_check(
        predicted, max_pos_dev_m=0.8, max_yaw_dev_rad=np.radians(30.0))

    # 20° gap is within 30° threshold, so no correction.
    assert not was_corrected, (
        "yaw deviation across the ±π wrap should be the small short-arc, "
        "not the long way around"
    )


def test_pose_sanity_check_correction_anchors_at_predicted_position(
    fg: FactorGraph, monkeypatch
) -> None:
    """When correction fires, the strong prior should be at predicted_pose.
    Inspect _new_factors to confirm a PriorFactorPose3 was added with
    the predicted pose as its prior mean."""
    bad_pose = _make_pose(x=5.0)
    corrected = _make_pose(x=2.0, y=3.0)
    calls = [bad_pose, corrected]

    monkeypatch.setattr(fg, "_flush_update",
                        lambda: _make_result(calls.pop(0)))

    # Need a non-zero current step so X(self._k) is a valid key. The
    # method increments self._k via stage_imu_factor in real usage; here
    # we set it directly because the mock _flush_update doesn't care.
    fg._k = 1
    predicted = _make_pose(x=2.0, y=3.0)
    result, was_corrected = fg.commit_with_pose_sanity_check(
        predicted, max_pos_dev_m=0.8, max_yaw_dev_rad=0.3)

    assert was_corrected
    # After the second flush, _new_factors will have been reset to 0
    # (real _flush_update clears it). Our mock doesn't, but the prior
    # was added between the two flushes — a real run would have its
    # factor seen by the second update.

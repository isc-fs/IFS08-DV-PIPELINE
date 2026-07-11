"""Tests for the pure `/slam/finished` lap/distance detector (#384).

Pins the completion contract the SLAM node relies on: lap counting with
re-arm hysteresis, the distance mode for straight missions, and — the
safety-critical part — that finish is never reported until the car is
stopped (entering AS Finished fires the EBS on the firmware).
"""
from __future__ import annotations

from cone_slam.lap_counter import LapCounter, LapCounterConfig

MOVING = 5.0    # m/s, well above the standstill threshold
STOPPED = 0.0   # m/s


def _drive_out_and_back(c: LapCounter, speed: float) -> None:
    """One lap: leave past the arm radius, then return inside the close
    radius (default geometry: arm 15 m, close 6 m)."""
    c.update(20.0, 0.0, speed)   # out past arm_radius → armed
    c.update(0.0, 0.0, speed)    # back inside close_radius → lap closes


def test_disabled_config_never_finishes() -> None:
    """No lap/distance criterion (e.g. skidpad) → never enabled, never fires."""
    c = LapCounter(LapCounterConfig())
    assert not c.enabled
    assert c.update(0.0, 0.0, STOPPED) is False
    assert c.update(30.0, 0.0, STOPPED) is False
    assert c.update(0.0, 0.0, STOPPED) is False
    assert not c.finished


def test_autocross_one_lap_then_stop() -> None:
    """autocross = 1 lap: the lap closes while moving (no finish), then the
    car stops → finish fires exactly once."""
    c = LapCounter(LapCounterConfig(laps_to_finish=1))
    assert c.update(0.0, 0.0, MOVING) is False      # at start, not armed
    assert c.update(20.0, 0.0, MOVING) is False     # armed
    assert c.update(0.0, 0.0, MOVING) is False      # lap 1 closes, but moving
    assert c.lap_count == 1
    assert c.update(0.0, 0.0, STOPPED) is True       # target met + stopped
    assert c.finished
    # Single rising edge: never fires again.
    assert c.update(0.0, 0.0, STOPPED) is False


def test_standstill_gate_blocks_finish_while_moving() -> None:
    """Target met but the car keeps rolling → no finish until it stops."""
    c = LapCounter(LapCounterConfig(laps_to_finish=1))
    _drive_out_and_back(c, MOVING)
    assert c.lap_count == 1
    for _ in range(20):
        assert c.update(1.0, 0.0, MOVING) is False   # still moving near origin
    assert not c.finished
    assert c.update(1.0, 0.0, STOPPED) is True


def test_trackdrive_counts_ten_laps() -> None:
    """trackdrive = 10 laps: does not finish at lap 9 (even stopped), only
    at lap 10 once stopped."""
    c = LapCounter(LapCounterConfig(laps_to_finish=10))
    for _ in range(9):
        _drive_out_and_back(c, MOVING)
    assert c.lap_count == 9
    assert c.update(0.0, 0.0, STOPPED) is False       # 9 < 10, not done
    _drive_out_and_back(c, MOVING)                     # lap 10
    assert c.lap_count == 10
    assert c.update(0.0, 0.0, STOPPED) is True
    assert c.finished


def test_rearm_hysteresis_no_double_count() -> None:
    """After a lap closes, jitter near origin without leaving past the arm
    radius must not count a second lap."""
    c = LapCounter(LapCounterConfig(laps_to_finish=2))
    _drive_out_and_back(c, MOVING)
    assert c.lap_count == 1
    # Wobble near the line without re-arming (never exceeds arm_radius).
    c.update(3.0, 0.0, MOVING)
    c.update(0.0, 0.0, MOVING)
    c.update(5.0, 0.0, MOVING)
    c.update(0.0, 0.0, MOVING)
    assert c.lap_count == 1                            # no phantom lap
    _drive_out_and_back(c, MOVING)                     # proper 2nd lap
    assert c.lap_count == 2


def test_start_line_jitter_does_not_count() -> None:
    """Sitting at the start (never past arm radius) counts no laps."""
    c = LapCounter(LapCounterConfig(laps_to_finish=1))
    for _ in range(10):
        c.update(2.0, 0.0, MOVING)
        c.update(0.0, 0.0, MOVING)
    assert c.lap_count == 0
    assert not c.finished


def test_accel_distance_then_stop() -> None:
    """accel = distance mode: never returns to origin. No finish while
    moving even past the distance; finish once stopped beyond it."""
    c = LapCounter(LapCounterConfig(laps_to_finish=0, finish_distance_m=75.0))
    assert c.update(50.0, 0.0, MOVING) is False       # short of distance
    assert c.update(80.0, 0.0, MOVING) is False       # past distance, moving
    assert c.update(120.0, 0.0, STOPPED) is True       # stopped past distance
    assert c.finished


def test_no_solve_speed_guard() -> None:
    """A caller passing +inf speed (no SLAM solve yet) can never finish
    even if the position criterion is met."""
    c = LapCounter(LapCounterConfig(finish_distance_m=10.0))
    assert c.update(50.0, 0.0, float("inf")) is False
    assert not c.finished


def test_reset_clears_state() -> None:
    """reset() returns the counter to a fresh run."""
    c = LapCounter(LapCounterConfig(laps_to_finish=1))
    _drive_out_and_back(c, MOVING)
    c.update(0.0, 0.0, STOPPED)
    assert c.finished and c.lap_count == 1
    c.reset()
    assert not c.finished and c.lap_count == 0
    assert c.update(0.0, 0.0, STOPPED) is False        # fresh: not armed yet

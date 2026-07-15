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


# ---------------------------------------------------------------------
# final_lap — the stop-anchor gate (#384 follow-up)
#
# Why this exists: control_node latched its stop anchor on the FIRST
# big-orange gate past stop_latch_min_travel, so trackdrive braked to a stop
# at the end of lap 1 and could never reach 10. It needs to know it is on the
# closing lap BEFORE the gate. It cannot use `finished` for that — `finished`
# requires standstill and the car only stops BECAUSE control braked, so gating
# the anchor on it would deadlock.
# ---------------------------------------------------------------------

def test_final_lap_true_immediately_for_single_lap_missions():
    """Autocross (1 lap): the first gate past the travel guard IS the finish,
    so the gate must be transparent from tick zero."""
    c = LapCounter(LapCounterConfig(laps_to_finish=1))
    assert c.final_lap is True


def test_final_lap_true_for_distance_missions():
    """Accel finishes on distance, not laps — nothing to gate on."""
    c = LapCounter(LapCounterConfig(finish_distance_m=75.0))
    assert c.final_lap is True


def test_final_lap_true_when_no_criterion():
    """Skidpad has no completion criterion; must not silently disarm the
    stop anchor and leave the car unable to ever stop."""
    c = LapCounter(LapCounterConfig())
    assert c.final_lap is True


def test_final_lap_false_until_closing_lap_for_trackdrive():
    """The actual fix: 10 laps → armed only for the 10th crossing."""
    c = LapCounter(LapCounterConfig(laps_to_finish=10))
    assert c.final_lap is False, "armed at lap 0 — would stop on lap 1"
    for lap in range(1, 9):
        _drive_out_and_back(c, MOVING)
        assert c.lap_count == lap
        assert c.final_lap is False, f"armed too early at lap {lap}"
    _drive_out_and_back(c, MOVING)          # 9th crossing complete
    assert c.lap_count == 9
    assert c.final_lap is True, "not armed for the closing lap"


def test_final_lap_stays_true_past_the_target():
    """Overshoot must not disarm the anchor and send the car round again."""
    c = LapCounter(LapCounterConfig(laps_to_finish=2))
    _drive_out_and_back(c, MOVING)
    assert c.final_lap is True
    _drive_out_and_back(c, MOVING)
    assert c.lap_count == 2
    assert c.final_lap is True


def test_trackdrive_finishes_only_after_ten_laps_and_standstill():
    """End-to-end on the pure core: 10 laps then stop → finished exactly once."""
    c = LapCounter(LapCounterConfig(laps_to_finish=10))
    for _ in range(9):
        _drive_out_and_back(c, MOVING)
    assert not c.finished
    # 10th crossing while still rolling — target met but not stopped yet.
    c.update(20.0, 0.0, MOVING)
    assert c.update(0.0, 0.0, MOVING) is False, "finished while still moving"
    assert c.lap_count == 10
    # Control brakes at the anchor; the car comes to rest on the line.
    assert c.update(0.0, 0.0, STOPPED) is True
    assert c.finished


# ---------------------------------------------------------------------
# target_met — the hard-stop trigger
#
# The car has no working service brake (regen is commanded and relayed but
# does nothing on the vehicle), so it cannot reach standstill unaided and
# `finished` would never fire. target_met is the earliest honest moment to say
# "the mission is over, bring it to rest" WITHOUT asserting a standstill that
# has not happened. Conflating the two would signal AS Finished at speed —
# which fires the EBS *and* opens the SDC.
# ---------------------------------------------------------------------

def test_target_met_false_before_the_criterion():
    c = LapCounter(LapCounterConfig(laps_to_finish=1))
    assert c.target_met is False
    c.update(20.0, 0.0, MOVING)          # out, not back yet
    assert c.target_met is False


def test_target_met_rises_while_still_moving():
    """The whole point: it must NOT wait for standstill."""
    c = LapCounter(LapCounterConfig(laps_to_finish=1))
    _drive_out_and_back(c, MOVING)
    assert c.lap_count == 1
    assert c.target_met is True
    assert c.finished is False, "finished must still require standstill"


def test_target_met_on_distance_missions():
    """Accel: crossing the finish distance at speed."""
    c = LapCounter(LapCounterConfig(finish_distance_m=75.0))
    c.update(50.0, 0.0, MOVING)
    assert c.target_met is False
    c.update(80.0, 0.0, MOVING)
    assert c.target_met is True
    assert c.finished is False


def test_target_met_latches_through_the_stop():
    """A criterion that could un-meet itself would release a stop already in
    progress. Accel drifting back under the distance must not do that."""
    c = LapCounter(LapCounterConfig(finish_distance_m=75.0))
    c.update(80.0, 0.0, MOVING)
    assert c.target_met
    c.update(70.0, 0.0, MOVING)          # back under the line
    assert c.target_met, "stop request released mid-stop"


def test_target_met_never_true_without_a_criterion():
    """Skidpad has no criterion — must never request a hard stop."""
    c = LapCounter(LapCounterConfig())
    for _ in range(5):
        _drive_out_and_back(c, MOVING)
    assert c.target_met is False


def test_target_met_precedes_finished_then_finished_follows():
    """The intended sequence: criterion → (brakes) → standstill → finished."""
    c = LapCounter(LapCounterConfig(laps_to_finish=1))
    _drive_out_and_back(c, MOVING)
    assert c.target_met and not c.finished     # request the hard stop
    assert c.update(0.0, 0.0, MOVING) is False  # still rolling
    assert c.update(0.0, 0.0, STOPPED) is True  # uDV braked it to rest
    assert c.finished

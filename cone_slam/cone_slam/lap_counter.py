"""Pure, ROS-free lap/finish detector for `/slam/finished` (#384).

The SLAM frame origin is the car spawn (≈ the start/finish line), so the
distance from origin traces each lap: it climbs as the car drives out and
falls back toward 0 as it returns. A lap is counted the moment the pose
returns within ``close_radius_m`` of origin AFTER having left by at least
``arm_radius_m`` (re-arm hysteresis, so start-line jitter or a slow crawl
across the line can't double-count).

Finish is reported ONLY when the per-mission target is met AND the car has
come to a standstill (``speed <= standstill_mps``). This gate is safety
critical: entering AS Finished fires the EBS and opens the SDC on the
firmware (`as_actuation`), so signalling `/slam/finished` while the car is
still moving would trigger a hard EBS stop at speed. FS rules also define
AS Finished as a stationary state.

Two mission shapes:
  * lap-based (``laps_to_finish > 0``): autocross = 1, trackdrive = 10.
    NB trackdrive additionally needs control_node to hold its stop-anchor
    to the final lap before it actually drives N laps (follow-up); this
    counter is forward-compatible — it just won't reach the target until
    then.
  * distance-based (``finish_distance_m > 0``): straight missions that
    never return to origin, e.g. acceleration (~75 m).

A mission with neither configured (e.g. skidpad's figure-8) never
auto-finishes.

The detector is deliberately dependency-free (no rclpy / numpy) so it is
unit-testable off-node, mirroring the firmware's pure `as_transition` /
`as_actuation` cores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LapCounterConfig:
    """Completion criteria for one mission run.

    laps_to_finish:   finish after N return-to-origin crossings (0 = off).
    finish_distance_m: finish when the pose is this far from origin, for
                       straight missions that never return (0 = off).
    arm_radius_m:      must leave origin by at least this before a return
                       counts as a lap (re-arm hysteresis).
    close_radius_m:    a return within this of origin closes a lap.
    standstill_mps:    speed at/below which the car counts as stopped.
    """

    laps_to_finish: int = 0
    finish_distance_m: float = 0.0
    arm_radius_m: float = 15.0
    close_radius_m: float = 6.0
    standstill_mps: float = 0.5


class LapCounter:
    """Stateful lap/distance completion detector. Feed it one pose+speed
    per scan via :meth:`update`; it returns True on the single rising edge
    where the finish condition first holds."""

    def __init__(self, config: LapCounterConfig | None = None) -> None:
        self.cfg = config if config is not None else LapCounterConfig()
        self.reset()

    def reset(self) -> None:
        self.lap_count = 0
        self._armed = False
        self.finished = False

    @property
    def enabled(self) -> bool:
        return self.cfg.laps_to_finish > 0 or self.cfg.finish_distance_m > 0.0

    @property
    def final_lap(self) -> bool:
        """True once the NEXT finish-line crossing is the closing one.

        This is what control_node gates its stop-anchor on. It needs to know it
        is on the closing lap BEFORE it reaches the gate, so it can brake AT
        the line instead of a lap early — which is why this cannot be derived
        from `finished`: `finished` requires standstill, and the car only
        reaches standstill BECAUSE control braked at the anchor. Gating the
        anchor on `finished` would deadlock (never brake → never stop → never
        finish). Hence a separate, earlier signal.

        Missions with no lap criterion (accel's distance mode, skidpad) report
        True: they have no lap to gate on, so the stop anchor keeps its
        historical behaviour and `stop_latch_min_travel` remains the only gate.
        Autocross (1 lap) is True from the start for the same reason — its
        first gate past the travel threshold IS the finish.

        Trackdrive (10 laps) is the only mission this actually changes: False
        for laps 0..8, True from lap 9, so the anchor arms for the 10th and
        final crossing.
        """
        if self.cfg.laps_to_finish <= 0:
            return True
        return self.lap_count >= self.cfg.laps_to_finish - 1

    def update(self, pose_x: float, pose_y: float, speed_mps: float) -> bool:
        """Feed one pose sample (SLAM-absolute, metres) plus the current
        planar speed (m/s). Returns True exactly once — on the scan where
        the finish condition (target reached AND stopped) first holds."""
        if self.finished or not self.enabled:
            return False

        dist = math.hypot(pose_x, pose_y)

        # Lap counting with re-arm hysteresis (only for lap-based missions).
        if self.cfg.laps_to_finish > 0:
            if dist >= self.cfg.arm_radius_m:
                self._armed = True
            elif self._armed and dist <= self.cfg.close_radius_m:
                self.lap_count += 1
                self._armed = False

        lap_done = (
            self.cfg.laps_to_finish > 0
            and self.lap_count >= self.cfg.laps_to_finish
        )
        dist_done = (
            self.cfg.finish_distance_m > 0.0
            and dist >= self.cfg.finish_distance_m
        )
        stopped = speed_mps <= self.cfg.standstill_mps

        if (lap_done or dist_done) and stopped:
            self.finished = True
            return True
        return False

    def summary(self) -> str:
        """One-line human description of why/how it finished (for logs)."""
        if self.cfg.laps_to_finish > 0:
            return f"{self.lap_count}/{self.cfg.laps_to_finish} laps"
        if self.cfg.finish_distance_m > 0.0:
            return f"distance >= {self.cfg.finish_distance_m:.0f} m"
        return "no completion criterion"

"""Deterministic skidpad driver — pure, ROS-free.

The driver is the per-tick brain of the deterministic skidpad: given the current
EKF pose and speed it advances arc-length progress along the fixed reference
(`skidpad_reference`) and hands back the forward path window the controller
should track, plus whether the mission is finished. It holds NO rclpy — the
`skidpad_planner_runtime` node wrapper feeds it `/odom` and publishes its output,
so every decision here is unit-testable off-node (mirrors
`cone_slam.lap_counter`).

Why position-delta, not speed·dt, for progress
-----------------------------------------------
Progress is advanced by the Euclidean step between consecutive poses, not by
integrating speed. Odometry position is what we trust (EKF, low drift); a noisy
speed estimate integrated over time would accumulate error. Speed is used only
for the standstill gate on finish.

Stopping is deliberately dumb
-----------------------------
There is no perception and no speed side-channel to the controller, so the
driver cannot ask control to brake early. It publishes the forward window until
the car reaches the end of the reference, then returns an EMPTY window — which
control fail-safes to a zero command, so the car coasts (the real car has no
service brake anyway). Once the car is both at the end AND stopped, `finished`
latches true; the node turns that into `/slam/finished`, and the uDV runs the
normal AS Finished actuation. See skidpad-deterministic-design memory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from skidpad.skidpad_reference import (
    PathPoint,
    ReferencePath,
    SkidpadGeometry,
    SkidpadProgress,
    build_reference,
)


@dataclass(frozen=True)
class DriverOutput:
    """One tick's result. `path` is the forward window for the controller
    (empty once the car has run off the end of the reference)."""

    path: list[PathPoint]
    finished: bool
    progress_s: float
    remaining_m: float


class SkidpadDriver:
    """Tracks arc-length progress along a fixed skidpad reference and emits the
    forward window + finish flag. One instance per run; call `reset()` on
    (re)activation."""

    def __init__(
        self,
        reference: ReferencePath | None = None,
        *,
        geometry: SkidpadGeometry | None = None,
        window_len_m: float = 10.0,
        standstill_mps: float = 0.5,
        finish_margin_m: float = 0.5,
    ) -> None:
        if reference is None:
            reference = build_reference(geometry or SkidpadGeometry())
        if window_len_m <= 0.0:
            raise ValueError("window_len_m must be > 0")
        self.reference = reference
        self.window_len_m = window_len_m
        self.standstill_mps = standstill_mps
        self.finish_margin_m = finish_margin_m
        self._progress = SkidpadProgress(reference)
        self._last_xy: tuple[float, float] | None = None
        self._finished = False

    def reset(self) -> None:
        """Start a fresh run: progress back to the spawn, finish latch cleared."""
        self._progress = SkidpadProgress(self.reference)
        self._last_xy = None
        self._finished = False

    @property
    def progress_s(self) -> float:
        return self._progress.s

    def step(self, pose_x: float, pose_y: float, speed_mps: float) -> DriverOutput:
        """Advance one tick from the current pose + speed. Returns the forward
        window to track and whether the run is finished."""
        # Distance travelled since the last pose — Euclidean, not integrated
        # speed. The very first tick has no reference, so it advances 0 and just
        # anchors progress via the forward pose refinement.
        if self._last_xy is None:
            dist = 0.0
        else:
            dist = math.hypot(pose_x - self._last_xy[0], pose_y - self._last_xy[1])
        self._last_xy = (pose_x, pose_y)

        s = self._progress.update(dist, pose_x, pose_y)
        remaining = self._progress.remaining()

        # Finish latches once: the crossing self-intersection means a
        # late-arriving stale pose must never un-finish a completed run.
        if not self._finished and self._progress.is_finished(
            speed_mps,
            standstill_mps=self.standstill_mps,
            finish_margin_m=self.finish_margin_m,
        ):
            self._finished = True

        # Empty the window once within finish_margin of the end (same threshold
        # that gates `finished`): control fail-safes a zero command, so the car
        # coasts the last stretch to a stop rather than tracking a vanishing
        # sliver of path. The real car has no service brake, so this is a coast,
        # not a brake — the AS Finished actuation clamps it once stopped.
        path = [] if remaining <= self.finish_margin_m else self._window(s)
        return DriverOutput(
            path=path,
            finished=self._finished,
            progress_s=s,
            remaining_m=remaining,
        )

    def _window(self, s: float) -> list[PathPoint]:
        """Forward slice for the controller. A <2-point window is degenerate
        (control needs a segment), so collapse it to empty → zero command."""
        w = self.reference.window(s, self.window_len_m)
        if len(w) < 2:
            return []
        return w

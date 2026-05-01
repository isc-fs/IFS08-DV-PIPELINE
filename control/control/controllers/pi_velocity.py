"""PI velocity tracker (skeleton — returns zeros).

Real implementation lands in the next commit. Wraps the speed-setpoint pre-pass
and the throttle/regen split with deadband.
"""
from __future__ import annotations
from typing import Tuple

from control.controllers.base import LongitudinalController
from control.state import VehicleState
from control.reference import ReferenceTrajectory


class PIVelocity(LongitudinalController):
    def __init__(self,
                 v_max: float = 12.0,
                 a_lat_max: float = 6.0,
                 a_dec_max: float = 4.0,
                 kp: float = 0.5,
                 ki: float = 0.05,
                 deadband: float = 0.2):
        self.v_max = v_max
        self.a_lat_max = a_lat_max
        self.a_dec_max = a_dec_max
        self.kp = kp
        self.ki = ki
        self.deadband = deadband
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> Tuple[float, float]:
        # TODO: implement
        return 0.0, 0.0

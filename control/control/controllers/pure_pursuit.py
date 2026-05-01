"""Pure Pursuit lateral controller (skeleton — returns 0).

Real implementation lands in the next commit. This stub exists so the ROS node
compiles and we can verify TF + subscriptions + publish path end-to-end before
adding algorithm code.
"""
from __future__ import annotations
from control.controllers.base import LateralController
from control.state import VehicleState
from control.reference import ReferenceTrajectory


class PurePursuit(LateralController):
    def __init__(self, lookahead_min: float = 1.5, lookahead_k: float = 0.5):
        self.lookahead_min = lookahead_min
        self.lookahead_k = lookahead_k

    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> float:
        # TODO: implement
        return 0.0

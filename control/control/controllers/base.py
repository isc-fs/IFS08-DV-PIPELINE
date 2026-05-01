"""Strategy contracts for lateral and longitudinal controllers.

Every controller (Pure Pursuit, Stanley, LQR, MPC, ...) implements one of these
ABCs. The ROS node picks one of each via parameters and never knows what's
inside — `compute()` is the entire interface.

Inputs are the immutable per-tick `VehicleState` and the `ReferenceTrajectory`
shared across both controllers. Outputs are normalized actuator demands in the
ranges the bridge sends as-is to setVehicleCommand.

Adding a new algorithm:
  1. New file under controllers/ implementing the ABC
  2. Register in control_node._make_lateral / _make_longitudinal
  3. Set the ROS param at launch time

LQR/MPC will fit by holding internal state (linearization point, prediction
horizon, solver) but exposing the same compute() signature.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

from control.state import VehicleState
from control.reference import ReferenceTrajectory


class LateralController(ABC):
    @abstractmethod
    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> float:
        """Return normalized steering in [-1, +1]; positive = left."""

    def reset(self) -> None:
        """Clear any internal state (called on new event start / EBS reset)."""


class LongitudinalController(ABC):
    @abstractmethod
    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> Tuple[float, float]:
        """Return (throttle, regen), each in [0, 1]. At most one is non-zero
        per tick (deadband around setpoint enforces single-channel command)."""

    def reset(self) -> None:
        """Clear integrator etc."""

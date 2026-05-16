"""Strategy contracts for vehicle control.

Decoupled lateral / longitudinal (Pure Pursuit + PI) implement
:class:`LateralController` and :class:`LongitudinalController`, composed by
:class:`control.controllers.composite_drive.CompositeDriveController`.

A *joint* controller (LQR, nonlinear MPC, …) should instead subclass
:class:`DriveController` and return a full :class:`~control.controllers.actuation.ActuationCommand`
in one :meth:`DriveController.compute` call.

The ROS node wires exactly one :class:`DriveController` per run (``drive_controller``
param, default ``composite``).

Adding a decoupled algorithm:
  1. New file under ``controllers/`` implementing the lateral or longitudinal ABC
  2. Register in ``control_node._build_strategies``
  3. Set the ROS param or mode_manager behavior at launch

Adding a joint algorithm:
  1. Subclass :class:`DriveController`, implement :meth:`compute` → :class:`ActuationCommand`
  2. Register a new ``drive_controller`` value in ``control_node._build_strategies``
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

from control.controllers.actuation import ActuationCommand
from control.state import VehicleState
from control.reference import ReferenceTrajectory


class DriveController(ABC):
    """Single strategy object for all motion axes (steer, throttle, regen)."""

    @abstractmethod
    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> ActuationCommand:
        """Full actuation for this tick."""

    def reset(self) -> None:
        """Clear integrators / warm-start cache when a run starts or resets."""
        pass


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

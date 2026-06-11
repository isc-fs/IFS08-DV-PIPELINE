"""Decoupled lateral + longitudinal behind :class:`DriveController`."""
from __future__ import annotations

from control.controllers.actuation import ActuationCommand
from control.controllers.base import (
    DriveController,
    LateralController,
    LongitudinalController,
)
from control.reference import ReferenceTrajectory
from control.state import VehicleState


class CompositeDriveController(DriveController):
    """Pure Pursuit + PI velocity (or any Lateral + Longitudinal pair)."""

    def __init__(
        self,
        lateral: LateralController,
        longitudinal: LongitudinalController,
    ) -> None:
        self._lateral = lateral
        self._longitudinal = longitudinal

    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> ActuationCommand:
        steer = self._lateral.compute(state, ref)
        thr, regen = self._longitudinal.compute(state, ref)
        return ActuationCommand(
            steering_normalized=steer,
            throttle=float(thr),
            regen=float(regen),
        )

    def reset(self) -> None:
        self._lateral.reset()
        self._longitudinal.reset()

    @property
    def last_v_set(self) -> float:
        return float(getattr(self._longitudinal, "last_v_set", 0.0))

    @property
    def last_kappa_max(self) -> float:
        return float(getattr(self._longitudinal, "last_kappa_max", 0.0))

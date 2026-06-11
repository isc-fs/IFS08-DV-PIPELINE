"""IMU sample buffering + GTSAM preintegration helper.

Wraps gtsam.PreintegratedImuMeasurements with an init-time bias estimator.
The car must be stationary during INIT so we can read off the gravity
vector and the gyro/accel bias means before any motion gets folded into
the graph.

Per the BMI088 datasheet (and our settings.json sim model):
  accel noise std ≈ 0.024 m/s²    (175 µg/√Hz × √(200 Hz BW))
  gyro  noise std ≈ 0.0035 rad/s  (0.014 °/s/√Hz × √(200 Hz BW))

These get squared into per-axis covariances and passed to GTSAM.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import gtsam


# BMI088 noise floor at 400 Hz sample rate (200 Hz Nyquist BW).
# Squared into variances for GTSAM's covariance matrices.
ACCEL_NOISE_STD = 0.024  # m/s²
GYRO_NOISE_STD = 0.0035  # rad/s

# Bias random-walk noise — small, but non-zero so the optimizer can
# evolve the bias estimate over time. Tightened 10× on 2026-04-28
# (see BIAS_RW_SIGMAS in factor_graph.py for the full rationale and
# the actual values used at factor-construction time).
ACCEL_BIAS_RW_STD = 1e-4   # m/s²/√s (so variance per dt is this² × dt)
GYRO_BIAS_RW_STD = 1e-5    # rad/s/√s

# Tiny integration noise — accounts for discretization error in the
# trapezoidal IMU integration. GTSAM default is 1e-8 m²/s³.
INTEGRATION_NOISE_VAR = 1e-8


@dataclass
class ImuSample:
    """One raw IMU sample, body-frame, ROS REP-103 right-handed."""
    t: float                # absolute time in seconds
    accel: np.ndarray       # (3,) m/s²
    gyro: np.ndarray        # (3,) rad/s


class ImuPreintegrator:
    """Buffers IMU samples between pose nodes and produces GTSAM
    preintegrated factors on demand.

    Lifecycle:
      1. push_sample()       — every IMU callback during all states
      2. estimate_bias()     — once at end of INIT_CALIBRATING
      3. integrate_to(t)     — at each pose-add trigger; returns the
                               PreintegratedImuMeasurements covering all
                               buffered samples up to time t. Resets
                               internal accumulator.
    """

    def __init__(self) -> None:
        self._buffer: List[ImuSample] = []

        # Built lazily on first integrate_to() call after estimate_bias().
        self._params: Optional[gtsam.PreintegrationParams] = None
        self._bias: Optional[gtsam.imuBias.ConstantBias] = None
        self._pim: Optional[gtsam.PreintegratedImuMeasurements] = None

        # Set after estimate_bias().
        self._initialized: bool = False
        self._last_integration_t: Optional[float] = None

        # Reentrant lock so push_sample (IMU thread) and integrate_to /
        # estimate_bias (cone thread, via MultiThreadedExecutor) don't
        # race on _buffer or _pim. Push is sub-microsecond; integrate's
        # buffer walk is sub-millisecond; iSAM2 happens AFTER
        # integrate_to returns and outside the lock — so contention is
        # negligible.
        self._lock = threading.RLock()

    # ----- raw sample ingest -------------------------------------------------

    def push_sample(self, sample: ImuSample) -> None:
        """Append a raw IMU sample to the buffer.

        Called from the /imu callback at IMU rate (~400 Hz). Cheap.
        Lock-protected so a concurrent integrate_to() in the cone
        thread cannot read mid-mutation.
        """
        with self._lock:
            self._buffer.append(sample)

    # ----- INIT_CALIBRATING --------------------------------------------------

    def has_enough_for_calibration(self, min_seconds: float = 3.0) -> bool:
        """Return True if the buffered samples span at least min_seconds.

        Used by the node to decide when to transition out of
        INIT_CALIBRATING.
        """
        with self._lock:
            if len(self._buffer) < 2:
                return False
            span = self._buffer[-1].t - self._buffer[0].t
            return span >= min_seconds

    def estimate_bias(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute biases from the calibration window.

        Assumes the car was STATIONARY during the buffered window — so
        the mean accel reading is gravity (in body frame) and the mean
        gyro reading is the gyro bias.

        Returns:
            (accel_bias, gyro_bias, gravity_body_frame)

        Sets up self._params / self._bias / self._pim ready for
        subsequent integrate_to() calls.
        """
        with self._lock:
            return self._estimate_bias_unlocked()

    def _estimate_bias_unlocked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._buffer:
            raise RuntimeError("No IMU samples buffered for calibration")

        # Stack buffered samples into Nx3 arrays.
        accels = np.array([s.accel for s in self._buffer])
        gyros = np.array([s.gyro for s in self._buffer])

        accel_mean = accels.mean(axis=0)
        gyro_mean = gyros.mean(axis=0)

        # The mean accel is the IMU's specific-force reading at rest.
        # An accelerometer in a level body frame at rest reads
        # **+9.81 m/s² in Z**, not -9.81: the sensor measures the
        # normal force from the floor pushing up against gravity
        # (specific force = a_inertial − g, with g pointing -Z, so the
        # reading at rest = +g along the body Z axis).
        #
        # Bias = mean − expected-at-rest = mean − (0, 0, +9.81).
        # GTSAM's PreintegrationParams.MakeSharedU(9.81) defines gravity
        # as (0, 0, -9.81) in the nav frame, consistent with this.
        gravity_body = accel_mean.copy()  # raw mean reading in body frame
        accel_bias = accel_mean - np.array([0.0, 0.0, 9.81])
        gyro_bias = gyro_mean.copy()

        # Build the preintegration params now that we know gravity.
        # GTSAM convention: gravity is given as the value the IMU reads
        # when stationary, in the WORLD/navigation frame. For a level
        # car, this is (0, 0, -9.81) — and the IMU's accel reading
        # (after bias correction) integrates against this in the world
        # frame. MakeSharedU sets gravity to (0, 0, -|g|).
        self._params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        self._params.setAccelerometerCovariance(
            np.eye(3) * (ACCEL_NOISE_STD ** 2)
        )
        self._params.setGyroscopeCovariance(
            np.eye(3) * (GYRO_NOISE_STD ** 2)
        )
        self._params.setIntegrationCovariance(
            np.eye(3) * INTEGRATION_NOISE_VAR
        )

        self._bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)
        self._pim = gtsam.PreintegratedImuMeasurements(
            self._params, self._bias
        )

        self._initialized = True
        # Mark the calibration end time as the start of integration —
        # subsequent samples already in the buffer past this point will
        # be folded into the first pose's IMU factor.
        self._last_integration_t = self._buffer[-1].t

        # Drop samples older than the integration start; we don't need them.
        self._buffer = [s for s in self._buffer if s.t > self._last_integration_t]

        return accel_bias, gyro_bias, gravity_body

    # ----- SLAM_RUNNING ------------------------------------------------------

    def integrate_to(
        self, t_end: float
    ) -> tuple[gtsam.PreintegratedImuMeasurements, float]:
        """Integrate all buffered samples up to t_end into a PIM.

        Returns the PIM (caller passes to gtsam.ImuFactor) and the dt
        covered. After return, the internal PIM is reset and buffer is
        trimmed so the next integrate_to() starts fresh.
        """
        with self._lock:
            return self._integrate_to_unlocked(t_end)

    def _integrate_to_unlocked(
        self, t_end: float
    ) -> tuple[gtsam.PreintegratedImuMeasurements, float]:
        if not self._initialized or self._pim is None:
            raise RuntimeError(
                "integrate_to() called before estimate_bias() "
                "— preintegrator not initialized"
            )

        # Reset accumulator before integrating this window.
        self._pim.resetIntegration()

        # Walk buffered samples that fall in (last_integration_t, t_end].
        # Use trapezoidal integration with the dt to the next sample as
        # the step. Samples beyond t_end stay in the buffer for the
        # next call.
        kept: List[ImuSample] = []
        consumed = 0
        for i, s in enumerate(self._buffer):
            if s.t > t_end:
                kept = self._buffer[i:]
                break
            # dt to the next sample (or to t_end for the final one).
            if i + 1 < len(self._buffer) and self._buffer[i + 1].t <= t_end:
                dt = self._buffer[i + 1].t - s.t
            else:
                dt = t_end - s.t
            if dt > 0:
                self._pim.integrateMeasurement(s.accel, s.gyro, dt)
                consumed += 1
        else:
            kept = []

        if consumed == 0:
            buf_first = self._buffer[0].t if self._buffer else None
            buf_last  = self._buffer[-1].t if self._buffer else None
            raise RuntimeError(
                f"No IMU samples to integrate up to t_end={t_end:.6f} "
                f"(last_integration_t={self._last_integration_t:.6f}, "
                f"Δ_to_t_end={t_end - (self._last_integration_t or 0):+.6f}s, "
                f"buffer={len(self._buffer)} samples"
                + (f" [{buf_first:.6f} → {buf_last:.6f}, "
                   f"first-t_end={buf_first - t_end:+.6f}s]" if buf_first is not None else "")
                + ")"
            )

        dt_total = t_end - self._last_integration_t
        self._last_integration_t = t_end
        self._buffer = kept

        return self._pim, dt_total

    # ----- accessors ---------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def current_bias(self) -> Optional[gtsam.imuBias.ConstantBias]:
        return self._bias

    @property
    def params(self) -> Optional[gtsam.PreintegrationParams]:
        return self._params

    def update_bias(self, new_bias: gtsam.imuBias.ConstantBias) -> None:
        """Replace the working bias (called after each iSAM2 update so
        the next preintegration uses the most recent estimate)."""
        with self._lock:
            if not self._initialized or self._params is None:
                raise RuntimeError("update_bias() before estimate_bias()")
            self._bias = new_bias
            self._pim = gtsam.PreintegratedImuMeasurements(self._params, new_bias)

"""Quintic motion laws with zero endpoint velocity and acceleration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial


def _build_quintic_polynomials() -> tuple[Polynomial, Polynomial, Polynomial]:
    """Return the normalized quintic profile and its first two algorithmic derivatives."""

    profile = Polynomial([0.0, 0.0, 0.0, 10.0, -15.0, 6.0])
    velocity = profile.deriv(1)
    acceleration = profile.deriv(2)
    return profile, velocity, acceleration


_PROFILE_EXPR, _PROFILE_D1_EXPR, _PROFILE_D2_EXPR = _build_quintic_polynomials()
_PROFILE = _PROFILE_EXPR
_PROFILE_D1 = _PROFILE_D1_EXPR
_PROFILE_D2 = _PROFILE_D2_EXPR


@dataclass(frozen=True)
class QuinticBoundaryTrajectory:
    """Time-scaled quintic trajectory with null endpoint velocity and acceleration.

    This class implements the normalized motion law
    :math:`q(x) = 6x^5 - 15x^4 + 10x^3`, where
    :math:`x = (t - t_0) / (t_1 - t_0)`.
    """

    t0: float
    t1: float
    q0: float
    q1: float

    def __post_init__(self) -> None:
        """Validate the time interval."""

        if self.t1 <= self.t0:
            raise ValueError("t1 must be strictly greater than t0.")

    @property
    def duration(self) -> float:
        """Return the trajectory duration."""

        return self.t1 - self.t0

    def phase(self, time: float | np.ndarray) -> np.ndarray:
        """Return the normalized phase clipped to the [0, 1] interval."""

        raw_phase = (np.asarray(time, dtype=float) - self.t0) / self.duration
        return np.clip(raw_phase, 0.0, 1.0)

    def position(self, time: float | np.ndarray) -> np.ndarray:
        """Return the trajectory position at the requested time."""

        phase = self.phase(time)
        return self.q0 + (self.q1 - self.q0) * _PROFILE(phase)

    def velocity(self, time: float | np.ndarray) -> np.ndarray:
        """Return the time derivative of the trajectory position."""

        phase = self.phase(time)
        active = (np.asarray(time, dtype=float) >= self.t0) & (
            np.asarray(time, dtype=float) <= self.t1
        )
        return np.where(active, (self.q1 - self.q0) * _PROFILE_D1(phase) / self.duration, 0.0)

    def acceleration(self, time: float | np.ndarray) -> np.ndarray:
        """Return the second time derivative of the trajectory position."""

        phase = self.phase(time)
        active = (np.asarray(time, dtype=float) >= self.t0) & (
            np.asarray(time, dtype=float) <= self.t1
        )
        scale = (self.q1 - self.q0) / (self.duration**2)
        return np.where(active, scale * _PROFILE_D2(phase), 0.0)

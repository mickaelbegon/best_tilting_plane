"""Tests for the quintic Yeadon-style trajectory generator."""

import numpy as np
import pytest

from best_tilting_plane.trajectories import QuinticBoundaryTrajectory


def test_quintic_boundary_conditions_are_respected() -> None:
    """The trajectory should satisfy the endpoint position, velocity, and acceleration conditions."""

    trajectory = QuinticBoundaryTrajectory(t0=0.0, t1=0.3, q0=np.pi, q1=0.0)

    assert trajectory.position(0.0) == pytest.approx(np.pi)
    assert trajectory.position(0.3) == pytest.approx(0.0)
    assert trajectory.velocity(0.0) == pytest.approx(0.0)
    assert trajectory.velocity(0.3) == pytest.approx(0.0)
    assert trajectory.acceleration(0.0) == pytest.approx(0.0)
    assert trajectory.acceleration(0.3) == pytest.approx(0.0)


def test_quintic_trajectory_is_constant_outside_its_active_interval() -> None:
    """The trajectory should clamp position and null derivatives outside the motion window."""

    trajectory = QuinticBoundaryTrajectory(t0=0.1, t1=0.4, q0=1.2, q1=-0.3)

    assert trajectory.position(-1.0) == pytest.approx(1.2)
    assert trajectory.position(10.0) == pytest.approx(-0.3)
    assert trajectory.velocity(-1.0) == pytest.approx(0.0)
    assert trajectory.velocity(10.0) == pytest.approx(0.0)
    assert trajectory.acceleration(-1.0) == pytest.approx(0.0)
    assert trajectory.acceleration(10.0) == pytest.approx(0.0)


def test_quintic_trajectory_vectorized_evaluation_matches_scalar_calls() -> None:
    """Vectorized evaluation should preserve the scalar results."""

    trajectory = QuinticBoundaryTrajectory(t0=0.0, t1=0.3, q0=np.deg2rad(180.0), q1=0.0)
    samples = np.linspace(-0.1, 0.5, 9)

    expected_positions = np.array([trajectory.position(sample) for sample in samples])
    expected_velocities = np.array([trajectory.velocity(sample) for sample in samples])
    expected_accelerations = np.array([trajectory.acceleration(sample) for sample in samples])

    np.testing.assert_allclose(trajectory.position(samples), expected_positions)
    np.testing.assert_allclose(trajectory.velocity(samples), expected_velocities)
    np.testing.assert_allclose(trajectory.acceleration(samples), expected_accelerations)


def test_quintic_trajectory_rejects_invalid_time_bounds() -> None:
    """The trajectory should reject zero or negative durations."""

    with pytest.raises(ValueError, match="strictly greater"):
        QuinticBoundaryTrajectory(t0=0.2, t1=0.2, q0=0.0, q1=1.0)

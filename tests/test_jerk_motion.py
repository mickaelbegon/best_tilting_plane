"""Tests for the piecewise-constant-jerk arm-motion helpers."""

from __future__ import annotations

import matplotlib
import numpy as np

from best_tilting_plane.simulation import (
    PiecewiseConstantJerkTrajectory,
    TwistOptimizationVariables,
    create_first_arm_piecewise_constant_comparison_figure,
    first_arm_piecewise_constant_comparison_data,
)

matplotlib.use("Agg")


def test_piecewise_constant_jerk_trajectory_matches_constant_jerk_kinematics() -> None:
    """A single active jerk interval should reproduce the analytical polynomial state update."""

    trajectory = PiecewiseConstantJerkTrajectory(
        q0=1.0,
        qdot0=2.0,
        qddot0=3.0,
        step=0.1,
        jerks=np.array([4.0], dtype=float),
        active_start=0.0,
        active_end=0.1,
    )

    q, qdot, qddot = trajectory.state(0.1)

    np.testing.assert_allclose(
        [q, qdot, qddot],
        [
            1.0 + 0.1 * 2.0 + 0.5 * 0.1**2 * 3.0 + (0.1**3) * 4.0 / 6.0,
            2.0 + 0.1 * 3.0 + 0.5 * 0.1**2 * 4.0,
            3.0 + 0.1 * 4.0,
        ],
    )


def test_piecewise_constant_jerk_trajectory_keeps_zero_jerk_outside_active_window() -> None:
    """Before the active window starts, the trajectory should only integrate the current acceleration."""

    trajectory = PiecewiseConstantJerkTrajectory(
        q0=0.0,
        qdot0=1.0,
        qddot0=2.0,
        step=0.2,
        jerks=np.array([10.0, 10.0], dtype=float),
        active_start=0.2,
        active_end=0.4,
    )

    q, qdot, qddot = trajectory.state(0.1)

    np.testing.assert_allclose(
        [q, qdot, qddot],
        [0.1 * 1.0 + 0.5 * 0.1**2 * 2.0, 1.0 + 0.1 * 2.0, 2.0],
    )


def test_first_arm_piecewise_constant_comparison_helpers_return_consistent_shapes() -> None:
    """The comparison helpers should expose aligned time histories and figure axes."""

    variables = TwistOptimizationVariables(
        right_arm_start=0.1,
        left_plane_initial=0.0,
        left_plane_final=0.2,
        right_plane_initial=0.0,
        right_plane_final=0.0,
    )

    data = first_arm_piecewise_constant_comparison_data(
        variables,
        total_time=1.0,
        jerk_step=0.02,
        sample_step=0.01,
    )

    assert data["time"].shape == data["reference_q"].shape
    assert data["time"].shape == data["approximate_q"].shape
    assert data["time"].shape == data["reference_qdot"].shape
    assert data["time"].shape == data["approximate_qdot"].shape
    assert data["time"].shape == data["reference_qddot"].shape
    assert data["time"].shape == data["approximate_qddot"].shape
    assert data["jerk_nodes"].shape == data["jerk_time"].shape
    assert np.all(np.isfinite(data["approximate_q"]))
    assert np.all(np.isfinite(data["approximate_qdot"]))
    assert np.all(np.isfinite(data["approximate_qddot"]))

    figure, axes, figure_data = create_first_arm_piecewise_constant_comparison_figure(
        variables,
        total_time=1.0,
        jerk_step=0.02,
        sample_step=0.01,
    )

    assert len(axes) == 3
    assert figure.axes[:3] == list(axes)
    assert np.array_equal(figure_data["time"], data["time"])

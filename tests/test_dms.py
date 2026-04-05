"""Tests for the direct multiple-shooting optimizer."""

from __future__ import annotations

from pathlib import Path

import casadi as ca
import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.optimization import DirectMultipleShootingOptimizer
from best_tilting_plane.optimization import dms as dms_module
from best_tilting_plane.simulation import (
    PiecewiseConstantJerkArmMotion,
    PiecewiseConstantJerkTrajectory,
    SimulationConfiguration,
    TwistOptimizationVariables,
    approximate_first_arm_elevation_motion,
)


def test_direct_multiple_shooting_initial_guess_motion_respects_activation_windows(
    tmp_path: Path,
) -> None:
    """The warm start should activate the first arm immediately and the second arm at `t1`."""

    optimizer = DirectMultipleShootingOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=1.0, steps=201, integrator="rk4", rk4_step=0.005),
        shooting_step=0.02,
    )

    motion = optimizer._initial_guess_motion(
        TwistOptimizationVariables(
            right_arm_start=0.16,
            left_plane_initial=-0.2,
            left_plane_final=0.1,
            right_plane_initial=0.3,
            right_plane_final=-0.1,
        )
    )

    assert motion.left_plane.active_start == 0.0
    assert motion.left_plane.active_end == dms_module.LEFT_ARM_ACTIVE_DURATION
    assert motion.right_plane.active_start == 0.0
    assert motion.right_plane.active_end == dms_module.RIGHT_ARM_ACTIVE_DURATION
    assert motion.left_plane.jerks.shape == (optimizer.active_control_count,)
    assert motion.right_plane.jerks.shape == (optimizer.active_control_count,)
    assert motion.left_plane.duration == optimizer.configuration.final_time
    assert motion.right_plane.duration == optimizer.configuration.final_time - 0.16


def test_direct_multiple_shooting_jerk_bound_matches_left_elevation_fitting(
    tmp_path: Path,
) -> None:
    """The jerk bounds should come from the maximum absolute jerk of the left-arm elevation fitting."""

    configuration = SimulationConfiguration(final_time=1.0, steps=201, integrator="rk4", rk4_step=0.005)
    optimizer = DirectMultipleShootingOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=configuration,
        shooting_step=0.02,
    )

    reference = approximate_first_arm_elevation_motion(
        total_time=dms_module.LEFT_ARM_ACTIVE_DURATION,
        step=optimizer.shooting_step,
    )

    assert optimizer.jerk_bound == np.max(np.abs(reference.jerks))


def test_direct_multiple_shooting_solve_fixed_start_builds_float_bounds_and_returns_motion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """One fixed-start DMS solve should assemble a float-bounded NLP and rebuild the jerk-controlled motion."""

    optimizer = DirectMultipleShootingOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=0.4, steps=81, integrator="rk4", rk4_step=0.005),
        shooting_step=0.02,
    )
    initial_guess = TwistOptimizationVariables(
        right_arm_start=0.1,
        left_plane_initial=0.0,
        left_plane_final=0.0,
        right_plane_initial=0.0,
        right_plane_final=0.0,
    )
    warm_start_motion = PiecewiseConstantJerkArmMotion(
        left_plane=PiecewiseConstantJerkTrajectory(
            q0=0.0,
            qdot0=0.0,
            qddot0=0.0,
            step=optimizer.shooting_step,
            jerks=np.zeros(optimizer.active_control_count, dtype=float),
            active_start=0.0,
            active_end=dms_module.LEFT_ARM_ACTIVE_DURATION,
            total_duration=optimizer.configuration.final_time,
        ),
        right_plane=PiecewiseConstantJerkTrajectory(
            q0=0.0,
            qdot0=0.0,
            qddot0=0.0,
            step=optimizer.shooting_step,
            jerks=np.zeros(optimizer.active_control_count, dtype=float),
            active_start=0.0,
            active_end=dms_module.RIGHT_ARM_ACTIVE_DURATION,
            total_duration=optimizer.configuration.final_time - initial_guess.right_arm_start,
        ),
        right_arm_start=initial_guess.right_arm_start,
    )
    initial_state = np.asarray(optimizer._symbolic_initial_root_state(initial_guess), dtype=float).reshape(-1)
    state_history = np.tile(initial_state.reshape(-1, 1), (1, optimizer.interval_count + 1))
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        optimizer,
        "_initial_guess_motion",
        lambda _variables, **_kwargs: warm_start_motion,
    )
    monkeypatch.setattr(
        optimizer,
        "_initial_guess_root_state_history",
        lambda _variables, _motion: state_history,
    )

    class FakeSimulator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def simulate():
            return type(
                "SimulationResult",
                (),
                {
                    "final_twist_angle": -0.5,
                    "final_twist_turns": -0.08,
                },
            )()

    class FakeSolver:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"x": ca.DM(np.asarray(kwargs["x0"], dtype=float)), "f": ca.DM([[-0.5]])}

        @staticmethod
        def stats():
            return {"return_status": "Solve_Succeeded"}

    def fake_nlpsol(_name, _solver_name, _problem, options):
        captured["options"] = options
        return FakeSolver()

    monkeypatch.setattr(dms_module, "PredictiveAerialTwistSimulator", FakeSimulator)
    monkeypatch.setattr(dms_module.ca, "nlpsol", fake_nlpsol)

    result = optimizer.solve_fixed_start(
        initial_guess,
        right_arm_start=initial_guess.right_arm_start,
        max_iter=3,
        print_level=5,
        print_time=True,
    )

    assert result.success
    assert result.solver_status == "Solve_Succeeded"
    assert result.variables == initial_guess
    assert result.arm_node_times.shape == (optimizer.active_control_count + 1,)
    assert result.root_state_nodes.shape == (dms_module.ROOT_STATE_SIZE, optimizer.interval_count + 1)
    assert result.left_plane_jerk.shape == (optimizer.active_control_count,)
    assert result.right_plane_jerk.shape == (optimizer.active_control_count,)
    assert result.left_plane_state_nodes.shape == (dms_module.PLANE_STATE_SIZE, optimizer.active_control_count + 1)
    assert result.right_plane_state_nodes.shape == (dms_module.PLANE_STATE_SIZE, optimizer.active_control_count + 1)
    assert result.prescribed_motion.right_arm_start == initial_guess.right_arm_start
    assert result.objective == -0.5
    assert captured["options"]["ipopt.max_iter"] == 3
    assert captured["options"]["ipopt.print_level"] == 5
    assert captured["options"]["print_time"] == 1
    assert np.asarray(captured["p"], dtype=float).shape == (25,)
    assert np.asarray(captured["p"], dtype=float)[0] == initial_guess.right_arm_start
    assert np.asarray(captured["x0"], dtype=float).shape == (
        (optimizer.interval_count + 1) * dms_module.ROOT_STATE_SIZE
        + 2 * (optimizer.active_control_count + 1) * dms_module.PLANE_STATE_SIZE
        + 2 * optimizer.active_control_count,
    )
    assert np.asarray(captured["lbx"], dtype=float).dtype == float
    assert np.asarray(captured["ubx"], dtype=float).dtype == float
    control_lower_bounds = np.asarray(captured["lbx"], dtype=float)[-2 * optimizer.active_control_count :]
    control_upper_bounds = np.asarray(captured["ubx"], dtype=float)[-2 * optimizer.active_control_count :]
    np.testing.assert_allclose(control_lower_bounds, -optimizer.jerk_bound)
    np.testing.assert_allclose(control_upper_bounds, optimizer.jerk_bound)


def test_direct_multiple_shooting_sweep_keeps_the_best_successful_candidate(tmp_path: Path) -> None:
    """The discrete sweep should retain the best successful fixed-start solution."""

    optimizer = DirectMultipleShootingOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=1.0, steps=201, integrator="rk4", rk4_step=0.005),
        shooting_step=0.02,
    )

    start_times = np.array([0.10, 0.12, 0.14], dtype=float)
    optimizer.candidate_start_times = lambda: start_times  # type: ignore[method-assign]

    def fake_solve_fixed_start(
        _initial_guess,
        *,
        right_arm_start: float,
        max_iter: int = 100,
        print_level: int = 0,
        print_time: bool = False,
    ):
        del max_iter, print_level, print_time
        rounded_start = round(right_arm_start, 2)
        objective_map = {0.10: -0.20, 0.12: -0.50, 0.14: -0.80}
        success_map = {0.10: True, 0.12: False, 0.14: True}
        return dms_module.DirectMultipleShootingResult(
            variables=TwistOptimizationVariables(
                right_arm_start=right_arm_start,
                left_plane_initial=0.0,
                left_plane_final=0.0,
                right_plane_initial=0.0,
                right_plane_final=0.0,
            ),
            right_arm_start_node_index=int(round(right_arm_start / optimizer.shooting_step)),
            left_plane_jerk=np.zeros(optimizer.active_control_count),
            right_plane_jerk=np.zeros(optimizer.active_control_count),
            node_times=optimizer.node_times.copy(),
            arm_node_times=optimizer.arm_node_times.copy(),
            root_state_nodes=np.zeros((dms_module.ROOT_STATE_SIZE, optimizer.interval_count + 1)),
            left_plane_state_nodes=np.zeros((dms_module.PLANE_STATE_SIZE, optimizer.active_control_count + 1)),
            right_plane_state_nodes=np.zeros((dms_module.PLANE_STATE_SIZE, optimizer.active_control_count + 1)),
            prescribed_motion=PiecewiseConstantJerkArmMotion(
                left_plane=PiecewiseConstantJerkTrajectory(
                    q0=0.0,
                    qdot0=0.0,
                    qddot0=0.0,
                    step=optimizer.shooting_step,
                    jerks=np.zeros(optimizer.active_control_count),
                    active_start=0.0,
                    active_end=dms_module.LEFT_ARM_ACTIVE_DURATION,
                    total_duration=optimizer.configuration.final_time,
                ),
                right_plane=PiecewiseConstantJerkTrajectory(
                    q0=0.0,
                    qdot0=0.0,
                    qddot0=0.0,
                    step=optimizer.shooting_step,
                    jerks=np.zeros(optimizer.active_control_count),
                    active_start=0.0,
                    active_end=dms_module.RIGHT_ARM_ACTIVE_DURATION,
                    total_duration=optimizer.configuration.final_time - right_arm_start,
                ),
                right_arm_start=right_arm_start,
            ),
            simulation=type(
                "SimulationResult",
                (),
                {
                    "final_twist_angle": 2.0 * np.pi * objective_map[rounded_start],
                    "final_twist_turns": objective_map[rounded_start],
                },
            )(),
            objective=objective_map[rounded_start],
            solver_status="Solve_Succeeded" if success_map[rounded_start] else "Maximum_Iterations_Exceeded",
            success=success_map[rounded_start],
        )

    optimizer.solve_fixed_start = fake_solve_fixed_start  # type: ignore[method-assign]

    sweep = optimizer.solve(
        TwistOptimizationVariables(
            right_arm_start=0.10,
            left_plane_initial=0.0,
            left_plane_final=0.0,
            right_plane_initial=0.0,
            right_plane_final=0.0,
        )
    )

    np.testing.assert_allclose(sweep.start_times, start_times)
    np.testing.assert_allclose(sweep.objective_values, [-0.20, -0.50, -0.80])
    np.testing.assert_allclose(sweep.final_twist_turns, [-0.20, -0.50, -0.80])
    np.testing.assert_array_equal(sweep.success_mask, [True, False, True])
    assert sweep.best_result.variables.right_arm_start == 0.14

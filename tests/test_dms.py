"""Tests for the direct multiple-shooting optimizer."""

from __future__ import annotations

from pathlib import Path

import casadi as ca
import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.optimization import DirectMultipleShootingOptimizer
from best_tilting_plane.optimization import dms as dms_module
from best_tilting_plane.optimization import solver_options as solver_options_module
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


def test_direct_multiple_shooting_candidate_start_times_use_the_reduced_search_window(
    tmp_path: Path,
) -> None:
    """The DMS sweep should test only the reduced `t1` window on the shooting grid."""

    optimizer = DirectMultipleShootingOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=1.0, steps=201, integrator="rk4", rk4_step=0.005),
        shooting_step=0.02,
    )

    np.testing.assert_allclose(
        optimizer.candidate_start_times(),
        np.arange(0.24, 0.44 + 0.001, 0.02, dtype=float),
    )


def test_direct_multiple_shooting_uses_a_very_small_default_jerk_regularization(
    tmp_path: Path,
) -> None:
    """The DMS default regularization should be small enough not to flatten the arm planes."""

    optimizer = DirectMultipleShootingOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=1.0, steps=201, integrator="rk4", rk4_step=0.005),
        shooting_step=0.02,
    )

    assert optimizer.jerk_regularization == 1e-9


def test_direct_multiple_shooting_solve_fixed_start_builds_float_bounds_and_returns_motion(
    monkeypatch,
    tmp_path: Path,
    capsys,
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
    configured_threads: list[int] = []

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
    monkeypatch.setattr(
        dms_module,
        "configure_optimization_threads",
        lambda thread_count=6: configured_threads.append(int(thread_count)) or int(thread_count),
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
    monkeypatch.setattr(
        solver_options_module,
        "locate_ipopt_hsl_library",
        lambda: "/tmp/libhsl.dylib",
    )

    result = optimizer.solve_fixed_start(
        initial_guess,
        right_arm_start=initial_guess.right_arm_start,
        max_iter=3,
        print_level=5,
        print_time=True,
    )
    stdout = capsys.readouterr().out

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
    assert configured_threads == [6]
    assert captured["options"]["ipopt.max_iter"] == 3
    assert captured["options"]["ipopt.print_level"] == 5
    assert captured["options"]["ipopt.linear_solver"] == "ma57"
    assert captured["options"]["ipopt.hsllib"] == "/tmp/libhsl.dylib"
    assert captured["options"]["print_time"] == 1
    assert optimizer._constraint_count == optimizer.interval_count * (
        dms_module.ROOT_STATE_SIZE + 2 * dms_module.PLANE_STATE_SIZE
    )
    assert np.asarray(captured["lbg"], dtype=float).shape == (optimizer._constraint_count,)
    assert "DMS objective terms:" in stdout
    assert "twist=" in stdout
    assert "jerk_reg=" in stdout
    assert "ipopt_f=" in stdout
    assert np.asarray(captured["p"], dtype=float).shape == (
        dms_module.ELEVATION_STAGE_BLOCK_SIZE * optimizer.interval_count,
    )
    assert np.asarray(captured["x0"], dtype=float).shape == (
        (optimizer.interval_count + 1) * dms_module.ROOT_STATE_SIZE
        + 2 * (optimizer.interval_count + 1) * dms_module.PLANE_STATE_SIZE
        + 2 * optimizer.interval_count,
    )
    assert np.asarray(captured["lbx"], dtype=float).dtype == float
    assert np.asarray(captured["ubx"], dtype=float).dtype == float
    all_lower_bounds = np.asarray(captured["lbx"], dtype=float)
    all_upper_bounds = np.asarray(captured["ubx"], dtype=float)
    control_block_size = optimizer.interval_count
    left_control_lower_bounds = all_lower_bounds[-2 * control_block_size : -control_block_size]
    left_control_upper_bounds = all_upper_bounds[-2 * control_block_size : -control_block_size]
    right_control_lower_bounds = all_lower_bounds[-control_block_size:]
    right_control_upper_bounds = all_upper_bounds[-control_block_size:]
    np.testing.assert_allclose(left_control_lower_bounds[: optimizer.active_control_count], -optimizer.jerk_bound)
    np.testing.assert_allclose(left_control_upper_bounds[: optimizer.active_control_count], optimizer.jerk_bound)
    np.testing.assert_allclose(left_control_lower_bounds[optimizer.active_control_count :], 0.0)
    np.testing.assert_allclose(left_control_upper_bounds[optimizer.active_control_count :], 0.0)
    np.testing.assert_allclose(
        right_control_lower_bounds[: int(round(initial_guess.right_arm_start / optimizer.shooting_step))],
        0.0,
    )
    np.testing.assert_allclose(
        right_control_upper_bounds[: int(round(initial_guess.right_arm_start / optimizer.shooting_step))],
        0.0,
    )


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
        previous_result=None,
        max_iter: int = 100,
        print_level: int = 0,
        print_time: bool = False,
    ):
        del previous_result, max_iter, print_level, print_time
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


def test_direct_multiple_shooting_sweep_warm_starts_each_node_with_previous_solution(tmp_path: Path) -> None:
    """The discrete sweep should pass the previous node solution as warm start to the next solve."""

    optimizer = DirectMultipleShootingOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=1.0, steps=201, integrator="rk4", rk4_step=0.005),
        shooting_step=0.02,
    )

    start_times = np.array([0.10, 0.12, 0.14], dtype=float)
    optimizer.candidate_start_times = lambda: start_times  # type: ignore[method-assign]
    previous_inputs: list[float | None] = []

    def fake_solve_fixed_start(
        _initial_guess,
        *,
        right_arm_start: float,
        previous_result=None,
        max_iter: int = 100,
        print_level: int = 0,
        print_time: bool = False,
    ):
        del max_iter, print_level, print_time
        previous_inputs.append(None if previous_result is None else previous_result.variables.right_arm_start)
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
                    "final_twist_angle": 2.0 * np.pi * right_arm_start,
                    "final_twist_turns": right_arm_start,
                },
            )(),
            objective=right_arm_start,
            solver_status="Solve_Succeeded",
            success=True,
            warm_start_primal=np.full(10, right_arm_start),
        )

    optimizer.solve_fixed_start = fake_solve_fixed_start  # type: ignore[method-assign]

    optimizer.solve(
        TwistOptimizationVariables(
            right_arm_start=0.10,
            left_plane_initial=0.0,
            left_plane_final=0.0,
            right_plane_initial=0.0,
            right_plane_final=0.0,
        )
    )

    assert previous_inputs == [None, 0.10, 0.12]


def test_direct_multiple_shooting_fixed_start_passes_previous_primal_and_duals_to_ipopt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A previous node solution should be forwarded to IPOPT as a warm start."""

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
            size = np.asarray(kwargs["x0"], dtype=float).shape[0]
            return {
                "x": ca.DM(np.asarray(kwargs["x0"], dtype=float)),
                "f": ca.DM([[-0.5]]),
                "lam_x": ca.DM(np.zeros(size)),
                "lam_g": ca.DM(np.zeros(np.asarray(kwargs["lbg"], dtype=float).shape[0])),
            }

        @staticmethod
        def stats():
            return {"return_status": "Solve_Succeeded"}

    def fake_nlpsol(_name, _solver_name, _problem, options):
        captured["options"] = options
        return FakeSolver()

    monkeypatch.setattr(dms_module, "PredictiveAerialTwistSimulator", FakeSimulator)
    monkeypatch.setattr(dms_module.ca, "nlpsol", fake_nlpsol)

    previous_result = dms_module.DirectMultipleShootingResult(
        variables=initial_guess,
        right_arm_start_node_index=5,
        left_plane_jerk=np.zeros(optimizer.active_control_count),
        right_plane_jerk=np.zeros(optimizer.active_control_count),
        node_times=optimizer.node_times.copy(),
        arm_node_times=optimizer.arm_node_times.copy(),
        root_state_nodes=np.zeros((dms_module.ROOT_STATE_SIZE, optimizer.interval_count + 1)),
        left_plane_state_nodes=np.zeros((dms_module.PLANE_STATE_SIZE, optimizer.active_control_count + 1)),
        right_plane_state_nodes=np.zeros((dms_module.PLANE_STATE_SIZE, optimizer.active_control_count + 1)),
        prescribed_motion=warm_start_motion,
        simulation=FakeSimulator.simulate(),
        objective=-0.5,
        solver_status="Solve_Succeeded",
        success=True,
        warm_start_primal=np.full(
            (optimizer.interval_count + 1) * dms_module.ROOT_STATE_SIZE
            + 2 * (optimizer.interval_count + 1) * dms_module.PLANE_STATE_SIZE
            + 2 * optimizer.interval_count,
            42.0,
        ),
        warm_start_lam_x=np.full(
            (optimizer.interval_count + 1) * dms_module.ROOT_STATE_SIZE
            + 2 * (optimizer.interval_count + 1) * dms_module.PLANE_STATE_SIZE
            + 2 * optimizer.interval_count,
            1.5,
        ),
        warm_start_lam_g=np.full(optimizer.interval_count * (dms_module.ROOT_STATE_SIZE + 2 * dms_module.PLANE_STATE_SIZE), 2.5),
    )

    optimizer.solve_fixed_start(
        initial_guess,
        right_arm_start=initial_guess.right_arm_start,
        previous_result=previous_result,
        max_iter=3,
        print_level=5,
        print_time=True,
    )

    assert "lam_x0" in captured
    assert "lam_g0" in captured
    assert np.asarray(captured["lam_x0"], dtype=float).shape == previous_result.warm_start_primal.shape
    assert np.asarray(captured["lam_g0"], dtype=float).shape == previous_result.warm_start_lam_g.shape

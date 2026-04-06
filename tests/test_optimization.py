"""Tests for the IPOPT-based optimization helpers."""

from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.optimization import (
    IpoptBounds,
    TwistStrategyOptimizer,
    optimize_black_box_ipopt,
)
from best_tilting_plane.optimization import ipopt as ipopt_module
from best_tilting_plane.optimization import solver_options as solver_options_module
from best_tilting_plane.simulation import SimulationConfiguration, TwistOptimizationVariables


def test_build_ipopt_solver_options_prefers_ma57_when_hsl_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared IPOPT options should switch to MA57 when one HSL library is available."""

    monkeypatch.setattr(
        solver_options_module,
        "locate_ipopt_hsl_library",
        lambda: "/tmp/libhsl.dylib",
    )

    options = solver_options_module.build_ipopt_solver_options(
        max_iter=25,
        print_level=3,
        print_time=True,
        expand=True,
        warm_start=True,
    )

    assert options["ipopt.linear_solver"] == "ma57"
    assert options["ipopt.hsllib"] == "/tmp/libhsl.dylib"
    assert options["expand"] is True
    assert options["ipopt.warm_start_init_point"] == "yes"
    assert options["ipopt.max_iter"] == 25
    assert options["ipopt.print_level"] == 3
    assert options["print_time"] == 1


def test_configure_optimization_threads_forces_six_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared optimization helper should force the project-wide thread count to six."""

    for variable_name in solver_options_module.THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)

    configured = solver_options_module.configure_optimization_threads()

    assert configured == 6
    for variable_name in solver_options_module.THREAD_ENVIRONMENT_VARIABLES:
        assert solver_options_module.os.environ[variable_name] == "6"


def test_black_box_ipopt_solves_a_simple_quadratic_problem() -> None:
    """The generic IPOPT wrapper should solve a smooth bounded quadratic objective."""

    def evaluator(vector: np.ndarray) -> float:
        x, y = vector
        return float((x - 1.5) ** 2 + (y + 2.0) ** 2)

    result = optimize_black_box_ipopt(
        evaluator,
        initial_guess=np.array([0.0, 0.0]),
        bounds=IpoptBounds(lower=np.array([-5.0, -5.0]), upper=np.array([5.0, 5.0])),
        max_iter=20,
    )

    np.testing.assert_allclose(result.solution, np.array([1.5, -2.0]), atol=1e-4)
    assert result.objective == pytest.approx(0.0, abs=1e-8)
    assert result.success


def test_black_box_ipopt_forwards_iteration_display_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The IPOPT wrapper should forward the iteration-display options to CasADi."""

    captured_options = {}

    class FakeSolver:
        def __call__(self, **_kwargs):
            return {"x": ca.DM([[0.0], [0.0]]), "f": ca.DM([[0.0]])}

        @staticmethod
        def stats():
            return {"return_status": "Solve_Succeeded"}

    def fake_nlpsol(_name, _solver_name, _problem, options):
        captured_options.update(options)
        return FakeSolver()

    monkeypatch.setattr(ipopt_module.ca, "nlpsol", fake_nlpsol)
    monkeypatch.setattr(ipopt_module, "build_ipopt_solver_options", solver_options_module.build_ipopt_solver_options)
    monkeypatch.setattr(
        solver_options_module,
        "locate_ipopt_hsl_library",
        lambda: "/tmp/libhsl.dylib",
    )

    optimize_black_box_ipopt(
        lambda vector: float(np.sum(vector**2)),
        initial_guess=np.array([0.0, 0.0]),
        bounds=IpoptBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0])),
        print_level=5,
        print_time=True,
    )

    assert captured_options["ipopt.print_level"] == 5
    assert captured_options["print_time"] == 1
    assert captured_options["ipopt.linear_solver"] == "ma57"
    assert captured_options["ipopt.hsllib"] == "/tmp/libhsl.dylib"


def test_twist_strategy_optimizer_vector_round_trip() -> None:
    """The structured decision variables should round-trip through the flat IPOPT vector."""

    variables = TwistOptimizationVariables(0.2, -0.3, 0.4, 0.5, -0.6)
    vector = TwistStrategyOptimizer.to_vector(variables)
    rebuilt = TwistStrategyOptimizer.from_vector(vector)

    assert rebuilt == variables


def test_twist_strategy_objective_returns_a_finite_value(tmp_path: Path) -> None:
    """The real simulator-backed objective should produce a finite scalar value."""

    optimizer = TwistStrategyOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=0.08, steps=9),
    )
    objective, result = optimizer.evaluate(np.zeros(5))

    assert np.isfinite(objective)
    assert np.isfinite(result.final_twist_angle)


def test_twist_strategy_objective_now_matches_the_final_twist_angle(tmp_path: Path) -> None:
    """The minimization objective should directly equal the final twist angle."""

    optimizer = TwistStrategyOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=0.08, steps=9, integrator="rk4", rk4_step=0.005),
    )

    objective, result = optimizer.evaluate(np.zeros(5))

    assert objective == pytest.approx(result.final_twist_angle)


def test_right_arm_start_only_optimizer_keeps_both_arm_planes_at_zero(
    tmp_path: Path,
) -> None:
    """The reduced 2D mode should only optimize the second-arm start time."""

    optimizer = TwistStrategyOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=0.08, steps=9, integrator="rk4", rk4_step=0.005),
    )

    result = optimizer.optimize_right_arm_start_only(0.2, max_iter=5)

    assert result.variables.left_plane_initial == pytest.approx(0.0)
    assert result.variables.left_plane_final == pytest.approx(0.0)
    assert result.variables.right_plane_initial == pytest.approx(0.0)
    assert result.variables.right_plane_final == pytest.approx(0.0)
    assert 0.0 <= result.variables.right_arm_start <= 0.7
    assert result.solver_status == "Discrete_Sweep"


def test_right_arm_start_only_sweep_uses_the_discrete_0p02_grid(tmp_path: Path) -> None:
    """The reduced 1D mode should scan the admissible start times on the 0.02 s grid."""

    optimizer = TwistStrategyOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=0.08, steps=9, integrator="rk4", rk4_step=0.01),
    )

    sweep = optimizer.sweep_right_arm_start_only(step=0.02)

    np.testing.assert_allclose(sweep.start_times, np.arange(0.0, 0.7 + 0.001, 0.02))
    assert sweep.best_result.variables.left_plane_initial == pytest.approx(0.0)
    assert sweep.best_result.variables.left_plane_final == pytest.approx(0.0)
    assert sweep.best_result.variables.right_plane_initial == pytest.approx(0.0)
    assert sweep.best_result.variables.right_plane_final == pytest.approx(0.0)
    assert sweep.objective_values.shape == sweep.start_times.shape
    assert sweep.final_twist_turns.shape == sweep.start_times.shape


def test_twist_strategy_optimizer_evaluate_right_arm_start_only_matches_zero_plane_vector(
    tmp_path: Path,
) -> None:
    """The reduced evaluation helper should match the full vector evaluation at zero plane angles."""

    optimizer = TwistStrategyOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=0.08, steps=9, integrator="rk4", rk4_step=0.01),
    )
    right_arm_start = 0.2
    variables = optimizer.zero_plane_variables(right_arm_start)

    reduced_objective, _ = optimizer.evaluate_right_arm_start_only(right_arm_start)
    full_objective, _ = optimizer.evaluate(optimizer.to_vector(variables))

    assert reduced_objective == pytest.approx(full_objective)

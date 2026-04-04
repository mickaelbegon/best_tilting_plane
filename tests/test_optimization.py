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
from best_tilting_plane.simulation import SimulationConfiguration, TwistOptimizationVariables


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

    optimize_black_box_ipopt(
        lambda vector: float(np.sum(vector**2)),
        initial_guess=np.array([0.0, 0.0]),
        bounds=IpoptBounds(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0])),
        print_level=5,
        print_time=True,
    )

    assert captured_options["ipopt.print_level"] == 5
    assert captured_options["print_time"] == 1


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


def test_symbolic_objective_matches_black_box_evaluation_for_5d_mode(tmp_path: Path) -> None:
    """The symbolic RK4 objective should match the classic simulator-backed evaluation."""

    optimizer = TwistStrategyOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=0.08, steps=9, integrator="rk4", rk4_step=0.01),
    )
    vector = np.array([0.2, -0.1, 0.0, 0.1, 0.0], dtype=float)

    symbolic_value = float(optimizer._build_symbolic_objective_function(5)(vector))
    objective, _result = optimizer.evaluate(vector)

    assert symbolic_value == pytest.approx(objective, rel=1e-8, abs=1e-8)


def test_symbolic_objective_matches_black_box_evaluation_for_1d_mode(tmp_path: Path) -> None:
    """The reduced symbolic RK4 objective should match the reduced classic evaluation."""

    optimizer = TwistStrategyOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=0.08, steps=9, integrator="rk4", rk4_step=0.01),
    )

    symbolic_value = float(optimizer._build_symbolic_objective_function(1)(np.array([0.2], dtype=float)))
    objective, _result = optimizer.evaluate_right_arm_start_only(0.2)

    assert symbolic_value == pytest.approx(objective, rel=1e-8, abs=1e-8)

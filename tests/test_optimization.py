"""Tests for the IPOPT-based optimization helpers."""

from pathlib import Path

import numpy as np
import pytest

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.optimization import (
    IpoptBounds,
    TwistStrategyOptimizer,
    optimize_black_box_ipopt,
)
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

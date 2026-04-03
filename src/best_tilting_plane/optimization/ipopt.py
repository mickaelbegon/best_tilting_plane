"""Black-box IPOPT optimization for the predictive aerial twisting simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import biorbd
import casadi as ca
import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.simulation import (
    AerialSimulationResult,
    PredictiveAerialTwistSimulator,
    PrescribedArmMotion,
    SimulationConfiguration,
    TwistOptimizationVariables,
)


@dataclass(frozen=True)
class IpoptBounds:
    """Simple lower and upper bounds for an IPOPT problem."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        """Validate the bound dimensions."""

        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.shape != upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")
        if np.any(lower > upper):
            raise ValueError(
                "Each lower bound must be smaller than or equal to the corresponding upper bound."
            )


@dataclass(frozen=True)
class IpoptResult:
    """Generic IPOPT result for a black-box scalar objective."""

    solution: np.ndarray
    objective: float
    solver_status: str
    success: bool


@dataclass(frozen=True)
class TwistOptimizationResult:
    """Optimization result specialized for the aerial twisting strategy."""

    variables: TwistOptimizationVariables
    final_twist_angle: float
    final_twist_turns: float
    objective: float
    solver_status: str
    success: bool


class _ScalarObjectiveCallback(ca.Callback):
    """CasADi callback that exposes a Python scalar objective to IPOPT."""

    def __init__(self, name: str, evaluator: Callable[[np.ndarray], float], size: int) -> None:
        """Store the evaluator and enable finite-difference derivatives."""

        self._evaluator = evaluator
        self._size = size
        super().__init__()
        self.construct(name, {"enable_fd": True})

    def get_n_in(self) -> int:
        """Return the number of inputs."""

        return 1

    def get_n_out(self) -> int:
        """Return the number of outputs."""

        return 1

    def get_sparsity_in(self, _index: int) -> ca.Sparsity:
        """Describe the vector-valued decision variable input."""

        return ca.Sparsity.dense(self._size, 1)

    def get_sparsity_out(self, _index: int) -> ca.Sparsity:
        """Describe the scalar objective output."""

        return ca.Sparsity.dense(1, 1)

    def eval(self, args: list[ca.DM]) -> list[ca.DM]:
        """Evaluate the scalar objective."""

        value = float(self._evaluator(np.asarray(args[0].full(), dtype=float).reshape(-1)))
        return [ca.DM([[value]])]


def optimize_black_box_ipopt(
    evaluator: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    bounds: IpoptBounds,
    *,
    max_iter: int = 50,
    print_level: int = 0,
) -> IpoptResult:
    """Optimize a scalar black-box objective with IPOPT and finite-difference derivatives."""

    x0 = np.asarray(initial_guess, dtype=float).reshape(-1)
    if x0.shape != np.asarray(bounds.lower, dtype=float).shape:
        raise ValueError("The initial guess must have the same shape as the bounds.")

    callback = _ScalarObjectiveCallback("black_box_objective", evaluator, x0.size)
    x_symbol = ca.MX.sym("x", x0.size)
    solver = ca.nlpsol(
        "twist_solver",
        "ipopt",
        {"x": x_symbol, "f": callback(x_symbol)},
        {
            "ipopt.max_iter": int(max_iter),
            "ipopt.print_level": int(print_level),
            "print_time": 0,
        },
    )
    solution = solver(
        x0=x0, lbx=np.asarray(bounds.lower, dtype=float), ubx=np.asarray(bounds.upper, dtype=float)
    )
    status = solver.stats()["return_status"]
    solution_vector = np.asarray(solution["x"].full(), dtype=float).reshape(-1)
    normalized_status = status.lower().replace("_", " ")
    return IpoptResult(
        solution=solution_vector,
        objective=float(solution["f"]),
        solver_status=status,
        success=(
            "success" in normalized_status
            or "succeeded" in normalized_status
            or "solved" in normalized_status
        ),
    )


class TwistStrategyOptimizer:
    """Optimize the arm strategy that maximizes the final number of twists."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        configuration: SimulationConfiguration | None = None,
        model: biorbd.Model | None = None,
    ) -> None:
        """Store the reusable model and simulation settings."""

        self.model_path = str(model_path)
        self.configuration = configuration or SimulationConfiguration()
        self.model = model if model is not None else biorbd.Model(self.model_path)
        self._cache: dict[tuple[float, ...], tuple[float, AerialSimulationResult]] = {}

    @classmethod
    def from_builder(
        cls,
        output_path: str | Path,
        *,
        model_builder: ReducedAerialBiomod | None = None,
        configuration: SimulationConfiguration | None = None,
    ) -> "TwistStrategyOptimizer":
        """Generate the model file and build an optimizer on top of it."""

        builder = model_builder or ReducedAerialBiomod()
        model_path = builder.write(output_path)
        return cls(model_path, configuration=configuration)

    @staticmethod
    def default_bounds() -> IpoptBounds:
        """Return the default bounds for the five decision variables."""

        return IpoptBounds(
            lower=np.array([0.0, -np.pi, -np.pi, -np.pi, -np.pi], dtype=float),
            upper=np.array([0.7, np.pi, np.pi, np.pi, np.pi], dtype=float),
        )

    @staticmethod
    def to_vector(variables: TwistOptimizationVariables) -> np.ndarray:
        """Convert the decision variables to a flat optimization vector."""

        return np.array(
            [
                variables.right_arm_start,
                variables.left_plane_initial,
                variables.left_plane_final,
                variables.right_plane_initial,
                variables.right_plane_final,
            ],
            dtype=float,
        )

    @staticmethod
    def from_vector(vector: np.ndarray) -> TwistOptimizationVariables:
        """Convert a flat optimization vector to structured decision variables."""

        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.size != 5:
            raise ValueError("The twist optimizer expects exactly 5 decision variables.")
        return TwistOptimizationVariables(
            right_arm_start=float(values[0]),
            left_plane_initial=float(values[1]),
            left_plane_final=float(values[2]),
            right_plane_initial=float(values[3]),
            right_plane_final=float(values[4]),
        )

    def evaluate(self, vector: np.ndarray) -> tuple[float, AerialSimulationResult]:
        """Evaluate the black-box objective and return the cached simulation result."""

        point = tuple(np.round(np.asarray(vector, dtype=float).reshape(-1), decimals=10))
        cached = self._cache.get(point)
        if cached is not None:
            return cached

        variables = self.from_vector(np.asarray(point, dtype=float))
        simulator = PredictiveAerialTwistSimulator(
            self.model_path,
            PrescribedArmMotion(variables),
            configuration=self.configuration,
            model=self.model,
        )
        result = simulator.simulate()
        objective = -result.final_twist_angle
        self._cache[point] = (objective, result)
        return objective, result

    def objective(self, vector: np.ndarray) -> float:
        """Return the IPOPT scalar objective, i.e. minus the final twist angle."""

        return self.evaluate(vector)[0]

    def optimize(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        bounds: IpoptBounds | None = None,
        max_iter: int = 50,
        print_level: int = 0,
    ) -> TwistOptimizationResult:
        """Optimize the decision variables and return the best twist strategy."""

        chosen_bounds = bounds or self.default_bounds()
        raw_result = optimize_black_box_ipopt(
            self.objective,
            self.to_vector(initial_guess),
            chosen_bounds,
            max_iter=max_iter,
            print_level=print_level,
        )
        variables = self.from_vector(raw_result.solution)
        _, simulation = self.evaluate(raw_result.solution)
        return TwistOptimizationResult(
            variables=variables,
            final_twist_angle=simulation.final_twist_angle,
            final_twist_turns=simulation.final_twist_turns,
            objective=raw_result.objective,
            solver_status=raw_result.solver_status,
            success=raw_result.success,
        )

"""Black-box IPOPT optimization for the predictive aerial twisting simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import biorbd
import casadi as ca
import numpy as np
import biorbd_casadi as biorbd_ca

from best_tilting_plane.modeling import (
    LEFT_ARM_PLANE_BOUNDS_DEG,
    RIGHT_ARM_PLANE_BOUNDS_DEG,
    ReducedAerialBiomod,
)
from best_tilting_plane.optimization.solver_options import build_ipopt_solver_options
from best_tilting_plane.simulation import (
    AerialSimulationResult,
    build_piecewise_constant_jerk_arm_motion,
    PredictiveAerialTwistSimulator,
    SimulationConfiguration,
    TwistOptimizationVariables,
)

RIGHT_ARM_START_BOUNDS = (0.0, 0.7)
LEFT_ARM_PLANE_BOUNDS = tuple(np.deg2rad(LEFT_ARM_PLANE_BOUNDS_DEG))
RIGHT_ARM_PLANE_BOUNDS = tuple(np.deg2rad(RIGHT_ARM_PLANE_BOUNDS_DEG))
SYMBOLIC_RK4_TOLERANCE = 1e-12


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
    simulation: AerialSimulationResult | None = None


@dataclass(frozen=True)
class RightArmStartSweepResult:
    """Discrete sweep result for the 1D right-arm-start optimization mode."""

    best_result: TwistOptimizationResult
    candidate_results: tuple[TwistOptimizationResult, ...]

    @property
    def start_times(self) -> np.ndarray:
        """Return the scanned start times."""

        return np.array([result.variables.right_arm_start for result in self.candidate_results], dtype=float)

    @property
    def objective_values(self) -> np.ndarray:
        """Return the objective value at every scanned start time."""

        return np.array([result.objective for result in self.candidate_results], dtype=float)

    @property
    def final_twist_turns(self) -> np.ndarray:
        """Return the final twist count at every scanned start time."""

        return np.array([result.final_twist_turns for result in self.candidate_results], dtype=float)

    @property
    def success_mask(self) -> np.ndarray:
        """Return the solver-success mask for the scanned start times."""

        return np.array([result.success for result in self.candidate_results], dtype=bool)


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
    print_time: bool = False,
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
        build_ipopt_solver_options(
            max_iter=max_iter,
            print_level=print_level,
            print_time=print_time,
        ),
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
    """Optimize the arm strategy that minimizes the final number of twists."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        configuration: SimulationConfiguration | None = None,
        model: biorbd.Model | None = None,
    ) -> None:
        """Store the reusable model and simulation settings."""

        self.model_path = str(model_path)
        self.configuration = configuration or SimulationConfiguration(
            integrator="rk4",
            rk4_step=0.005,
        )
        self.model = model if model is not None else biorbd.Model(self.model_path)
        self.symbolic_model = biorbd_ca.Model(self.model_path)
        self._cache: dict[tuple[float, ...], tuple[float, AerialSimulationResult]] = {}
        self._symbolic_objectives: dict[int, ca.Function] = {}

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
            lower=np.array(
                [
                    RIGHT_ARM_START_BOUNDS[0],
                    LEFT_ARM_PLANE_BOUNDS[0],
                    LEFT_ARM_PLANE_BOUNDS[0],
                    RIGHT_ARM_PLANE_BOUNDS[0],
                    RIGHT_ARM_PLANE_BOUNDS[0],
                ],
                dtype=float,
            ),
            upper=np.array(
                [
                    RIGHT_ARM_START_BOUNDS[1],
                    LEFT_ARM_PLANE_BOUNDS[1],
                    LEFT_ARM_PLANE_BOUNDS[1],
                    RIGHT_ARM_PLANE_BOUNDS[1],
                    RIGHT_ARM_PLANE_BOUNDS[1],
                ],
                dtype=float,
            ),
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

        variables = TwistOptimizationVariables(
            right_arm_start=float(point[0]),
            left_plane_initial=float(point[1]),
            left_plane_final=float(point[2]),
            right_plane_initial=float(point[3]),
            right_plane_final=float(point[4]),
            contact_twist_rate=float(self.configuration.contact_twist_rate),
        )
        simulator = PredictiveAerialTwistSimulator(
            self.model_path,
            build_piecewise_constant_jerk_arm_motion(
                variables,
                total_time=self.configuration.final_time,
                step=0.02,
            ),
            configuration=self.configuration,
            model=self.model,
        )
        result = simulator.simulate()
        objective = result.final_twist_angle
        self._cache[point] = (objective, result)
        return objective, result

    def objective(self, vector: np.ndarray) -> float:
        """Return the IPOPT scalar objective, i.e. the final twist angle."""

        return self.evaluate(vector)[0]

    @staticmethod
    def zero_plane_variables(right_arm_start: float) -> TwistOptimizationVariables:
        """Return a reduced strategy where both arm planes stay at zero."""

        return TwistOptimizationVariables(
            right_arm_start=float(right_arm_start),
            left_plane_initial=0.0,
            left_plane_final=0.0,
            right_plane_initial=0.0,
            right_plane_final=0.0,
        )

    def _fixed_contact_twist_rate(self) -> float:
        """Return the contact-twist rate carried as a fixed parameter of the current solve."""

        return float(self.configuration.contact_twist_rate)

    @staticmethod
    def right_arm_start_only_bounds() -> IpoptBounds:
        """Return the 1D bounds used by the reduced optimization mode."""

        return IpoptBounds(
            lower=np.array([RIGHT_ARM_START_BOUNDS[0]], dtype=float),
            upper=np.array([RIGHT_ARM_START_BOUNDS[1]], dtype=float),
        )

    def evaluate_right_arm_start_only(
        self, right_arm_start: float
    ) -> tuple[float, AerialSimulationResult]:
        """Evaluate the reduced 1D optimization mode with both arm planes fixed at zero."""

        variables = TwistOptimizationVariables(
            right_arm_start=float(right_arm_start),
            left_plane_initial=0.0,
            left_plane_final=0.0,
            right_plane_initial=0.0,
            right_plane_final=0.0,
            contact_twist_rate=self._fixed_contact_twist_rate(),
        )
        return self.evaluate(self.to_vector(variables))

    @staticmethod
    def _quintic_profile(phase: ca.MX) -> tuple[ca.MX, ca.MX, ca.MX]:
        """Return the quintic profile and its first two derivatives."""

        phase2 = phase * phase
        phase3 = phase2 * phase
        phase4 = phase3 * phase
        phase5 = phase4 * phase
        profile = 6.0 * phase5 - 15.0 * phase4 + 10.0 * phase3
        velocity = 30.0 * phase4 - 60.0 * phase3 + 30.0 * phase2
        acceleration = 120.0 * phase3 - 180.0 * phase2 + 60.0 * phase
        return profile, velocity, acceleration

    @staticmethod
    def _clipped_phase(time: ca.MX, start: ca.MX, duration: float) -> ca.MX:
        """Return the symbolic motion phase clipped to `[0, 1]`."""

        return ca.fmin(1.0, ca.fmax(0.0, (time - start) / duration))

    @staticmethod
    def _active_motion_mask(time: ca.MX, start: ca.MX, end: ca.MX) -> ca.MX:
        """Return `1` while the motion is active and `0` otherwise."""

        return ca.if_else(ca.logic_and(time >= start, time <= end), 1.0, 0.0)

    def _symbolic_joint_kinematics(
        self, variables: ca.MX, time: ca.MX
    ) -> tuple[ca.MX, ca.MX, ca.MX]:
        """Return symbolic prescribed arm kinematics."""

        duration = 0.3
        left_start = 0.0
        right_start = variables[0]
        left_end = left_start + duration
        right_end = right_start + duration

        left_phase = self._clipped_phase(time, left_start, duration)
        right_phase = self._clipped_phase(time, right_start, duration)
        left_profile, left_velocity_profile, left_acceleration_profile = self._quintic_profile(
            left_phase
        )
        right_profile, right_velocity_profile, right_acceleration_profile = self._quintic_profile(
            right_phase
        )
        left_active = self._active_motion_mask(time, left_start, left_end)
        right_active = self._active_motion_mask(time, right_start, right_end)

        def position(profile: ca.MX, q0: ca.MX, q1: ca.MX) -> ca.MX:
            return q0 + (q1 - q0) * profile

        def velocity(active: ca.MX, profile: ca.MX, q0: ca.MX, q1: ca.MX) -> ca.MX:
            return active * (q1 - q0) * profile / duration

        def acceleration(active: ca.MX, profile: ca.MX, q0: ca.MX, q1: ca.MX) -> ca.MX:
            return active * (q1 - q0) * profile / (duration**2)

        left_plane_initial = variables[1]
        left_plane_final = variables[2]
        right_plane_initial = variables[3]
        right_plane_final = variables[4]
        left_elevation_initial = -np.pi
        left_elevation_final = 0.0
        right_elevation_initial = np.pi
        right_elevation_final = 0.0

        q_joint = ca.vertcat(
            position(left_profile, left_plane_initial, left_plane_final),
            position(left_profile, left_elevation_initial, left_elevation_final),
            position(right_profile, right_plane_initial, right_plane_final),
            position(right_profile, right_elevation_initial, right_elevation_final),
        )
        qdot_joint = ca.vertcat(
            velocity(left_active, left_velocity_profile, left_plane_initial, left_plane_final),
            velocity(left_active, left_velocity_profile, left_elevation_initial, left_elevation_final),
            velocity(right_active, right_velocity_profile, right_plane_initial, right_plane_final),
            velocity(
                right_active,
                right_velocity_profile,
                right_elevation_initial,
                right_elevation_final,
            ),
        )
        qddot_joint = ca.vertcat(
            acceleration(
                left_active,
                left_acceleration_profile,
                left_plane_initial,
                left_plane_final,
            ),
            acceleration(
                left_active,
                left_acceleration_profile,
                left_elevation_initial,
                left_elevation_final,
            ),
            acceleration(
                right_active,
                right_acceleration_profile,
                right_plane_initial,
                right_plane_final,
            ),
            acceleration(
                right_active,
                right_acceleration_profile,
                right_elevation_initial,
                right_elevation_final,
            ),
        )
        return q_joint, qdot_joint, qddot_joint

    def _symbolic_initial_state(self, variables: ca.MX) -> ca.MX:
        """Return the symbolic initial root state."""

        q_joint, qdot_joint, _ = self._symbolic_joint_kinematics(variables, ca.MX(0.0))
        q_root = ca.MX.zeros(6, 1)
        q_full = ca.vertcat(q_root, q_joint)
        qdot_without_translation = ca.vertcat(
            ca.MX.zeros(3, 1),
            ca.vertcat(
                self.configuration.somersault_rate,
                0.0,
                self.configuration.contact_twist_rate,
            ),
            qdot_joint,
        )
        # Root translations are expressed directly in the global `x, y, z` axes, so their
        # contribution to the CoM velocity is additive. Cancelling the CoM motion therefore
        # amounts to negating the CoM velocity obtained with zero translational root speed.
        translation_velocity = -self.symbolic_model.CoMdot(
            q_full,
            qdot_without_translation,
            True,
        ).to_mx()
        qdot_root = ca.vertcat(
            translation_velocity,
            self.configuration.somersault_rate,
            0.0,
            self.configuration.contact_twist_rate,
        )
        return ca.vertcat(q_root, qdot_root)

    def _symbolic_dynamics(self, time: ca.MX, state: ca.MX, variables: ca.MX) -> ca.MX:
        """Return the symbolic time derivative of the root state."""

        q_root = state[:6]
        qdot_root = state[6:]
        q_joint, qdot_joint, qddot_joint = self._symbolic_joint_kinematics(variables, time)
        q_full = ca.vertcat(q_root, q_joint)
        qdot_full = ca.vertcat(qdot_root, qdot_joint)
        qddot_root = self.symbolic_model.ForwardDynamicsFreeFloatingBase(
            q_full,
            qdot_full,
            qddot_joint,
        ).to_mx()
        return ca.vertcat(qdot_root, qddot_root)

    def _build_symbolic_objective_function(self, size: int) -> ca.Function:
        """Build and cache the symbolic final-twist objective."""

        cached = self._symbolic_objectives.get(size)
        if cached is not None:
            return cached

        if self.configuration.rk4_step is None:
            raise ValueError("The symbolic optimizer requires a fixed RK4 step.")
        if self.configuration.integrator.lower() != "rk4":
            raise ValueError("The symbolic optimizer currently supports only RK4.")

        step = float(self.configuration.rk4_step)
        step_count = int(round(self.configuration.final_time / step))
        if abs(step_count * step - self.configuration.final_time) > SYMBOLIC_RK4_TOLERANCE:
            raise ValueError("The symbolic optimizer requires `final_time` to be a multiple of `rk4_step`.")

        decision_variables = ca.MX.sym("x", size, 1)
        if size == 1:
            full_variables = ca.vertcat(decision_variables[0], 0.0, 0.0, 0.0, 0.0)
        elif size == 5:
            full_variables = decision_variables
        else:
            raise ValueError("Unsupported symbolic decision-variable size.")

        state = self._symbolic_initial_state(full_variables)
        time = 0.0
        for _ in range(step_count):
            k1 = self._symbolic_dynamics(time, state, full_variables)
            k2 = self._symbolic_dynamics(time + 0.5 * step, state + 0.5 * step * k1, full_variables)
            k3 = self._symbolic_dynamics(time + 0.5 * step, state + 0.5 * step * k2, full_variables)
            k4 = self._symbolic_dynamics(time + step, state + step * k3, full_variables)
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            time += step

        objective_function = ca.Function(
            f"final_twist_symbolic_{size}d",
            [decision_variables],
            [state[5]],
        )
        self._symbolic_objectives[size] = objective_function
        return objective_function

    def optimize_symbolic(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        bounds: IpoptBounds | None = None,
        max_iter: int = 50,
        print_level: int = 0,
        print_time: bool = False,
    ) -> TwistOptimizationResult:
        """Optimize the full 5D strategy with an exact symbolic RK4 objective."""

        chosen_bounds = bounds or self.default_bounds()
        x0 = self.to_vector(initial_guess)
        x_symbol = ca.MX.sym("x", x0.size)
        objective = self._build_symbolic_objective_function(5)(x_symbol)
        solver = ca.nlpsol(
            "twist_solver_symbolic_5d",
            "ipopt",
            {"x": x_symbol, "f": objective},
            build_ipopt_solver_options(
                max_iter=max_iter,
                print_level=print_level,
                print_time=print_time,
            ),
        )
        solution = solver(
            x0=x0,
            lbx=np.asarray(chosen_bounds.lower, dtype=float),
            ubx=np.asarray(chosen_bounds.upper, dtype=float),
        )
        status = solver.stats()["return_status"]
        solution_vector = np.asarray(solution["x"].full(), dtype=float).reshape(-1)
        variables = TwistOptimizationVariables(
            right_arm_start=float(solution_vector[0]),
            left_plane_initial=float(solution_vector[1]),
            left_plane_final=float(solution_vector[2]),
            right_plane_initial=float(solution_vector[3]),
            right_plane_final=float(solution_vector[4]),
            contact_twist_rate=self._fixed_contact_twist_rate(),
        )
        _, simulation = self.evaluate(solution_vector)
        normalized_status = status.lower().replace("_", " ")
        return TwistOptimizationResult(
            variables=variables,
            final_twist_angle=simulation.final_twist_angle,
            final_twist_turns=simulation.final_twist_turns,
            objective=float(solution["f"]),
            solver_status=status,
            success=(
                "success" in normalized_status
                or "succeeded" in normalized_status
                or "solved" in normalized_status
            ),
        )

    def optimize(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        bounds: IpoptBounds | None = None,
        max_iter: int = 50,
        print_level: int = 0,
        print_time: bool = False,
    ) -> TwistOptimizationResult:
        """Optimize the decision variables and return the best twist strategy."""

        return self.optimize_symbolic(
            initial_guess,
            bounds=bounds,
            max_iter=max_iter,
            print_level=print_level,
            print_time=print_time,
        )

    def optimize_black_box(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        bounds: IpoptBounds | None = None,
        max_iter: int = 50,
        print_level: int = 0,
        print_time: bool = False,
    ) -> TwistOptimizationResult:
        """Optimize the decision variables with the legacy black-box callback path."""

        chosen_bounds = bounds or self.default_bounds()
        raw_result = optimize_black_box_ipopt(
            self.objective,
            self.to_vector(initial_guess),
            chosen_bounds,
            max_iter=max_iter,
            print_level=print_level,
            print_time=print_time,
        )
        variables = TwistOptimizationVariables(
            right_arm_start=float(raw_result.solution[0]),
            left_plane_initial=float(raw_result.solution[1]),
            left_plane_final=float(raw_result.solution[2]),
            right_plane_initial=float(raw_result.solution[3]),
            right_plane_final=float(raw_result.solution[4]),
            contact_twist_rate=self._fixed_contact_twist_rate(),
        )
        _, simulation = self.evaluate(raw_result.solution)
        return TwistOptimizationResult(
            variables=variables,
            final_twist_angle=simulation.final_twist_angle,
            final_twist_turns=simulation.final_twist_turns,
            objective=raw_result.objective,
            solver_status=raw_result.solver_status,
            success=raw_result.success,
            simulation=simulation,
        )

    def optimize_right_arm_start_only(
        self,
        initial_right_arm_start: float,
        *,
        bounds: IpoptBounds | None = None,
        max_iter: int = 50,
        print_level: int = 0,
        print_time: bool = False,
    ) -> TwistOptimizationResult:
        """Optimize only the right-arm start time by scanning the 0.02 s grid and keeping the best."""

        del initial_right_arm_start, max_iter, print_level, print_time
        return self.sweep_right_arm_start_only(bounds=bounds).best_result

    def sweep_right_arm_start_only(
        self,
        *,
        bounds: IpoptBounds | None = None,
        step: float = 0.02,
    ) -> RightArmStartSweepResult:
        """Evaluate every admissible 1D start-time node and keep the best result."""

        chosen_bounds = bounds or self.right_arm_start_only_bounds()
        lower_bound = float(np.asarray(chosen_bounds.lower, dtype=float).reshape(-1)[0])
        upper_bound = float(np.asarray(chosen_bounds.upper, dtype=float).reshape(-1)[0])
        first_node = int(round(lower_bound / step))
        last_node = int(round(upper_bound / step))
        start_times = step * np.arange(first_node, last_node + 1, dtype=float)
        candidate_results: list[TwistOptimizationResult] = []

        for start_time in start_times:
            variables = TwistOptimizationVariables(
                right_arm_start=float(start_time),
                left_plane_initial=0.0,
                left_plane_final=0.0,
                right_plane_initial=0.0,
                right_plane_final=0.0,
                contact_twist_rate=self._fixed_contact_twist_rate(),
            )
            objective, simulation = self.evaluate(self.to_vector(variables))
            candidate_results.append(
                TwistOptimizationResult(
                    variables=variables,
                    final_twist_angle=simulation.final_twist_angle,
                    final_twist_turns=simulation.final_twist_turns,
                    objective=float(objective),
                    solver_status="Discrete_Sweep",
                    success=True,
                    simulation=simulation,
                )
            )

        candidate_results_tuple = tuple(candidate_results)
        best_result = min(candidate_results_tuple, key=lambda result: result.objective)
        return RightArmStartSweepResult(
            best_result=best_result,
            candidate_results=candidate_results_tuple,
        )


def create_right_arm_start_sweep_figure(
    *,
    start_times: np.ndarray,
    final_twist_turns: np.ndarray,
    objective_values: np.ndarray,
    best_start_time: float,
):
    """Create one figure summarizing the discrete 1D sweep over the right-arm start time."""

    import matplotlib.pyplot as plt

    del objective_values
    times = np.asarray(start_times, dtype=float)
    twists = np.asarray(final_twist_turns, dtype=float)
    best_index = int(np.argmin(np.abs(times - float(best_start_time))))

    figure, axis = plt.subplots(1, 1, figsize=(7.5, 4.5), tight_layout=True)
    axis.plot(times, twists, color="tab:blue", linewidth=1.6, marker="o", markersize=4.0)
    axis.scatter(
        [times[best_index]],
        [twists[best_index]],
        color="tab:orange",
        s=64,
        zorder=3,
        label="Meilleure solution",
    )
    axis.set_ylabel("Vrille finale (tours)")
    axis.set_xlabel("Debut bras droit t1 (s)")
    axis.set_title("Balayage 1D sur les noeuds de t1")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    return figure, axis


def show_right_arm_start_sweep_figure(
    *,
    start_times: np.ndarray,
    final_twist_turns: np.ndarray,
    objective_values: np.ndarray,
    best_start_time: float,
):
    """Open an external figure summarizing the discrete 1D sweep."""

    import matplotlib.pyplot as plt

    figure, axis = create_right_arm_start_sweep_figure(
        start_times=start_times,
        final_twist_turns=final_twist_turns,
        objective_values=objective_values,
        best_start_time=best_start_time,
    )
    figure.canvas.draw_idle()
    plt.show(block=False)
    return figure, axis

"""Classical jerk-driven direct multiple shooting for the reduced aerial model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import biorbd
import biorbd_casadi as biorbd_ca
import casadi as ca
import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.simulation import (
    AerialSimulationResult,
    PiecewiseConstantJerkArmMotion,
    PiecewiseConstantJerkTrajectory,
    PredictiveAerialTwistSimulator,
    SimulationConfiguration,
    TwistOptimizationVariables,
    approximate_first_arm_elevation_motion,
    approximate_quintic_segment_with_piecewise_constant_jerk,
)

LEFT_ARM_ACTIVE_DURATION = 0.3
RIGHT_ARM_ACTIVE_DURATION = 0.3
RIGHT_ARM_START_BOUNDS = (0.0, 0.7)
RIGHT_ARM_SWEEP_BOUNDS = (0.1, 0.7)
PLANE_STATE_SIZE = 3
ROOT_STATE_SIZE = 12
ELEVATION_STAGE_BLOCK_SIZE = 18


@dataclass(frozen=True)
class DirectMultipleShootingResult:
    """Optimization result for one fixed second-arm start time."""

    variables: TwistOptimizationVariables
    right_arm_start_node_index: int
    left_plane_jerk: np.ndarray
    right_plane_jerk: np.ndarray
    node_times: np.ndarray
    arm_node_times: np.ndarray
    root_state_nodes: np.ndarray
    left_plane_state_nodes: np.ndarray
    right_plane_state_nodes: np.ndarray
    prescribed_motion: PiecewiseConstantJerkArmMotion
    simulation: AerialSimulationResult
    objective: float
    solver_status: str
    success: bool

    @property
    def final_twist_angle(self) -> float:
        """Return the final twist angle in radians."""

        return self.simulation.final_twist_angle

    @property
    def final_twist_turns(self) -> float:
        """Return the final twist count in turns."""

        return self.simulation.final_twist_turns


@dataclass(frozen=True)
class DirectMultipleShootingSweepResult:
    """Result of the discrete sweep over admissible second-arm start nodes."""

    best_result: DirectMultipleShootingResult
    candidate_results: tuple[DirectMultipleShootingResult, ...]

    @property
    def start_times(self) -> np.ndarray:
        """Return the scanned second-arm start times."""

        return np.array([result.variables.right_arm_start for result in self.candidate_results], dtype=float)

    @property
    def objective_values(self) -> np.ndarray:
        """Return the objective value associated with every scanned start time."""

        return np.array([result.objective for result in self.candidate_results], dtype=float)

    @property
    def final_twist_turns(self) -> np.ndarray:
        """Return the final twist count associated with every scanned start time."""

        return np.array([result.final_twist_turns for result in self.candidate_results], dtype=float)

    @property
    def success_mask(self) -> np.ndarray:
        """Return whether each candidate solve reached a successful NLP status."""

        return np.array([result.success for result in self.candidate_results], dtype=bool)

    @property
    def solver_statuses(self) -> tuple[str, ...]:
        """Return the solver status associated with every scanned start time."""

        return tuple(result.solver_status for result in self.candidate_results)


class DirectMultipleShootingOptimizer:
    """Classical jerk-driven multiple shooting with one global graph and fixed known values."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        configuration: SimulationConfiguration | None = None,
        shooting_step: float = 0.02,
        jerk_regularization: float = 1e-4,
        model: biorbd.Model | None = None,
    ) -> None:
        """Store the models and the direct multiple-shooting settings."""

        self.model_path = str(model_path)
        self.configuration = configuration or SimulationConfiguration(integrator="rk4", rk4_step=0.005)
        self.shooting_step = float(shooting_step)
        self.jerk_regularization = float(jerk_regularization)
        self.model = model if model is not None else biorbd.Model(self.model_path)
        self.symbolic_model = biorbd_ca.Model(self.model_path)
        self._solver = None
        self._solver_options_key: tuple[int, int, bool] | None = None
        self._constraint_count = 0
        self._interval_parallelization = "serial"

        if self.shooting_step <= 0.0:
            raise ValueError("The shooting step must be strictly positive.")

        step_count = int(round(self.configuration.final_time / self.shooting_step))
        if abs(step_count * self.shooting_step - self.configuration.final_time) > 1e-12:
            raise ValueError("The final time must be a multiple of the shooting step.")
        active_control_count = int(round(LEFT_ARM_ACTIVE_DURATION / self.shooting_step))
        if abs(active_control_count * self.shooting_step - LEFT_ARM_ACTIVE_DURATION) > 1e-12:
            raise ValueError("The active arm duration must be a multiple of the shooting step.")

        self.interval_count = step_count
        self.active_control_count = active_control_count
        self.node_times = np.linspace(0.0, self.configuration.final_time, self.interval_count + 1)
        self.arm_node_times = np.linspace(0.0, LEFT_ARM_ACTIVE_DURATION, self.active_control_count + 1)
        elevation_fit = approximate_first_arm_elevation_motion(
            total_time=LEFT_ARM_ACTIVE_DURATION,
            step=self.shooting_step,
        )
        self.jerk_bound = float(np.max(np.abs(elevation_fit.jerks)))

    @classmethod
    def from_builder(
        cls,
        output_path: str | Path,
        *,
        model_builder: ReducedAerialBiomod | None = None,
        configuration: SimulationConfiguration | None = None,
        shooting_step: float = 0.02,
        jerk_regularization: float = 1e-4,
    ) -> "DirectMultipleShootingOptimizer":
        """Generate the model file and build a DMS optimizer on top of it."""

        builder = model_builder or ReducedAerialBiomod()
        model_path = builder.write(output_path)
        return cls(
            model_path,
            configuration=configuration,
            shooting_step=shooting_step,
            jerk_regularization=jerk_regularization,
        )

    @staticmethod
    def _quintic_profile(phase: float) -> tuple[float, float, float]:
        """Return the quintic profile and its first two derivatives."""

        phase2 = phase * phase
        phase3 = phase2 * phase
        phase4 = phase3 * phase
        phase5 = phase4 * phase
        profile = 6.0 * phase5 - 15.0 * phase4 + 10.0 * phase3
        velocity = 30.0 * phase4 - 60.0 * phase3 + 30.0 * phase2
        acceleration = 120.0 * phase3 - 180.0 * phase2 + 60.0 * phase
        return profile, velocity, acceleration

    @classmethod
    def _numeric_quintic_segment(
        cls,
        *,
        time: float,
        start: float,
        duration: float,
        q0: float,
        q1: float,
    ) -> tuple[float, float, float]:
        """Return one known quintic segment and its first two derivatives."""

        if time <= start:
            return q0, 0.0, 0.0
        if time >= start + duration:
            return q1, 0.0, 0.0
        phase = (time - start) / duration
        profile, velocity, acceleration = cls._quintic_profile(phase)
        delta = q1 - q0
        return (
            q0 + delta * profile,
            delta * velocity / duration,
            delta * acceleration / (duration**2),
        )

    def _fixed_elevation_kinematics(
        self,
        *,
        time: float,
        right_arm_start: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the known elevation kinematics at one fixed time for one fixed second-arm start."""

        left_q, left_qdot, left_qddot = self._numeric_quintic_segment(
            time=time,
            start=0.0,
            duration=LEFT_ARM_ACTIVE_DURATION,
            q0=-np.pi,
            q1=0.0,
        )
        right_q, right_qdot, right_qddot = self._numeric_quintic_segment(
            time=time,
            start=right_arm_start,
            duration=RIGHT_ARM_ACTIVE_DURATION,
            q0=np.pi,
            q1=0.0,
        )
        return (
            np.array([left_q, right_q], dtype=float),
            np.array([left_qdot, right_qdot], dtype=float),
            np.array([left_qddot, right_qddot], dtype=float),
        )

    def _elevation_parameter_values(self, *, right_arm_start: float) -> np.ndarray:
        """Return the known elevation values injected into the root dynamics over the whole horizon."""

        values: list[float] = []
        for interval_index in range(self.interval_count):
            interval_start = float(self.node_times[interval_index])
            for stage_time in (
                interval_start,
                interval_start + 0.5 * self.shooting_step,
                interval_start + self.shooting_step,
            ):
                q, qdot, qddot = self._fixed_elevation_kinematics(
                    time=stage_time,
                    right_arm_start=right_arm_start,
                )
                values.extend(q.tolist())
                values.extend(qdot.tolist())
                values.extend(qddot.tolist())
        return np.asarray(values, dtype=float)

    def _symbolic_initial_root_state(self, variables: TwistOptimizationVariables) -> np.ndarray:
        """Return the initial root state."""

        q = np.zeros(self.model.nbQ(), dtype=float)
        qdot = np.zeros(self.model.nbQ(), dtype=float)
        q[6:] = (
            variables.left_plane_initial,
            -np.pi,
            variables.right_plane_initial,
            np.pi,
        )
        qdot[3] = self.configuration.somersault_rate
        qdot[:3] = -np.asarray(self.model.CoMdot(q, qdot, True).to_array(), dtype=float).reshape(3)
        return np.array(
            [
                q[0],
                q[1],
                q[2],
                q[3],
                q[4],
                q[5],
                qdot[0],
                qdot[1],
                qdot[2],
                qdot[3],
                qdot[4],
                qdot[5],
            ],
            dtype=float,
        )

    @staticmethod
    def _plane_state_vector(q_value: float) -> np.ndarray:
        """Return one `(q, qdot, qddot)` vector."""

        return np.array([q_value, 0.0, 0.0], dtype=float)

    @staticmethod
    def _advance_symbolic_constant_jerk(
        plane_state: ca.MX,
        jerk: ca.MX,
        duration: float,
    ) -> ca.MX:
        """Advance one plane state over a constant-jerk interval."""

        q, qdot, qddot = plane_state[0], plane_state[1], plane_state[2]
        return ca.vertcat(
            q + duration * qdot + 0.5 * duration * duration * qddot + duration**3 * jerk / 6.0,
            qdot + duration * qddot + 0.5 * duration * duration * jerk,
            qddot + duration * jerk,
        )

    def _symbolic_root_dynamics(
        self,
        root_state: ca.MX,
        left_plane_state: ca.MX,
        right_plane_state: ca.MX,
        elevation_q: ca.MX,
        elevation_qdot: ca.MX,
        elevation_qddot: ca.MX,
    ) -> ca.MX:
        """Return the time derivative of the root state."""

        q_root = root_state[:6]
        qdot_root = root_state[6:12]
        left_q, left_qdot, left_qddot = left_plane_state[0], left_plane_state[1], left_plane_state[2]
        right_q, right_qdot, right_qddot = right_plane_state[0], right_plane_state[1], right_plane_state[2]

        q_joint = ca.vertcat(left_q, elevation_q[0], right_q, elevation_q[1])
        qdot_joint = ca.vertcat(left_qdot, elevation_qdot[0], right_qdot, elevation_qdot[1])
        qddot_joint = ca.vertcat(left_qddot, elevation_qddot[0], right_qddot, elevation_qddot[1])

        q_full = ca.vertcat(q_root, q_joint)
        qdot_full = ca.vertcat(qdot_root, qdot_joint)
        qddot_root = self.symbolic_model.ForwardDynamicsFreeFloatingBase(
            q_full,
            qdot_full,
            qddot_joint,
        ).to_mx()
        return ca.vertcat(qdot_root, qddot_root)

    def _integrate_root_interval(
        self,
        root_state: ca.MX,
        left_plane_state: ca.MX,
        left_jerk: ca.MX,
        right_plane_state: ca.MX,
        right_jerk: ca.MX,
        elevation_start_q: ca.MX,
        elevation_start_qdot: ca.MX,
        elevation_start_qddot: ca.MX,
        elevation_mid_q: ca.MX,
        elevation_mid_qdot: ca.MX,
        elevation_mid_qddot: ca.MX,
        elevation_end_q: ca.MX,
        elevation_end_qdot: ca.MX,
        elevation_end_qddot: ca.MX,
    ) -> ca.MX:
        """Integrate one root shooting interval with RK4."""

        dt = self.shooting_step
        left_mid = self._advance_symbolic_constant_jerk(left_plane_state, left_jerk, 0.5 * dt)
        left_end = self._advance_symbolic_constant_jerk(left_plane_state, left_jerk, dt)
        right_mid = self._advance_symbolic_constant_jerk(right_plane_state, right_jerk, 0.5 * dt)
        right_end = self._advance_symbolic_constant_jerk(right_plane_state, right_jerk, dt)

        k1 = self._symbolic_root_dynamics(
            root_state,
            left_plane_state,
            right_plane_state,
            elevation_start_q,
            elevation_start_qdot,
            elevation_start_qddot,
        )
        k2 = self._symbolic_root_dynamics(
            root_state + 0.5 * dt * k1,
            left_mid,
            right_mid,
            elevation_mid_q,
            elevation_mid_qdot,
            elevation_mid_qddot,
        )
        k3 = self._symbolic_root_dynamics(
            root_state + 0.5 * dt * k2,
            left_mid,
            right_mid,
            elevation_mid_q,
            elevation_mid_qdot,
            elevation_mid_qddot,
        )
        k4 = self._symbolic_root_dynamics(
            root_state + dt * k3,
            left_end,
            right_end,
            elevation_end_q,
            elevation_end_qdot,
            elevation_end_qddot,
        )
        return root_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _build_interval_defect_function(self) -> ca.Function:
        """Build one interval defect function that CasADi can map over all shooting nodes."""

        root_state = ca.MX.sym("xk", ROOT_STATE_SIZE, 1)
        root_state_next = ca.MX.sym("xk_next", ROOT_STATE_SIZE, 1)
        left_plane_state = ca.MX.sym("xlk", PLANE_STATE_SIZE, 1)
        left_plane_state_next = ca.MX.sym("xlk_next", PLANE_STATE_SIZE, 1)
        right_plane_state = ca.MX.sym("xrk", PLANE_STATE_SIZE, 1)
        right_plane_state_next = ca.MX.sym("xrk_next", PLANE_STATE_SIZE, 1)
        left_jerk = ca.MX.sym("ulk")
        right_jerk = ca.MX.sym("urk")
        elevation_block = ca.MX.sym("pk", ELEVATION_STAGE_BLOCK_SIZE, 1)

        left_plane_prediction = self._advance_symbolic_constant_jerk(
            left_plane_state,
            left_jerk,
            self.shooting_step,
        )
        right_plane_prediction = self._advance_symbolic_constant_jerk(
            right_plane_state,
            right_jerk,
            self.shooting_step,
        )
        root_state_prediction = self._integrate_root_interval(
            root_state,
            left_plane_state,
            left_jerk,
            right_plane_state,
            right_jerk,
            elevation_block[0:2],
            elevation_block[2:4],
            elevation_block[4:6],
            elevation_block[6:8],
            elevation_block[8:10],
            elevation_block[10:12],
            elevation_block[12:14],
            elevation_block[14:16],
            elevation_block[16:18],
        )
        defect = ca.vertcat(
            root_state_next - root_state_prediction,
            left_plane_state_next - left_plane_prediction,
            right_plane_state_next - right_plane_prediction,
        )
        return ca.Function(
            "dms_interval_defect",
            [
                root_state,
                root_state_next,
                left_plane_state,
                left_plane_state_next,
                right_plane_state,
                right_plane_state_next,
                left_jerk,
                right_jerk,
                elevation_block,
            ],
            [defect],
        )

    def _initial_guess_motion(
        self,
        variables: TwistOptimizationVariables,
        *,
        right_arm_start: float | None = None,
    ) -> PiecewiseConstantJerkArmMotion:
        """Return a piecewise-constant-jerk motion used as a warm start for one fixed-start problem."""

        fixed_right_arm_start = variables.right_arm_start if right_arm_start is None else float(right_arm_start)
        left_fit = approximate_quintic_segment_with_piecewise_constant_jerk(
            total_time=LEFT_ARM_ACTIVE_DURATION,
            step=self.shooting_step,
            active_start=0.0,
            active_duration=LEFT_ARM_ACTIVE_DURATION,
            q0=variables.left_plane_initial,
            q1=variables.left_plane_final,
        )
        right_fit = approximate_quintic_segment_with_piecewise_constant_jerk(
            total_time=RIGHT_ARM_ACTIVE_DURATION,
            step=self.shooting_step,
            active_start=0.0,
            active_duration=RIGHT_ARM_ACTIVE_DURATION,
            q0=variables.right_plane_initial,
            q1=variables.right_plane_final,
        )
        left_plane = PiecewiseConstantJerkTrajectory(
            q0=variables.left_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=left_fit.jerks.copy(),
            active_start=0.0,
            active_end=LEFT_ARM_ACTIVE_DURATION,
            total_duration=self.configuration.final_time,
        )
        right_plane = PiecewiseConstantJerkTrajectory(
            q0=variables.right_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=right_fit.jerks.copy(),
            active_start=0.0,
            active_end=RIGHT_ARM_ACTIVE_DURATION,
            total_duration=self.configuration.final_time - fixed_right_arm_start,
        )
        return PiecewiseConstantJerkArmMotion(
            left_plane=left_plane,
            right_plane=right_plane,
            right_arm_start=fixed_right_arm_start,
        )

    def _initial_guess_root_state_history(
        self,
        variables: TwistOptimizationVariables,
        motion: PiecewiseConstantJerkArmMotion,
    ) -> np.ndarray:
        """Return a warm-start root-state history sampled on the shooting grid."""

        simulation = PredictiveAerialTwistSimulator(
            self.model_path,
            motion,
            configuration=SimulationConfiguration(
                final_time=self.configuration.final_time,
                steps=self.interval_count + 1,
                integrator="rk4",
                rk4_step=self.shooting_step,
                somersault_rate=self.configuration.somersault_rate,
            ),
            model=self.model,
        ).simulate()
        states = np.zeros((ROOT_STATE_SIZE, self.interval_count + 1), dtype=float)
        states[:6, :] = simulation.q[:, :6].T
        states[6:12, :] = simulation.qdot[:, :6].T
        return states

    @staticmethod
    def _plane_state_history(trajectory: PiecewiseConstantJerkTrajectory, sample_times: np.ndarray) -> np.ndarray:
        """Return `(q, qdot, qddot)` at arbitrary sample times for one plane trajectory."""

        return np.vstack(
            (
                np.asarray(trajectory.position(sample_times), dtype=float),
                np.asarray(trajectory.velocity(sample_times), dtype=float),
                np.asarray(trajectory.acceleration(sample_times), dtype=float),
            )
        )

    def candidate_start_times(self) -> np.ndarray:
        """Return the admissible discrete second-arm start times on the shooting grid."""

        first_node = int(round(RIGHT_ARM_SWEEP_BOUNDS[0] / self.shooting_step))
        last_node = int(round(RIGHT_ARM_SWEEP_BOUNDS[1] / self.shooting_step))
        return self.shooting_step * np.arange(first_node, last_node + 1, dtype=float)

    def _build_solver(
        self,
        *,
        max_iter: int,
        print_level: int,
        print_time: bool,
    ):
        """Build one global DMS solver and cache it."""

        options_key = (int(max_iter), int(print_level), bool(print_time))
        if self._solver is not None and self._solver_options_key == options_key:
            return self._solver

        parameters = ca.MX.sym("p", ELEVATION_STAGE_BLOCK_SIZE * self.interval_count, 1)
        root_state_symbols = [
            ca.MX.sym(f"X_root_{index}", ROOT_STATE_SIZE, 1) for index in range(self.interval_count + 1)
        ]
        left_plane_state_symbols = [
            ca.MX.sym(f"X_left_{index}", PLANE_STATE_SIZE, 1) for index in range(self.interval_count + 1)
        ]
        right_plane_state_symbols = [
            ca.MX.sym(f"X_right_{index}", PLANE_STATE_SIZE, 1) for index in range(self.interval_count + 1)
        ]
        left_control_symbols = ca.MX.sym("U_left", self.interval_count, 1)
        right_control_symbols = ca.MX.sym("U_right", self.interval_count, 1)

        objective = root_state_symbols[-1][5] / (2.0 * np.pi)
        objective += self.jerk_regularization * self.shooting_step * (
            ca.sumsqr(left_control_symbols) + ca.sumsqr(right_control_symbols)
        )
        interval_defect = self._build_interval_defect_function()
        parallelization = "openmp" if self.interval_count > 1 else "serial"
        try:
            mapped_interval_defect = interval_defect.map(
                self.interval_count,
                parallelization,
            )
        except RuntimeError:
            parallelization = "serial"
            mapped_interval_defect = interval_defect.map(self.interval_count, parallelization)
        self._interval_parallelization = parallelization

        mapped_defects = mapped_interval_defect(
            ca.horzcat(*root_state_symbols[:-1]),
            ca.horzcat(*root_state_symbols[1:]),
            ca.horzcat(*left_plane_state_symbols[:-1]),
            ca.horzcat(*left_plane_state_symbols[1:]),
            ca.horzcat(*right_plane_state_symbols[:-1]),
            ca.horzcat(*right_plane_state_symbols[1:]),
            ca.reshape(left_control_symbols, 1, self.interval_count),
            ca.reshape(right_control_symbols, 1, self.interval_count),
            ca.reshape(parameters, ELEVATION_STAGE_BLOCK_SIZE, self.interval_count),
        )[0]
        constraints = [ca.reshape(mapped_defects, -1, 1)]

        decision_vector = ca.vertcat(
            *root_state_symbols,
            *left_plane_state_symbols,
            *right_plane_state_symbols,
            left_control_symbols,
            right_control_symbols,
        )
        self._constraint_count = int(sum(component.shape[0] for component in constraints))
        self._solver = ca.nlpsol(
            "dms_solver",
            "ipopt",
            {
                "x": decision_vector,
                "p": parameters,
                "f": objective,
                "g": ca.vertcat(*constraints),
            },
            {
                "ipopt.max_iter": int(max_iter),
                "ipopt.print_level": int(print_level),
                "print_time": int(bool(print_time)),
            },
        )
        self._solver_options_key = options_key
        return self._solver

    def solve_fixed_start(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        right_arm_start: float,
        max_iter: int = 100,
        print_level: int = 0,
        print_time: bool = False,
    ) -> DirectMultipleShootingResult:
        """Solve one direct multiple-shooting problem with a fixed second-arm start time."""

        if not RIGHT_ARM_START_BOUNDS[0] <= right_arm_start <= RIGHT_ARM_START_BOUNDS[1]:
            raise ValueError("The fixed second-arm start time is outside the admissible bounds.")

        solver = self._build_solver(max_iter=max_iter, print_level=print_level, print_time=print_time)
        fixed_variables = TwistOptimizationVariables(
            right_arm_start=float(right_arm_start),
            left_plane_initial=initial_guess.left_plane_initial,
            left_plane_final=initial_guess.left_plane_final,
            right_plane_initial=initial_guess.right_plane_initial,
            right_plane_final=initial_guess.right_plane_final,
        )
        right_start_node_index = int(round(right_arm_start / self.shooting_step))
        right_terminal_node_index = right_start_node_index + self.active_control_count

        initial_motion = self._initial_guess_motion(fixed_variables, right_arm_start=right_arm_start)
        initial_root_states = self._initial_guess_root_state_history(fixed_variables, initial_motion)

        left_global_motion = PiecewiseConstantJerkTrajectory(
            q0=fixed_variables.left_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=np.concatenate(
                (
                    np.asarray(initial_motion.left_plane.jerks, dtype=float),
                    np.zeros(self.interval_count - self.active_control_count, dtype=float),
                )
            ),
            active_start=0.0,
            active_end=self.configuration.final_time,
            total_duration=self.configuration.final_time,
        )
        right_global_jerks = np.zeros(self.interval_count, dtype=float)
        right_global_jerks[
            right_start_node_index : right_start_node_index + self.active_control_count
        ] = np.asarray(initial_motion.right_plane.jerks, dtype=float)
        right_global_motion = PiecewiseConstantJerkTrajectory(
            q0=fixed_variables.right_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=right_global_jerks,
            active_start=0.0,
            active_end=self.configuration.final_time,
            total_duration=self.configuration.final_time,
        )
        initial_left_states = self._plane_state_history(left_global_motion, self.node_times)
        initial_right_states = self._plane_state_history(right_global_motion, self.node_times)

        initial_root_state = self._symbolic_initial_root_state(fixed_variables)
        initial_left_plane_state = self._plane_state_vector(fixed_variables.left_plane_initial)
        initial_right_plane_state = self._plane_state_vector(fixed_variables.right_plane_initial)
        terminal_left_plane_state = self._plane_state_vector(fixed_variables.left_plane_final)
        terminal_right_plane_state = self._plane_state_vector(fixed_variables.right_plane_final)

        x0: list[float] = []
        lbx: list[float] = []
        ubx: list[float] = []
        for state_index in range(self.interval_count + 1):
            root_guess = initial_root_states[:, state_index]
            x0.extend(root_guess.tolist())
            if state_index == 0:
                lbx.extend(initial_root_state.tolist())
                ubx.extend(initial_root_state.tolist())
            else:
                lbx.extend([-float("inf")] * ROOT_STATE_SIZE)
                ubx.extend([float("inf")] * ROOT_STATE_SIZE)

        for state_index in range(self.interval_count + 1):
            left_guess = initial_left_states[:, state_index]
            x0.extend(left_guess.tolist())
            if state_index == 0:
                lbx.extend(initial_left_plane_state.tolist())
                ubx.extend(initial_left_plane_state.tolist())
            elif state_index == self.active_control_count:
                lbx.extend(terminal_left_plane_state.tolist())
                ubx.extend(terminal_left_plane_state.tolist())
            else:
                lbx.extend([-float("inf")] * PLANE_STATE_SIZE)
                ubx.extend([float("inf")] * PLANE_STATE_SIZE)

        for state_index in range(self.interval_count + 1):
            right_guess = initial_right_states[:, state_index]
            x0.extend(right_guess.tolist())
            if state_index == 0:
                lbx.extend(initial_right_plane_state.tolist())
                ubx.extend(initial_right_plane_state.tolist())
            elif state_index == right_terminal_node_index:
                lbx.extend(terminal_right_plane_state.tolist())
                ubx.extend(terminal_right_plane_state.tolist())
            else:
                lbx.extend([-float("inf")] * PLANE_STATE_SIZE)
                ubx.extend([float("inf")] * PLANE_STATE_SIZE)

        for control_index in range(self.interval_count):
            jerk_guess = (
                initial_motion.left_plane.jerks[control_index]
                if control_index < self.active_control_count
                else 0.0
            )
            x0.append(float(jerk_guess))
            if control_index < self.active_control_count:
                lbx.append(-self.jerk_bound)
                ubx.append(self.jerk_bound)
            else:
                lbx.append(0.0)
                ubx.append(0.0)

        for control_index in range(self.interval_count):
            active = right_start_node_index <= control_index < right_terminal_node_index
            jerk_guess = (
                initial_motion.right_plane.jerks[control_index - right_start_node_index]
                if active
                else 0.0
            )
            x0.append(float(jerk_guess))
            if active:
                lbx.append(-self.jerk_bound)
                ubx.append(self.jerk_bound)
            else:
                lbx.append(0.0)
                ubx.append(0.0)

        solution = solver(
            x0=np.asarray(x0, dtype=float),
            lbx=np.asarray(lbx, dtype=float),
            ubx=np.asarray(ubx, dtype=float),
            lbg=np.zeros(self._constraint_count, dtype=float),
            ubg=np.zeros(self._constraint_count, dtype=float),
            p=self._elevation_parameter_values(right_arm_start=right_arm_start),
        )
        status = solver.stats()["return_status"]
        raw_solution = np.asarray(solution["x"].full(), dtype=float).reshape(-1)
        offset = 0
        root_state_values = raw_solution[offset : offset + ROOT_STATE_SIZE * (self.interval_count + 1)]
        offset += ROOT_STATE_SIZE * (self.interval_count + 1)
        left_plane_global_values = raw_solution[offset : offset + PLANE_STATE_SIZE * (self.interval_count + 1)]
        offset += PLANE_STATE_SIZE * (self.interval_count + 1)
        right_plane_global_values = raw_solution[offset : offset + PLANE_STATE_SIZE * (self.interval_count + 1)]
        offset += PLANE_STATE_SIZE * (self.interval_count + 1)
        left_control_global_values = raw_solution[offset : offset + self.interval_count]
        offset += self.interval_count
        right_control_global_values = raw_solution[offset : offset + self.interval_count]

        left_control_values = left_control_global_values[: self.active_control_count]
        right_control_values = right_control_global_values[
            right_start_node_index : right_start_node_index + self.active_control_count
        ]
        left_plane = PiecewiseConstantJerkTrajectory(
            q0=fixed_variables.left_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=left_control_values,
            active_start=0.0,
            active_end=LEFT_ARM_ACTIVE_DURATION,
            total_duration=self.configuration.final_time,
        )
        right_plane = PiecewiseConstantJerkTrajectory(
            q0=fixed_variables.right_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=right_control_values,
            active_start=0.0,
            active_end=RIGHT_ARM_ACTIVE_DURATION,
            total_duration=self.configuration.final_time - right_arm_start,
        )
        motion = PiecewiseConstantJerkArmMotion(
            left_plane=left_plane,
            right_plane=right_plane,
            right_arm_start=right_arm_start,
        )
        simulation = PredictiveAerialTwistSimulator(
            self.model_path,
            motion,
            configuration=self.configuration,
            model=self.model,
        ).simulate()

        normalized_status = status.lower().replace("_", " ")
        left_plane_global_states = left_plane_global_values.reshape(self.interval_count + 1, PLANE_STATE_SIZE).T
        right_plane_global_states = right_plane_global_values.reshape(self.interval_count + 1, PLANE_STATE_SIZE).T
        return DirectMultipleShootingResult(
            variables=fixed_variables,
            right_arm_start_node_index=right_start_node_index,
            left_plane_jerk=left_control_values.copy(),
            right_plane_jerk=right_control_values.copy(),
            node_times=self.node_times.copy(),
            arm_node_times=self.arm_node_times.copy(),
            root_state_nodes=root_state_values.reshape(self.interval_count + 1, ROOT_STATE_SIZE).T,
            left_plane_state_nodes=left_plane_global_states[:, : self.active_control_count + 1].copy(),
            right_plane_state_nodes=right_plane_global_states[
                :,
                right_start_node_index : right_terminal_node_index + 1,
            ].copy(),
            prescribed_motion=motion,
            simulation=simulation,
            objective=float(solution["f"]),
            solver_status=status,
            success=(
                "success" in normalized_status
                or "succeeded" in normalized_status
                or "solved" in normalized_status
            ),
        )

    def solve(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        max_iter: int = 100,
        print_level: int = 0,
        print_time: bool = False,
    ) -> DirectMultipleShootingSweepResult:
        """Solve one fixed-start OCP per admissible node and keep the best solution."""

        candidate_results = tuple(
            self.solve_fixed_start(
                initial_guess,
                right_arm_start=float(start_time),
                max_iter=max_iter,
                print_level=print_level,
                print_time=print_time,
            )
            for start_time in self.candidate_start_times()
        )
        successful_results = tuple(result for result in candidate_results if result.success)
        selection_pool = successful_results if successful_results else candidate_results
        best_result = min(selection_pool, key=lambda result: result.objective)
        return DirectMultipleShootingSweepResult(
            best_result=best_result,
            candidate_results=candidate_results,
        )


def create_dms_start_time_sweep_figure(
    *,
    start_times: np.ndarray,
    final_twist_turns: np.ndarray,
    objective_values: np.ndarray,
    success_mask: np.ndarray,
    best_start_time: float,
):
    """Create one simple figure summarizing the discrete sweep over the second-arm start time."""

    import matplotlib.pyplot as plt

    del objective_values
    times = np.asarray(start_times, dtype=float)
    twists = np.asarray(final_twist_turns, dtype=float)
    successes = np.asarray(success_mask, dtype=bool)
    best_index = int(np.argmin(np.abs(times - float(best_start_time))))

    figure, axis = plt.subplots(1, 1, figsize=(7.5, 4.5), tight_layout=True)
    axis.plot(times, twists, color="tab:blue", linewidth=1.6, marker="o", markersize=4.0)
    if np.any(~successes):
        axis.scatter(times[~successes], twists[~successes], color="tab:red", marker="x", s=36, label="Echec NLP")
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
    axis.set_title("Balayage DMS sur les noeuds de t1")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    return figure, axis


def show_dms_start_time_sweep_figure(
    *,
    start_times: np.ndarray,
    final_twist_turns: np.ndarray,
    objective_values: np.ndarray,
    success_mask: np.ndarray,
    best_start_time: float,
):
    """Open an external figure summarizing the discrete sweep over the second-arm start time."""

    import matplotlib.pyplot as plt

    figure, axis = create_dms_start_time_sweep_figure(
        start_times=start_times,
        final_twist_turns=final_twist_turns,
        objective_values=objective_values,
        success_mask=success_mask,
        best_start_time=best_start_time,
    )
    figure.canvas.draw_idle()
    plt.show(block=False)
    return figure, axis

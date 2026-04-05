"""Direct multiple-shooting optimization with a discrete sweep over the second-arm start node."""

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
    """Direct multiple-shooting twist optimizer with a discrete sweep over the second-arm start node."""

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
    def _clipped_phase(time: ca.MX, start: float | ca.MX, duration: float) -> ca.MX:
        """Return the symbolic motion phase clipped to `[0, 1]`."""

        return ca.fmin(1.0, ca.fmax(0.0, (time - start) / duration))

    def _symbolic_elevation_kinematics(
        self,
        time: ca.MX,
        right_arm_start: ca.MX,
    ) -> tuple[ca.MX, ca.MX, ca.MX]:
        """Return the fixed left/right elevation kinematics."""

        left_phase = self._clipped_phase(time, 0.0, LEFT_ARM_ACTIVE_DURATION)
        right_phase = self._clipped_phase(time, right_arm_start, RIGHT_ARM_ACTIVE_DURATION)
        left_profile, left_profile_dot, left_profile_ddot = self._quintic_profile(left_phase)
        right_profile, right_profile_dot, right_profile_ddot = self._quintic_profile(right_phase)

        left_q = -np.pi + np.pi * left_profile
        right_q = np.pi - np.pi * right_profile
        left_qdot = np.pi * left_profile_dot / LEFT_ARM_ACTIVE_DURATION
        right_qdot = -np.pi * right_profile_dot / RIGHT_ARM_ACTIVE_DURATION
        left_qddot = np.pi * left_profile_ddot / (LEFT_ARM_ACTIVE_DURATION**2)
        right_qddot = -np.pi * right_profile_ddot / (RIGHT_ARM_ACTIVE_DURATION**2)
        return (
            ca.vertcat(left_q, right_q),
            ca.vertcat(left_qdot, right_qdot),
            ca.vertcat(left_qddot, right_qddot),
        )

    def _symbolic_initial_root_state(self, variables: TwistOptimizationVariables) -> ca.DM:
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

        return ca.DM(
            np.array(
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
        )

    @staticmethod
    def _symbolic_initial_plane_state(q0: float) -> ca.DM:
        """Return the initial plane state `(q, qdot, qddot)`."""

        return ca.DM(np.array([q0, 0.0, 0.0], dtype=float))

    @staticmethod
    def _symbolic_terminal_plane_state(qf: float) -> ca.DM:
        """Return the desired terminal plane state `(q, qdot, qddot)`."""

        return ca.DM(np.array([qf, 0.0, 0.0], dtype=float))

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

    def _symbolic_interpolated_plane_state(
        self,
        plane_state_symbols: list[ca.MX],
        local_time: ca.MX,
    ) -> ca.MX:
        """Return a continuous piecewise-linear interpolation of one local arm-plane trajectory."""

        clipped_time = ca.fmin(LEFT_ARM_ACTIVE_DURATION, ca.fmax(0.0, local_time))
        interpolated_state = ca.MX.zeros(PLANE_STATE_SIZE, 1)
        for node_index, node_time in enumerate(self.arm_node_times):
            weight = ca.fmax(0.0, 1.0 - ca.fabs(clipped_time - float(node_time)) / self.shooting_step)
            interpolated_state += weight * plane_state_symbols[node_index]
        return interpolated_state

    def _symbolic_root_dynamics(
        self,
        time: ca.MX,
        root_state: ca.MX,
        left_plane_state: ca.MX,
        right_plane_state: ca.MX,
        right_arm_start: ca.MX,
    ) -> ca.MX:
        """Return the time derivative of the root state."""

        q_root = root_state[:6]
        qdot_root = root_state[6:12]
        left_q, left_qdot, left_qddot = left_plane_state[0], left_plane_state[1], left_plane_state[2]
        right_q, right_qdot, right_qddot = right_plane_state[0], right_plane_state[1], right_plane_state[2]

        elevation_q, elevation_qdot, elevation_qddot = self._symbolic_elevation_kinematics(time, right_arm_start)
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
        left_plane_state_symbols: list[ca.MX],
        right_plane_state_symbols: list[ca.MX],
        interval_start: float,
        right_arm_start: ca.MX,
    ) -> ca.MX:
        """Integrate one root shooting interval with RK4."""

        dt = self.shooting_step
        left_start = self._symbolic_interpolated_plane_state(left_plane_state_symbols, ca.MX(interval_start))
        left_mid = self._symbolic_interpolated_plane_state(left_plane_state_symbols, ca.MX(interval_start + 0.5 * dt))
        left_end = self._symbolic_interpolated_plane_state(left_plane_state_symbols, ca.MX(interval_start + dt))
        right_start = self._symbolic_interpolated_plane_state(
            right_plane_state_symbols,
            ca.MX(interval_start) - right_arm_start,
        )
        right_mid = self._symbolic_interpolated_plane_state(
            right_plane_state_symbols,
            ca.MX(interval_start + 0.5 * dt) - right_arm_start,
        )
        right_end = self._symbolic_interpolated_plane_state(
            right_plane_state_symbols,
            ca.MX(interval_start + dt) - right_arm_start,
        )

        k1 = self._symbolic_root_dynamics(interval_start, root_state, left_start, right_start, right_arm_start)
        k2 = self._symbolic_root_dynamics(
            interval_start + 0.5 * dt,
            root_state + 0.5 * dt * k1,
            left_mid,
            right_mid,
            right_arm_start,
        )
        k3 = self._symbolic_root_dynamics(
            interval_start + 0.5 * dt,
            root_state + 0.5 * dt * k2,
            left_mid,
            right_mid,
            right_arm_start,
        )
        k4 = self._symbolic_root_dynamics(
            interval_start + dt,
            root_state + dt * k3,
            left_end,
            right_end,
            right_arm_start,
        )
        return root_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

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
        """Build one fixed-start NLP solver and cache it for repeated start-time sweeps."""

        options_key = (int(max_iter), int(print_level), bool(print_time))
        if self._solver is not None and self._solver_options_key == options_key:
            return self._solver

        parameters = ca.MX.sym("parameters", 25, 1)
        right_arm_start_symbol = parameters[0]
        initial_root_state_symbol = parameters[1:13]
        initial_left_plane_state_symbol = parameters[13:16]
        initial_right_plane_state_symbol = parameters[16:19]
        terminal_left_plane_state_symbol = parameters[19:22]
        terminal_right_plane_state_symbol = parameters[22:25]
        root_state_symbols = [
            ca.MX.sym(f"X_root_{index}", ROOT_STATE_SIZE, 1) for index in range(self.interval_count + 1)
        ]
        left_plane_state_symbols = [
            ca.MX.sym(f"X_left_{index}", PLANE_STATE_SIZE, 1)
            for index in range(self.active_control_count + 1)
        ]
        right_plane_state_symbols = [
            ca.MX.sym(f"X_right_{index}", PLANE_STATE_SIZE, 1)
            for index in range(self.active_control_count + 1)
        ]
        left_control_symbols = ca.MX.sym("U_left", self.active_control_count, 1)
        right_control_symbols = ca.MX.sym("U_right", self.active_control_count, 1)

        constraints = [root_state_symbols[0] - initial_root_state_symbol]
        constraints.append(left_plane_state_symbols[0] - initial_left_plane_state_symbol)
        constraints.append(right_plane_state_symbols[0] - initial_right_plane_state_symbol)
        for index in range(self.active_control_count):
            constraints.append(
                left_plane_state_symbols[index + 1]
                - self._advance_symbolic_constant_jerk(
                    left_plane_state_symbols[index],
                    left_control_symbols[index],
                    self.shooting_step,
                )
            )
            constraints.append(
                right_plane_state_symbols[index + 1]
                - self._advance_symbolic_constant_jerk(
                    right_plane_state_symbols[index],
                    right_control_symbols[index],
                    self.shooting_step,
                )
            )

        objective = root_state_symbols[-1][5] / (2.0 * np.pi)
        objective += self.jerk_regularization * self.shooting_step * (
            ca.sumsqr(left_control_symbols) + ca.sumsqr(right_control_symbols)
        )
        for index in range(self.interval_count):
            constraints.append(
                root_state_symbols[index + 1]
                - self._integrate_root_interval(
                    root_state_symbols[index],
                    left_plane_state_symbols,
                    right_plane_state_symbols,
                    float(self.node_times[index]),
                    right_arm_start_symbol,
                )
            )

        constraints.append(left_plane_state_symbols[-1] - terminal_left_plane_state_symbol)
        constraints.append(right_plane_state_symbols[-1] - terminal_right_plane_state_symbol)

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
        initial_motion = self._initial_guess_motion(fixed_variables, right_arm_start=right_arm_start)
        initial_root_states = self._initial_guess_root_state_history(fixed_variables, initial_motion)
        initial_left_states = self._plane_state_history(initial_motion.left_plane, self.arm_node_times)
        initial_right_states = self._plane_state_history(initial_motion.right_plane, self.arm_node_times)
        parameter_values = np.concatenate(
            (
                np.asarray([right_arm_start], dtype=float),
                np.asarray(self._symbolic_initial_root_state(fixed_variables), dtype=float).reshape(-1),
                np.asarray(self._symbolic_initial_plane_state(fixed_variables.left_plane_initial), dtype=float).reshape(-1),
                np.asarray(self._symbolic_initial_plane_state(fixed_variables.right_plane_initial), dtype=float).reshape(-1),
                np.asarray(self._symbolic_terminal_plane_state(fixed_variables.left_plane_final), dtype=float).reshape(-1),
                np.asarray(self._symbolic_terminal_plane_state(fixed_variables.right_plane_final), dtype=float).reshape(-1),
            )
        )

        x0: list[float] = []
        lbx: list[float] = []
        ubx: list[float] = []
        for state_index in range(self.interval_count + 1):
            x0.extend(initial_root_states[:, state_index].tolist())
            lbx.extend([-float("inf")] * ROOT_STATE_SIZE)
            ubx.extend([float("inf")] * ROOT_STATE_SIZE)
        for state_index in range(self.active_control_count + 1):
            x0.extend(initial_left_states[:, state_index].tolist())
            lbx.extend([-float("inf")] * PLANE_STATE_SIZE)
            ubx.extend([float("inf")] * PLANE_STATE_SIZE)
        for state_index in range(self.active_control_count + 1):
            x0.extend(initial_right_states[:, state_index].tolist())
            lbx.extend([-float("inf")] * PLANE_STATE_SIZE)
            ubx.extend([float("inf")] * PLANE_STATE_SIZE)
        x0.extend(initial_motion.left_plane.jerks.tolist())
        lbx.extend([-self.jerk_bound] * self.active_control_count)
        ubx.extend([self.jerk_bound] * self.active_control_count)
        x0.extend(initial_motion.right_plane.jerks.tolist())
        lbx.extend([-self.jerk_bound] * self.active_control_count)
        ubx.extend([self.jerk_bound] * self.active_control_count)

        solution = solver(
            x0=np.asarray(x0, dtype=float),
            lbx=np.asarray(lbx, dtype=float),
            ubx=np.asarray(ubx, dtype=float),
            lbg=np.zeros(self._constraint_count, dtype=float),
            ubg=np.zeros(self._constraint_count, dtype=float),
            p=parameter_values,
        )
        status = solver.stats()["return_status"]
        raw_solution = np.asarray(solution["x"].full(), dtype=float).reshape(-1)
        offset = 0
        root_state_values = raw_solution[offset : offset + ROOT_STATE_SIZE * (self.interval_count + 1)]
        offset += ROOT_STATE_SIZE * (self.interval_count + 1)
        left_plane_state_values = raw_solution[offset : offset + PLANE_STATE_SIZE * (self.active_control_count + 1)]
        offset += PLANE_STATE_SIZE * (self.active_control_count + 1)
        right_plane_state_values = raw_solution[
            offset : offset + PLANE_STATE_SIZE * (self.active_control_count + 1)
        ]
        offset += PLANE_STATE_SIZE * (self.active_control_count + 1)
        left_control_values = raw_solution[offset : offset + self.active_control_count]
        offset += self.active_control_count
        right_control_values = raw_solution[offset : offset + self.active_control_count]

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
        return DirectMultipleShootingResult(
            variables=fixed_variables,
            right_arm_start_node_index=int(round(right_arm_start / self.shooting_step)),
            left_plane_jerk=left_control_values.copy(),
            right_plane_jerk=right_control_values.copy(),
            node_times=self.node_times.copy(),
            arm_node_times=self.arm_node_times.copy(),
            root_state_nodes=root_state_values.reshape(self.interval_count + 1, ROOT_STATE_SIZE).T,
            left_plane_state_nodes=left_plane_state_values.reshape(
                self.active_control_count + 1,
                PLANE_STATE_SIZE,
            ).T,
            right_plane_state_nodes=right_plane_state_values.reshape(
                self.active_control_count + 1,
                PLANE_STATE_SIZE,
            ).T,
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
    """Create a figure summarizing the discrete sweep over the second-arm start time."""

    import matplotlib.pyplot as plt

    times = np.asarray(start_times, dtype=float)
    twists = np.asarray(final_twist_turns, dtype=float)
    objectives = np.asarray(objective_values, dtype=float)
    successes = np.asarray(success_mask, dtype=bool)
    best_index = int(np.argmin(np.abs(times - float(best_start_time))))

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 6.5), tight_layout=True)
    axes[0].plot(times, twists, color="tab:blue", linewidth=1.6, marker="o", markersize=4.0)
    axes[1].plot(times, objectives, color="tab:green", linewidth=1.6, marker="o", markersize=4.0)
    if np.any(~successes):
        axes[0].scatter(times[~successes], twists[~successes], color="tab:red", marker="x", s=36, label="Echec NLP")
        axes[1].scatter(times[~successes], objectives[~successes], color="tab:red", marker="x", s=36)
    axes[0].scatter(
        [times[best_index]],
        [twists[best_index]],
        color="tab:orange",
        s=64,
        zorder=3,
        label="Meilleure solution",
    )
    axes[1].scatter(
        [times[best_index]],
        [objectives[best_index]],
        color="tab:orange",
        s=64,
        zorder=3,
    )
    axes[0].set_ylabel("Vrille finale (tours)")
    axes[1].set_ylabel("Objectif")
    axes[1].set_xlabel("Debut bras droit t1 (s)")
    axes[0].set_title("Balayage DMS sur les noeuds de t1")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    return figure, axes


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

    figure, axes = create_dms_start_time_sweep_figure(
        start_times=start_times,
        final_twist_turns=final_twist_turns,
        objective_values=objective_values,
        success_mask=success_mask,
        best_start_time=best_start_time,
    )
    figure.canvas.draw_idle()
    plt.show(block=False)
    return figure, axes

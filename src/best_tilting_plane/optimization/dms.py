"""Direct multiple-shooting optimization with piecewise-constant jerk controls."""

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
PLANE_STATE_SIZE = 3
ARM_STATE_SIZE = 2 * PLANE_STATE_SIZE
ROOT_STATE_SIZE = 12


@dataclass(frozen=True)
class DirectMultipleShootingResult:
    """Optimization result for the jerk-controlled direct multiple-shooting problem."""

    variables: TwistOptimizationVariables
    left_plane_jerk: np.ndarray
    right_plane_jerk: np.ndarray
    node_times: np.ndarray
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


class DirectMultipleShootingOptimizer:
    """Direct multiple-shooting twist optimizer with jerk controls on both arm planes."""

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
        elevation_fit = approximate_first_arm_elevation_motion(
            total_time=self.configuration.final_time,
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
    def _clipped_phase(time: ca.MX, start: ca.MX, duration: float) -> ca.MX:
        """Return the symbolic motion phase clipped to `[0, 1]`."""

        return ca.fmin(1.0, ca.fmax(0.0, (time - start) / duration))

    def _symbolic_elevation_kinematics(self, time: ca.MX, t1: ca.MX) -> tuple[ca.MX, ca.MX, ca.MX]:
        """Return the fixed left/right elevation kinematics."""

        left_phase = self._clipped_phase(time, 0.0, LEFT_ARM_ACTIVE_DURATION)
        right_phase = self._clipped_phase(time, t1, RIGHT_ARM_ACTIVE_DURATION)
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
    def _advance_symbolic_constant_acceleration(
        q: ca.MX,
        qdot: ca.MX,
        qddot: ca.MX,
        duration: ca.MX,
    ) -> tuple[ca.MX, ca.MX, ca.MX]:
        """Advance one scalar state over a zero-jerk symbolic interval."""

        return (
            q + duration * qdot + 0.5 * duration * duration * qddot,
            qdot + duration * qddot,
            qddot,
        )

    @staticmethod
    def _advance_symbolic_constant_jerk(
        q: ca.MX,
        qdot: ca.MX,
        qddot: ca.MX,
        jerk: ca.MX,
        duration: ca.MX,
    ) -> tuple[ca.MX, ca.MX, ca.MX]:
        """Advance one scalar state over a constant-jerk symbolic interval."""

        return (
            q + duration * qdot + 0.5 * duration * duration * qddot + duration * duration * duration * jerk / 6.0,
            qdot + duration * qddot + 0.5 * duration * duration * jerk,
            qddot + duration * jerk,
        )

    def _symbolic_local_plane_state(
        self,
        controls: ca.MX,
        local_time: ca.MX,
        q0: float,
    ) -> ca.MX:
        """Evaluate a plane state driven by active jerks defined on the local `[0, 0.3]` window."""

        clipped_time = ca.fmin(self.configuration.final_time, ca.fmax(0.0, local_time))
        q = ca.MX(q0)
        qdot = ca.MX(0.0)
        qddot = ca.MX(0.0)

        for interval_index in range(self.active_control_count):
            interval_start = interval_index * self.shooting_step
            interval_end = interval_start + self.shooting_step
            duration = ca.fmax(0.0, ca.fmin(clipped_time, interval_end) - interval_start)
            q, qdot, qddot = self._advance_symbolic_constant_jerk(
                q,
                qdot,
                qddot,
                controls[interval_index],
                duration,
            )

        passive_duration = ca.fmax(0.0, clipped_time - LEFT_ARM_ACTIVE_DURATION)
        q, qdot, qddot = self._advance_symbolic_constant_acceleration(
            q,
            qdot,
            qddot,
            passive_duration,
        )
        return ca.vertcat(q, qdot, qddot)

    def _symbolic_root_dynamics(
        self,
        time: ca.MX,
        root_state: ca.MX,
        left_plane_state: ca.MX,
        right_plane_state: ca.MX,
        t1: ca.MX,
    ) -> ca.MX:
        """Return the time derivative of the root state."""

        q_root = root_state[:6]
        qdot_root = root_state[6:12]
        left_q, left_qdot, left_qddot = left_plane_state[0], left_plane_state[1], left_plane_state[2]
        right_q, right_qdot, right_qddot = (
            right_plane_state[0],
            right_plane_state[1],
            right_plane_state[2],
        )

        elevation_q, elevation_qdot, elevation_qddot = self._symbolic_elevation_kinematics(time, t1)
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
        left_controls: ca.MX,
        right_controls: ca.MX,
        interval_start: float,
        t1: ca.MX,
        left_q0: float,
        right_q0: float,
    ) -> ca.MX:
        """Integrate one root shooting interval with RK4 using deterministic arm states."""

        dt = self.shooting_step
        left_start = self._symbolic_local_plane_state(
            left_controls,
            ca.MX(interval_start),
            left_q0,
        )
        right_start = self._symbolic_local_plane_state(
            right_controls,
            ca.MX(interval_start) - t1,
            right_q0,
        )
        left_mid = self._symbolic_local_plane_state(
            left_controls,
            ca.MX(interval_start + 0.5 * dt),
            left_q0,
        )
        right_mid = self._symbolic_local_plane_state(
            right_controls,
            ca.MX(interval_start + 0.5 * dt) - t1,
            right_q0,
        )
        left_end = self._symbolic_local_plane_state(
            left_controls,
            ca.MX(interval_start + dt),
            left_q0,
        )
        right_end = self._symbolic_local_plane_state(
            right_controls,
            ca.MX(interval_start + dt) - t1,
            right_q0,
        )
        k1 = self._symbolic_root_dynamics(interval_start, root_state, left_start, right_start, t1)
        k2 = self._symbolic_root_dynamics(
            interval_start + 0.5 * dt,
            root_state + 0.5 * dt * k1,
            left_mid,
            right_mid,
            t1,
        )
        k3 = self._symbolic_root_dynamics(
            interval_start + 0.5 * dt,
            root_state + 0.5 * dt * k2,
            left_mid,
            right_mid,
            t1,
        )
        k4 = self._symbolic_root_dynamics(
            interval_start + dt,
            root_state + dt * k3,
            left_end,
            right_end,
            t1,
        )
        return root_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _initial_guess_motion(self, variables: TwistOptimizationVariables) -> PiecewiseConstantJerkArmMotion:
        """Return a piecewise-constant-jerk motion used as a warm start for the DMS problem."""

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
            total_duration=self.configuration.final_time - variables.right_arm_start,
        )
        return PiecewiseConstantJerkArmMotion(
            left_plane=left_plane,
            right_plane=right_plane,
            right_arm_start=variables.right_arm_start,
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
    def _plane_state_history(
        trajectory: PiecewiseConstantJerkTrajectory,
        sample_times: np.ndarray,
    ) -> np.ndarray:
        """Return `(q, qdot, qddot)` at arbitrary sample times for one plane trajectory."""

        return np.vstack(
            (
                np.asarray(trajectory.position(sample_times), dtype=float),
                np.asarray(trajectory.velocity(sample_times), dtype=float),
                np.asarray(trajectory.acceleration(sample_times), dtype=float),
            )
        )

    def solve(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        max_iter: int = 100,
        print_level: int = 0,
        print_time: bool = False,
    ) -> DirectMultipleShootingResult:
        """Solve the direct multiple-shooting problem."""

        t1_symbol = ca.MX.sym("t1")
        root_state_symbols = [
            ca.MX.sym(f"X_{index}", ROOT_STATE_SIZE, 1) for index in range(self.interval_count + 1)
        ]
        left_control_symbols = ca.MX.sym("U_left", self.active_control_count, 1)
        right_control_symbols = ca.MX.sym("U_right", self.active_control_count, 1)

        constraints = [root_state_symbols[0] - self._symbolic_initial_root_state(initial_guess)]
        objective = root_state_symbols[-1][5] / (2.0 * np.pi)
        objective += self.jerk_regularization * self.shooting_step * (
            ca.sumsqr(left_control_symbols) + ca.sumsqr(right_control_symbols)
        )
        for index in range(self.interval_count):
            next_state = self._integrate_root_interval(
                root_state_symbols[index],
                left_control_symbols,
                right_control_symbols,
                float(self.node_times[index]),
                t1_symbol,
                initial_guess.left_plane_initial,
                initial_guess.right_plane_initial,
            )
            constraints.append(root_state_symbols[index + 1] - next_state)

        left_final_state = self._symbolic_local_plane_state(
            left_control_symbols,
            ca.MX(self.configuration.final_time),
            initial_guess.left_plane_initial,
        )
        right_final_state = self._symbolic_local_plane_state(
            right_control_symbols,
            ca.MX(self.configuration.final_time) - t1_symbol,
            initial_guess.right_plane_initial,
        )
        constraints.append(
            ca.vertcat(left_final_state, right_final_state)
            - ca.vertcat(
                initial_guess.left_plane_final,
                0.0,
                0.0,
                initial_guess.right_plane_final,
                0.0,
                0.0,
            )
        )

        decision_vector = ca.vertcat(
            t1_symbol,
            *root_state_symbols,
            left_control_symbols,
            right_control_symbols,
        )
        solver = ca.nlpsol(
            "dms_solver",
            "ipopt",
            {"x": decision_vector, "f": objective, "g": ca.vertcat(*constraints)},
            {
                "ipopt.max_iter": int(max_iter),
                "ipopt.print_level": int(print_level),
                "print_time": int(bool(print_time)),
            },
        )

        initial_motion = self._initial_guess_motion(initial_guess)
        initial_states = self._initial_guess_root_state_history(initial_guess, initial_motion)
        x0 = [float(initial_guess.right_arm_start)]
        lbx = [RIGHT_ARM_START_BOUNDS[0]]
        ubx = [RIGHT_ARM_START_BOUNDS[1]]
        for state_index in range(self.interval_count + 1):
            x0.extend(initial_states[:, state_index].tolist())
            lbx.extend([-float("inf")] * ROOT_STATE_SIZE)
            ubx.extend([float("inf")] * ROOT_STATE_SIZE)
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
            lbg=np.zeros(sum(component.shape[0] for component in constraints), dtype=float),
            ubg=np.zeros(sum(component.shape[0] for component in constraints), dtype=float),
        )
        status = solver.stats()["return_status"]
        raw_solution = np.asarray(solution["x"].full(), dtype=float).reshape(-1)
        offset = 0
        right_arm_start = float(raw_solution[offset])
        offset += 1
        root_state_values = raw_solution[offset : offset + ROOT_STATE_SIZE * (self.interval_count + 1)]
        offset += ROOT_STATE_SIZE * (self.interval_count + 1)
        left_control_values = raw_solution[offset : offset + self.active_control_count]
        offset += self.active_control_count
        right_control_values = raw_solution[offset : offset + self.active_control_count]

        left_plane = PiecewiseConstantJerkTrajectory(
            q0=initial_guess.left_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=left_control_values,
            active_start=0.0,
            active_end=LEFT_ARM_ACTIVE_DURATION,
            total_duration=self.configuration.final_time,
        )
        right_plane = PiecewiseConstantJerkTrajectory(
            q0=initial_guess.right_plane_initial,
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
        root_state_nodes = root_state_values.reshape(self.interval_count + 1, ROOT_STATE_SIZE).T
        left_plane_state_nodes = self._plane_state_history(left_plane, self.node_times)
        right_plane_state_nodes = self._plane_state_history(
            right_plane,
            self.node_times - right_arm_start,
        )
        simulation = PredictiveAerialTwistSimulator(
            self.model_path,
            motion,
            configuration=self.configuration,
            model=self.model,
        ).simulate()

        normalized_status = status.lower().replace("_", " ")
        return DirectMultipleShootingResult(
            variables=TwistOptimizationVariables(
                right_arm_start=right_arm_start,
                left_plane_initial=initial_guess.left_plane_initial,
                left_plane_final=initial_guess.left_plane_final,
                right_plane_initial=initial_guess.right_plane_initial,
                right_plane_final=initial_guess.right_plane_final,
            ),
            left_plane_jerk=left_control_values.copy(),
            right_plane_jerk=right_control_values.copy(),
            node_times=self.node_times.copy(),
            root_state_nodes=root_state_nodes,
            left_plane_state_nodes=left_plane_state_nodes,
            right_plane_state_nodes=right_plane_state_nodes,
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

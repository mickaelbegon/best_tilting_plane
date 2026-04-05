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
ROOT_STATE_SIZE = 12
FULL_STATE_SIZE = ROOT_STATE_SIZE + 2 * PLANE_STATE_SIZE


@dataclass(frozen=True)
class DirectMultipleShootingResult:
    """Optimization result for the jerk-controlled direct multiple-shooting problem."""

    variables: TwistOptimizationVariables
    left_plane_jerk: np.ndarray
    right_plane_jerk: np.ndarray
    node_times: np.ndarray
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
        self.interval_count = step_count
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

    def _symbolic_initial_state(self, variables: TwistOptimizationVariables) -> ca.DM:
        """Return the combined root-and-plane initial state."""

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
                    variables.left_plane_initial,
                    0.0,
                    0.0,
                    variables.right_plane_initial,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        )

    def _symbolic_dynamics(self, time: ca.MX, state: ca.MX, control: ca.MX, t1: ca.MX) -> ca.MX:
        """Return the time derivative of the root and plane states."""

        q_root = state[:6]
        qdot_root = state[6:12]
        left_q, left_qdot, left_qddot = state[12], state[13], state[14]
        right_q, right_qdot, right_qddot = state[15], state[16], state[17]

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

        left_jerk = ca.if_else(time < LEFT_ARM_ACTIVE_DURATION, control[0], 0.0)
        right_jerk = ca.if_else(
            ca.logic_and(time >= t1, time < t1 + RIGHT_ARM_ACTIVE_DURATION),
            control[1],
            0.0,
        )
        return ca.vertcat(
            qdot_root,
            qddot_root,
            left_qdot,
            left_qddot,
            left_jerk,
            right_qdot,
            right_qddot,
            right_jerk,
        )

    def _integrate_interval(self, state: ca.MX, control: ca.MX, interval_start: float, t1: ca.MX) -> ca.MX:
        """Integrate one shooting interval with RK4."""

        dt = self.shooting_step
        k1 = self._symbolic_dynamics(interval_start, state, control, t1)
        k2 = self._symbolic_dynamics(interval_start + 0.5 * dt, state + 0.5 * dt * k1, control, t1)
        k3 = self._symbolic_dynamics(interval_start + 0.5 * dt, state + 0.5 * dt * k2, control, t1)
        k4 = self._symbolic_dynamics(interval_start + dt, state + dt * k3, control, t1)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _initial_guess_motion(self, variables: TwistOptimizationVariables) -> PiecewiseConstantJerkArmMotion:
        """Return a piecewise-constant-jerk motion used as a warm start for the DMS problem."""

        left_plane = approximate_quintic_segment_with_piecewise_constant_jerk(
            total_time=self.configuration.final_time,
            step=self.shooting_step,
            active_start=0.0,
            active_duration=LEFT_ARM_ACTIVE_DURATION,
            q0=variables.left_plane_initial,
            q1=variables.left_plane_final,
        )
        right_plane = approximate_quintic_segment_with_piecewise_constant_jerk(
            total_time=self.configuration.final_time,
            step=self.shooting_step,
            active_start=variables.right_arm_start,
            active_duration=RIGHT_ARM_ACTIVE_DURATION,
            q0=variables.right_plane_initial,
            q1=variables.right_plane_final,
        )
        return PiecewiseConstantJerkArmMotion(
            left_plane=left_plane,
            right_plane=right_plane,
            right_arm_start=variables.right_arm_start,
        )

    def _initial_guess_state_history(
        self,
        variables: TwistOptimizationVariables,
        motion: PiecewiseConstantJerkArmMotion,
    ) -> np.ndarray:
        """Return a warm-start state history sampled on the shooting grid."""

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
        states = np.zeros((FULL_STATE_SIZE, self.interval_count + 1), dtype=float)
        states[:6, :] = simulation.q[:, :6].T
        states[6:12, :] = simulation.qdot[:, :6].T
        for index, time in enumerate(self.node_times):
            left = motion.left(float(time))
            right = motion.right(float(time))
            states[12:15, index] = (
                left.elevation_plane.position,
                left.elevation_plane.velocity,
                left.elevation_plane.acceleration,
            )
            states[15:18, index] = (
                right.elevation_plane.position,
                right.elevation_plane.velocity,
                right.elevation_plane.acceleration,
            )
        return states

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
        state_symbols = [
            ca.MX.sym(f"X_{index}", FULL_STATE_SIZE, 1) for index in range(self.interval_count + 1)
        ]
        control_symbols = [
            ca.MX.sym(f"U_{index}", 2, 1) for index in range(self.interval_count)
        ]

        constraints = [state_symbols[0] - self._symbolic_initial_state(initial_guess)]
        objective = state_symbols[-1][5] / (2.0 * np.pi)
        for index, control in enumerate(control_symbols):
            objective += self.jerk_regularization * self.shooting_step * ca.sumsqr(control)
            next_state = self._integrate_interval(
                state_symbols[index],
                control,
                float(self.node_times[index]),
                t1_symbol,
            )
            constraints.append(state_symbols[index + 1] - next_state)

        constraints.append(
            state_symbols[-1][12:18]
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
            *state_symbols,
            *control_symbols,
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
        initial_states = self._initial_guess_state_history(initial_guess, initial_motion)
        x0 = [float(initial_guess.right_arm_start)]
        lbx = [RIGHT_ARM_START_BOUNDS[0]]
        ubx = [RIGHT_ARM_START_BOUNDS[1]]
        for state_index in range(self.interval_count + 1):
            x0.extend(initial_states[:, state_index].tolist())
            lbx.extend([-float("inf")] * FULL_STATE_SIZE)
            ubx.extend([float("inf")] * FULL_STATE_SIZE)
        for jerk_index in range(self.interval_count):
            x0.extend(
                [
                    float(initial_motion.left_plane.jerks[jerk_index]),
                    float(initial_motion.right_plane.jerks[jerk_index]),
                ]
            )
            lbx.extend([-self.jerk_bound, -self.jerk_bound])
            ubx.extend([self.jerk_bound, self.jerk_bound])

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
        state_values = raw_solution[offset : offset + FULL_STATE_SIZE * (self.interval_count + 1)]
        offset += FULL_STATE_SIZE * (self.interval_count + 1)
        control_values = raw_solution[offset:].reshape(self.interval_count, 2)

        left_plane = PiecewiseConstantJerkTrajectory(
            q0=initial_guess.left_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=control_values[:, 0],
            active_start=0.0,
            active_end=LEFT_ARM_ACTIVE_DURATION,
        )
        right_plane = PiecewiseConstantJerkTrajectory(
            q0=initial_guess.right_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=control_values[:, 1],
            active_start=right_arm_start,
            active_end=right_arm_start + RIGHT_ARM_ACTIVE_DURATION,
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
            variables=TwistOptimizationVariables(
                right_arm_start=right_arm_start,
                left_plane_initial=initial_guess.left_plane_initial,
                left_plane_final=initial_guess.left_plane_final,
                right_plane_initial=initial_guess.right_plane_initial,
                right_plane_final=initial_guess.right_plane_final,
            ),
            left_plane_jerk=control_values[:, 0].copy(),
            right_plane_jerk=control_values[:, 1].copy(),
            node_times=self.node_times.copy(),
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

"""Predictive zero-gravity simulation based on a reduced floating-base `biorbd` model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import biorbd
import numpy as np
from scipy.integrate import solve_ivp

from best_tilting_plane.modeling import ReducedAerialBiomod, root_twist_index
from best_tilting_plane.simulation.arm_motion import TwistOptimizationVariables
from best_tilting_plane.simulation.jerk_motion import (
    PiecewiseConstantJerkArmMotion,
    build_piecewise_constant_jerk_arm_motion,
)

ROOT_DOF = 6
JOINT_DOF = 4
FULL_TWIST_INDEX = 3 + root_twist_index()


@dataclass(frozen=True)
class SimulationConfiguration:
    """Numerical parameters used by the predictive simulation."""

    final_time: float = 1.0
    steps: int = 201
    somersault_rate: float = 2.0 * np.pi
    contact_twist_rate: float = 0.0
    rtol: float = 1e-8
    atol: float = 1e-10
    integrator: str = "rk45"
    rk4_step: float | None = None
    rk4_candidate_steps: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025)
    rk4_state_tolerance: float = 1e-4
    rk4_twist_tolerance: float = 1e-4


@dataclass(frozen=True)
class IntegratorSelection:
    """Summary of the integrator selection and benchmark outcome."""

    method: str
    rk4_step: float | None
    elapsed_seconds: float
    max_state_error: float | None = None
    final_twist_error: float | None = None
    reference_elapsed_seconds: float | None = None


@dataclass(frozen=True)
class AerialSimulationResult:
    """Time histories returned by the predictive simulation."""

    time: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray
    integrator_method: str = "rk45"
    rk4_step: float | None = None
    integration_seconds: float | None = None

    @property
    def final_twist_angle(self) -> float:
        """Return the final root twist angle in radians."""

        return float(self.q[-1, FULL_TWIST_INDEX])

    @property
    def final_twist_turns(self) -> float:
        """Return the final root twist angle expressed in turns."""

        return round(self.final_twist_angle / (2.0 * np.pi), 2)


class PredictiveAerialTwistSimulator:
    """Simulate aerial twisting with prescribed arm kinematics and a floating base."""

    def __init__(
        self,
        model_path: str | Path,
        prescribed_motion: PiecewiseConstantJerkArmMotion,
        *,
        configuration: SimulationConfiguration | None = None,
        model: biorbd.Model | None = None,
    ) -> None:
        """Load the `biorbd` model and store the simulation settings."""

        self.model_path = str(model_path)
        self.model = model if model is not None else biorbd.Model(self.model_path)
        self.prescribed_motion = prescribed_motion
        self.configuration = configuration or SimulationConfiguration()
        self._integrator_selection: IntegratorSelection | None = None

        if self.model.nbRoot() != ROOT_DOF:
            raise ValueError(f"The reduced aerial model must expose {ROOT_DOF} root DoFs.")
        if self.model.nbQ() - self.model.nbRoot() != JOINT_DOF:
            raise ValueError(f"The reduced aerial model must expose {JOINT_DOF} non-root DoFs.")

    @classmethod
    def from_builder(
        cls,
        output_path: str | Path,
        variables: TwistOptimizationVariables,
        *,
        model_builder: ReducedAerialBiomod | None = None,
        configuration: SimulationConfiguration | None = None,
    ) -> "PredictiveAerialTwistSimulator":
        """Build the model file, then instantiate the simulator."""

        builder = model_builder or ReducedAerialBiomod()
        model_path = builder.write(output_path)
        jerk_step = 0.02
        return cls(
            model_path,
            build_piecewise_constant_jerk_arm_motion(
                variables,
                total_time=(configuration or SimulationConfiguration()).final_time,
                step=jerk_step,
                first_arm_start=float(getattr(variables, "first_arm_start", 0.0)),
            ),
            configuration=configuration,
        )

    def joint_kinematics(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return non-root joint position, velocity, and acceleration at the requested time."""

        left = self.prescribed_motion.left(time)
        right = self.prescribed_motion.right(time)
        q = np.array(
            [
                left.elevation_plane.position,
                left.elevation.position,
                right.elevation_plane.position,
                right.elevation.position,
            ],
            dtype=float,
        )
        qdot = np.array(
            [
                left.elevation_plane.velocity,
                left.elevation.velocity,
                right.elevation_plane.velocity,
                right.elevation.velocity,
            ],
            dtype=float,
        )
        qddot = np.array(
            [
                left.elevation_plane.acceleration,
                left.elevation.acceleration,
                right.elevation_plane.acceleration,
                right.elevation.acceleration,
            ],
            dtype=float,
        )
        return q, qdot, qddot

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the initial full state and correct root translations so that `v_COM = 0`."""

        q_joint, qdot_joint, _ = self.joint_kinematics(0.0)
        q = np.zeros(self.model.nbQ())
        qdot = np.zeros(self.model.nbQ())
        q[ROOT_DOF:] = q_joint
        qdot[3] = self.configuration.somersault_rate
        qdot[FULL_TWIST_INDEX] = self.configuration.contact_twist_rate
        qdot[ROOT_DOF:] = qdot_joint
        qdot[:3] = self._translation_velocity_cancelling_com_motion(q, qdot)
        return q, qdot

    def simulate(self) -> AerialSimulationResult:
        """Run the predictive simulation with the configured integrator."""

        q0, qdot0 = self.initial_state()
        initial_state = np.concatenate((q0[:ROOT_DOF], qdot0[:ROOT_DOF]))
        t_eval = np.linspace(0.0, self.configuration.final_time, self.configuration.steps)
        selection = self.select_integrator(initial_state=initial_state, t_eval=t_eval)
        times, states, elapsed = self._integrate_with_selection(selection, initial_state, t_eval)

        q_hist = np.zeros((times.size, self.model.nbQ()))
        qdot_hist = np.zeros_like(q_hist)
        qddot_hist = np.zeros_like(q_hist)

        for index, time in enumerate(times):
            q_joint, qdot_joint, qddot_joint = self.joint_kinematics(float(time))
            q_root = states[:ROOT_DOF, index]
            qdot_root = states[ROOT_DOF:, index]
            qddot_root = self._root_acceleration(
                q_root, qdot_root, q_joint, qdot_joint, qddot_joint
            )

            q_hist[index, :ROOT_DOF] = q_root
            q_hist[index, ROOT_DOF:] = q_joint
            qdot_hist[index, :ROOT_DOF] = qdot_root
            qdot_hist[index, ROOT_DOF:] = qdot_joint
            qddot_hist[index, :ROOT_DOF] = qddot_root
            qddot_hist[index, ROOT_DOF:] = qddot_joint

        return AerialSimulationResult(
            time=times,
            q=q_hist,
            qdot=qdot_hist,
            qddot=qddot_hist,
            integrator_method=selection.method,
            rk4_step=selection.rk4_step,
            integration_seconds=elapsed,
        )

    def select_integrator(
        self, *, initial_state: np.ndarray | None = None, t_eval: np.ndarray | None = None
    ) -> IntegratorSelection:
        """Choose the requested integrator, optionally benchmarking RK4 against RK45."""

        if self._integrator_selection is not None:
            return self._integrator_selection

        y0 = (
            np.asarray(initial_state, dtype=float)
            if initial_state is not None
            else np.concatenate(tuple(component[:ROOT_DOF] for component in self.initial_state()))
        )
        evaluation_times = (
            np.asarray(t_eval, dtype=float)
            if t_eval is not None
            else np.linspace(0.0, self.configuration.final_time, self.configuration.steps)
        )
        integrator = self.configuration.integrator.lower()

        if integrator == "rk45":
            self._integrator_selection = IntegratorSelection(
                method="rk45",
                rk4_step=None,
                elapsed_seconds=float("nan"),
            )
            return self._integrator_selection

        if integrator == "rk4":
            if self.configuration.rk4_step is not None:
                self._integrator_selection = IntegratorSelection(
                    method="rk4",
                    rk4_step=self.configuration.rk4_step,
                    elapsed_seconds=float("nan"),
                )
                return self._integrator_selection
            self._integrator_selection = self.determine_rk4_step(y0, evaluation_times)
            return self._integrator_selection

        if integrator == "auto":
            self._integrator_selection = self.determine_rk4_step(
                y0,
                evaluation_times,
                fallback_to_rk45=True,
            )
            return self._integrator_selection

        raise ValueError(f"Unsupported integrator '{self.configuration.integrator}'.")

    def determine_rk4_step(
        self,
        initial_state: np.ndarray | None = None,
        t_eval: np.ndarray | None = None,
        *,
        fallback_to_rk45: bool = False,
    ) -> IntegratorSelection:
        """Benchmark RK4 candidates against RK45 and return the best acceptable choice."""

        y0 = (
            np.asarray(initial_state, dtype=float)
            if initial_state is not None
            else np.concatenate(tuple(component[:ROOT_DOF] for component in self.initial_state()))
        )
        evaluation_times = (
            np.asarray(t_eval, dtype=float)
            if t_eval is not None
            else np.linspace(0.0, self.configuration.final_time, self.configuration.steps)
        )
        reference_times, reference_states, reference_elapsed = self._integrate_rk45(
            y0, evaluation_times
        )
        if not np.array_equal(reference_times, evaluation_times):
            raise RuntimeError("RK45 reference times do not match the requested evaluation grid.")

        best_selection: IntegratorSelection | None = None
        for step in self.configuration.rk4_candidate_steps:
            _, candidate_states, candidate_elapsed = self._integrate_rk4(y0, evaluation_times, step)
            max_state_error = float(np.max(np.abs(candidate_states - reference_states)))
            final_twist_error = float(
                abs(candidate_states[FULL_TWIST_INDEX, -1] - reference_states[FULL_TWIST_INDEX, -1])
            )
            if (
                max_state_error <= self.configuration.rk4_state_tolerance
                and final_twist_error <= self.configuration.rk4_twist_tolerance
                and candidate_elapsed < reference_elapsed
            ):
                selection = IntegratorSelection(
                    method="rk4",
                    rk4_step=float(step),
                    elapsed_seconds=candidate_elapsed,
                    max_state_error=max_state_error,
                    final_twist_error=final_twist_error,
                    reference_elapsed_seconds=reference_elapsed,
                )
                if (
                    best_selection is None
                    or selection.elapsed_seconds < best_selection.elapsed_seconds
                ):
                    best_selection = selection

        if best_selection is not None:
            return best_selection

        if fallback_to_rk45:
            return IntegratorSelection(
                method="rk45",
                rk4_step=None,
                elapsed_seconds=reference_elapsed,
                reference_elapsed_seconds=reference_elapsed,
            )

        fallback_step = float(
            self.configuration.rk4_step
            if self.configuration.rk4_step is not None
            else self.configuration.rk4_candidate_steps[-1]
        )
        _, _, elapsed = self._integrate_rk4(y0, evaluation_times, fallback_step)
        return IntegratorSelection(
            method="rk4",
            rk4_step=fallback_step,
            elapsed_seconds=elapsed,
            max_state_error=float("nan"),
            final_twist_error=float("nan"),
            reference_elapsed_seconds=reference_elapsed,
        )

    def center_of_mass_velocity(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Evaluate the center-of-mass velocity of the reduced model."""

        return self.model.CoMdot(
            biorbd.GeneralizedCoordinates(np.asarray(q, dtype=float)),
            biorbd.GeneralizedVelocity(np.asarray(qdot, dtype=float)),
            True,
        ).to_array()

    def _translation_velocity_cancelling_com_motion(
        self, q: np.ndarray, qdot: np.ndarray
    ) -> np.ndarray:
        """Solve for the translational root velocity that enforces `v_COM = 0`."""

        jacobian = self.model.CoMJacobian(
            biorbd.GeneralizedCoordinates(np.asarray(q, dtype=float)), True
        ).to_array()
        a_matrix = jacobian[:, :3]
        rhs = -jacobian[:, 3:] @ np.asarray(qdot[3:], dtype=float)
        try:
            return np.linalg.solve(a_matrix, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(a_matrix, rhs, rcond=None)[0]

    def _dynamics(self, time: float, state: np.ndarray) -> np.ndarray:
        """Return the time derivative of the root state."""

        q_root = state[:ROOT_DOF]
        qdot_root = state[ROOT_DOF:]
        q_joint, qdot_joint, qddot_joint = self.joint_kinematics(time)
        qddot_root = self._root_acceleration(q_root, qdot_root, q_joint, qdot_joint, qddot_joint)
        return np.concatenate((qdot_root, qddot_root))

    def _root_acceleration(
        self,
        q_root: np.ndarray,
        qdot_root: np.ndarray,
        q_joint: np.ndarray,
        qdot_joint: np.ndarray,
        qddot_joint: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the root acceleration from the floating-base dynamics."""

        q = np.concatenate((q_root, q_joint))
        qdot = np.concatenate((qdot_root, qdot_joint))
        q_biorbd = biorbd.GeneralizedCoordinates(np.asarray(q, dtype=float))
        qdot_biorbd = biorbd.GeneralizedVelocity(np.asarray(qdot, dtype=float))
        out = self.model.ForwardDynamicsFreeFloatingBase(
            q_biorbd, qdot_biorbd, np.asarray(qddot_joint, dtype=float)
        )
        return out.to_array() if hasattr(out, "to_array") else np.asarray(out, dtype=float)

    def _integrate_with_selection(
        self,
        selection: IntegratorSelection,
        initial_state: np.ndarray,
        t_eval: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Integrate with the requested solver and return sampled root states."""

        if selection.method == "rk45":
            return self._integrate_rk45(initial_state, t_eval)
        if selection.method == "rk4" and selection.rk4_step is not None:
            return self._integrate_rk4(initial_state, t_eval, selection.rk4_step)
        raise RuntimeError("Invalid integrator selection.")

    def _integrate_rk45(
        self,
        initial_state: np.ndarray,
        t_eval: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Integrate the root state with RK45 on the requested evaluation grid."""

        start = perf_counter()
        solution = solve_ivp(
            fun=self._dynamics,
            t_span=(0.0, self.configuration.final_time),
            y0=np.asarray(initial_state, dtype=float),
            method="RK45",
            t_eval=np.asarray(t_eval, dtype=float),
            rtol=self.configuration.rtol,
            atol=self.configuration.atol,
        )
        elapsed = perf_counter() - start
        if not solution.success:
            raise RuntimeError(f"RK45 integration failed: {solution.message}")
        return np.asarray(solution.t, dtype=float), np.asarray(solution.y, dtype=float), elapsed

    def _integrate_rk4(
        self,
        initial_state: np.ndarray,
        t_eval: np.ndarray,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Integrate the root state with a fixed-step classical RK4 scheme."""

        fixed_step = float(step)
        if fixed_step <= 0.0:
            raise ValueError("The RK4 step must be strictly positive.")

        t_final = self.configuration.final_time
        state = np.asarray(initial_state, dtype=float).copy()
        sample_times = np.asarray(t_eval, dtype=float)
        time_history = [0.0]
        state_history = [state.copy()]
        current_time = 0.0

        start = perf_counter()
        while current_time < t_final - 1e-15:
            current_step = min(fixed_step, t_final - current_time)
            k1 = self._dynamics(current_time, state)
            k2 = self._dynamics(current_time + 0.5 * current_step, state + 0.5 * current_step * k1)
            k3 = self._dynamics(current_time + 0.5 * current_step, state + 0.5 * current_step * k2)
            k4 = self._dynamics(current_time + current_step, state + current_step * k3)
            state = state + (current_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            current_time += current_step
            time_history.append(current_time)
            state_history.append(state.copy())
        elapsed = perf_counter() - start

        dense_times = np.asarray(time_history, dtype=float)
        dense_states = np.asarray(state_history, dtype=float)
        sampled_states = np.vstack(
            [
                np.interp(sample_times, dense_times, dense_states[:, state_index])
                for state_index in range(dense_states.shape[1])
            ]
        )
        return sample_times, sampled_states, elapsed

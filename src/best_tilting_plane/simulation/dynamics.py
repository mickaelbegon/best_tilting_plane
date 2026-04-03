"""Predictive zero-gravity simulation based on a reduced floating-base `biorbd` model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import biorbd
import numpy as np
from scipy.integrate import solve_ivp

from best_tilting_plane.modeling import ReducedAerialBiomod, root_twist_index
from best_tilting_plane.simulation.arm_motion import PrescribedArmMotion, TwistOptimizationVariables

ROOT_DOF = 6
JOINT_DOF = 4
FULL_TWIST_INDEX = 3 + root_twist_index()


@dataclass(frozen=True)
class SimulationConfiguration:
    """Numerical parameters used by the predictive simulation."""

    final_time: float = 1.0
    steps: int = 201
    somersault_rate: float = 2.0 * np.pi
    rtol: float = 1e-8
    atol: float = 1e-10


@dataclass(frozen=True)
class AerialSimulationResult:
    """Time histories returned by the predictive simulation."""

    time: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray

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
        prescribed_motion: PrescribedArmMotion,
        *,
        configuration: SimulationConfiguration | None = None,
        model: biorbd.Model | None = None,
    ) -> None:
        """Load the `biorbd` model and store the simulation settings."""

        self.model_path = str(model_path)
        self.model = model if model is not None else biorbd.Model(self.model_path)
        self.prescribed_motion = prescribed_motion
        self.configuration = configuration or SimulationConfiguration()

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
        return cls(model_path, PrescribedArmMotion(variables), configuration=configuration)

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
        qdot[ROOT_DOF:] = qdot_joint
        qdot[:3] = self._translation_velocity_cancelling_com_motion(q, qdot)
        return q, qdot

    def simulate(self) -> AerialSimulationResult:
        """Run the predictive simulation with `RK45` until the configured final time."""

        q0, qdot0 = self.initial_state()
        initial_state = np.concatenate((q0[:ROOT_DOF], qdot0[:ROOT_DOF]))
        t_eval = np.linspace(0.0, self.configuration.final_time, self.configuration.steps)

        solution = solve_ivp(
            fun=self._dynamics,
            t_span=(0.0, self.configuration.final_time),
            y0=initial_state,
            method="RK45",
            t_eval=t_eval,
            rtol=self.configuration.rtol,
            atol=self.configuration.atol,
        )
        if not solution.success:
            raise RuntimeError(f"RK45 integration failed: {solution.message}")

        q_hist = np.zeros((solution.t.size, self.model.nbQ()))
        qdot_hist = np.zeros_like(q_hist)
        qddot_hist = np.zeros_like(q_hist)

        for index, time in enumerate(solution.t):
            q_joint, qdot_joint, qddot_joint = self.joint_kinematics(float(time))
            q_root = solution.y[:ROOT_DOF, index]
            qdot_root = solution.y[ROOT_DOF:, index]
            qddot_root = self._root_acceleration(
                q_root, qdot_root, q_joint, qdot_joint, qddot_joint
            )

            q_hist[index, :ROOT_DOF] = q_root
            q_hist[index, ROOT_DOF:] = q_joint
            qdot_hist[index, :ROOT_DOF] = qdot_root
            qdot_hist[index, ROOT_DOF:] = qdot_joint
            qddot_hist[index, :ROOT_DOF] = qddot_root
            qddot_hist[index, ROOT_DOF:] = qddot_joint

        return AerialSimulationResult(time=solution.t, q=q_hist, qdot=qdot_hist, qddot=qddot_hist)

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

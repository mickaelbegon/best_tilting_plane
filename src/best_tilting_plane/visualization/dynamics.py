"""Whole-system dynamic observables used by the GUI."""

from __future__ import annotations

from pathlib import Path

import biorbd
import numpy as np

LEFT_ARM_VELOCITY_SLICE = slice(6, 8)
RIGHT_ARM_VELOCITY_SLICE = slice(8, 10)
BODY_VELOCITY_SLICE = slice(0, 6)
SHOULDER_DOF_INDICES = (6, 7, 8, 9)


def system_observables(
    model_path: str | Path,
    q_history: np.ndarray,
    qddot_history: np.ndarray,
    qdot_history: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return center-of-mass and angular-momentum trajectories for the whole system."""

    model = biorbd.Model(str(model_path))
    q_history = np.asarray(q_history, dtype=float)
    qdot_history = np.asarray(qdot_history, dtype=float)
    qddot_history = np.asarray(qddot_history, dtype=float)

    center_of_mass = np.zeros((q_history.shape[0], 3), dtype=float)
    angular_momentum = np.zeros((q_history.shape[0], 3), dtype=float)
    angular_momentum_groups = np.zeros((q_history.shape[0], 3, 3), dtype=float)
    shoulder_torques = np.zeros((q_history.shape[0], len(SHOULDER_DOF_INDICES), 4), dtype=float)
    zero_qdot = np.zeros(model.nbQdot(), dtype=float)

    for frame_index, (q, qdot, qddot) in enumerate(zip(q_history, qdot_history, qddot_history)):
        q_biorbd = biorbd.GeneralizedCoordinates(q)
        qdot_biorbd = biorbd.GeneralizedVelocity(qdot)
        qddot_biorbd = biorbd.GeneralizedAcceleration(qddot)
        center_of_mass[frame_index, :] = model.CoM(q_biorbd).to_array()
        angular_momentum[frame_index, :] = model.angularMomentum(
            q_biorbd, qdot_biorbd, True
        ).to_array()
        inverse_dynamics = model.InverseDynamics(q_biorbd, qdot_biorbd, qddot_biorbd).to_array()
        nonlinear_static = model.NonLinearEffect(
            q_biorbd,
            biorbd.GeneralizedVelocity(zero_qdot),
        ).to_array()
        nonlinear_full = model.NonLinearEffect(q_biorbd, qdot_biorbd).to_array()
        inertial = model.massMatrix(q_biorbd).to_array() @ qddot
        coriolis_centrifugal = nonlinear_full - nonlinear_static
        for shoulder_index, dof_index in enumerate(SHOULDER_DOF_INDICES):
            shoulder_torques[frame_index, shoulder_index, 0] = inverse_dynamics[dof_index]
            shoulder_torques[frame_index, shoulder_index, 1] = inertial[dof_index]
            shoulder_torques[frame_index, shoulder_index, 2] = nonlinear_static[dof_index]
            shoulder_torques[frame_index, shoulder_index, 3] = coriolis_centrifugal[dof_index]
        for group_index, velocity_slice in enumerate(
            (LEFT_ARM_VELOCITY_SLICE, RIGHT_ARM_VELOCITY_SLICE, BODY_VELOCITY_SLICE)
        ):
            group_qdot = np.zeros_like(qdot)
            group_qdot[velocity_slice] = qdot[velocity_slice]
            angular_momentum_groups[frame_index, group_index, :] = model.angularMomentum(
                q_biorbd,
                biorbd.GeneralizedVelocity(group_qdot),
                True,
            ).to_array()

    return {
        "center_of_mass": center_of_mass,
        "angular_momentum": angular_momentum,
        "angular_momentum_groups": angular_momentum_groups,
        "shoulder_torques": shoulder_torques,
    }

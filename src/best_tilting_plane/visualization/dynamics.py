"""Whole-system dynamic observables used by the GUI."""

from __future__ import annotations

from pathlib import Path

import biorbd
import numpy as np

LEFT_ARM_VELOCITY_SLICE = slice(6, 8)
RIGHT_ARM_VELOCITY_SLICE = slice(8, 10)
BODY_VELOCITY_SLICE = slice(0, 6)
SHOULDER_DOF_INDICES = (6, 7, 8, 9)
PELVIS_SEGMENT_NAME = "pelvis"


def _principal_body_vertical_axis(
    model: biorbd.Model,
    q_biorbd: biorbd.GeneralizedCoordinates,
    *,
    pelvis_index: int,
) -> np.ndarray:
    """Return the principal-inertia axis aligned with the body's vertical direction."""

    inertia_tensor = np.asarray(model.bodyInertia(q_biorbd).to_array(), dtype=float)
    symmetric_inertia = 0.5 * (inertia_tensor + inertia_tensor.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_inertia)
    axis = np.asarray(eigenvectors[:, int(np.argmin(eigenvalues))], dtype=float)
    pelvis_axes = np.asarray(model.globalJCS(q_biorbd, pelvis_index).to_array()[:3, :3], dtype=float)
    pelvis_vertical = pelvis_axes[:, 2]
    if float(np.dot(axis, pelvis_vertical)) < 0.0:
        axis = -axis
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        return pelvis_vertical / np.linalg.norm(pelvis_vertical)
    return axis / axis_norm


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
    principal_vertical_axis = np.zeros((q_history.shape[0], 3), dtype=float)
    angular_momentum_principal_axis = np.zeros(q_history.shape[0], dtype=float)
    angular_momentum_groups_principal_axis = np.zeros((q_history.shape[0], 3), dtype=float)
    shoulder_torques = np.zeros((q_history.shape[0], len(SHOULDER_DOF_INDICES), 4), dtype=float)
    zero_qdot = np.zeros(model.nbQdot(), dtype=float)
    pelvis_index = next(
        index
        for index in range(model.nbSegment())
        if (
            model.segment(index).name().to_string()
            if hasattr(model.segment(index).name(), "to_string")
            else str(model.segment(index).name())
        )
        == PELVIS_SEGMENT_NAME
    )

    for frame_index, (q, qdot, qddot) in enumerate(zip(q_history, qdot_history, qddot_history)):
        q_biorbd = biorbd.GeneralizedCoordinates(q)
        qdot_biorbd = biorbd.GeneralizedVelocity(qdot)
        qddot_biorbd = biorbd.GeneralizedAcceleration(qddot)
        center_of_mass[frame_index, :] = model.CoM(q_biorbd).to_array()
        angular_momentum[frame_index, :] = model.angularMomentum(
            q_biorbd, qdot_biorbd, True
        ).to_array()
        principal_axis = _principal_body_vertical_axis(
            model,
            q_biorbd,
            pelvis_index=pelvis_index,
        )
        principal_vertical_axis[frame_index, :] = principal_axis
        angular_momentum_principal_axis[frame_index] = float(
            np.dot(angular_momentum[frame_index, :], principal_axis)
        )
        inverse_dynamics = model.InverseDynamics(q_biorbd, qdot_biorbd, qddot_biorbd).to_array()
        gravity_effects = model.NonLinearEffect(
            q_biorbd,
            biorbd.GeneralizedVelocity(zero_qdot),
        ).to_array()
        nonlinear_full = model.NonLinearEffect(q_biorbd, qdot_biorbd).to_array()
        inertial = model.massMatrix(q_biorbd).to_array() @ qddot
        for shoulder_index, dof_index in enumerate(SHOULDER_DOF_INDICES):
            shoulder_torques[frame_index, shoulder_index, 0] = inverse_dynamics[dof_index]
            shoulder_torques[frame_index, shoulder_index, 1] = inertial[dof_index]
            shoulder_torques[frame_index, shoulder_index, 2] = nonlinear_full[dof_index]
            shoulder_torques[frame_index, shoulder_index, 3] = gravity_effects[dof_index]
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
            angular_momentum_groups_principal_axis[frame_index, group_index] = float(
                np.dot(angular_momentum_groups[frame_index, group_index, :], principal_axis)
            )

    return {
        "center_of_mass": center_of_mass,
        "angular_momentum": angular_momentum,
        "angular_momentum_groups": angular_momentum_groups,
        "principal_vertical_axis": principal_vertical_axis,
        "angular_momentum_principal_axis": angular_momentum_principal_axis,
        "angular_momentum_groups_principal_axis": angular_momentum_groups_principal_axis,
        "shoulder_torques": shoulder_torques,
    }

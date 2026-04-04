"""Whole-system dynamic observables used by the GUI."""

from __future__ import annotations

from pathlib import Path

import biorbd
import numpy as np


def system_observables(
    model_path: str | Path,
    q_history: np.ndarray,
    qdot_history: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return center-of-mass and angular-momentum trajectories for the whole system."""

    model = biorbd.Model(str(model_path))
    q_history = np.asarray(q_history, dtype=float)
    qdot_history = np.asarray(qdot_history, dtype=float)

    center_of_mass = np.zeros((q_history.shape[0], 3), dtype=float)
    angular_momentum = np.zeros((q_history.shape[0], 3), dtype=float)

    for frame_index, (q, qdot) in enumerate(zip(q_history, qdot_history)):
        q_biorbd = biorbd.GeneralizedCoordinates(q)
        qdot_biorbd = biorbd.GeneralizedVelocity(qdot)
        center_of_mass[frame_index, :] = model.CoM(q_biorbd).to_array()
        angular_momentum[frame_index, :] = model.angularMomentum(
            q_biorbd, qdot_biorbd, True
        ).to_array()

    return {"center_of_mass": center_of_mass, "angular_momentum": angular_momentum}

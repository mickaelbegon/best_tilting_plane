"""Helpers for visualizing marker trajectories in the best-tilting-plane frame."""

from __future__ import annotations

import numpy as np
from best_tilting_plane.visualization.btp import best_tilting_plane_axes, best_tilting_plane_normal


def arm_btp_reference_trajectories(
    marker_trajectories: dict[str, np.ndarray],
    somersault_angles: np.ndarray,
    *,
    reference_marker: str = "pelvis_origin",
) -> dict[str, np.ndarray]:
    """Return marker trajectories expressed in the moving best-tilting-plane frame.

    The returned coordinates are:
    - `x`: along the somersault axis
    - `y`: along the twist axis rotated by the somersault angle
    - `z`: signed distance out of the best tilting plane
    """

    somersault_angles = np.asarray(somersault_angles, dtype=float).reshape(-1)
    reference = np.asarray(marker_trajectories[reference_marker], dtype=float)
    frame_count = reference.shape[0]
    if somersault_angles.shape[0] != frame_count:
        raise ValueError("The number of somersault angles must match the marker trajectory length.")

    basis = np.zeros((frame_count, 3, 3), dtype=float)
    for frame_index, somersault_angle in enumerate(somersault_angles):
        somersault_axis, twist_axis = best_tilting_plane_axes(float(somersault_angle))
        basis[frame_index, 0, :] = somersault_axis
        basis[frame_index, 1, :] = twist_axis
        basis[frame_index, 2, :] = best_tilting_plane_normal(float(somersault_angle))

    projected: dict[str, np.ndarray] = {}
    for marker_name, marker_values in marker_trajectories.items():
        relative = np.asarray(marker_values, dtype=float) - reference
        projected[marker_name] = np.einsum("fij,fj->fi", basis, relative)
    return projected

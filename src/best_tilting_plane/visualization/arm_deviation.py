"""Arm deviation metrics relative to the best tilting plane."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from best_tilting_plane.visualization.btp import best_tilting_plane_normal
from best_tilting_plane.visualization.frames import segment_frame_trajectories


def signed_deviation_from_plane(vector: np.ndarray, plane_normal: np.ndarray) -> float:
    """Return the signed angle between a vector and a plane in radians."""

    direction = np.asarray(vector, dtype=float).reshape(3)
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    direction_norm = np.linalg.norm(direction)
    normal_norm = np.linalg.norm(normal)
    if direction_norm == 0.0 or normal_norm == 0.0:
        return 0.0

    direction_unit = direction / direction_norm
    normal_unit = normal / normal_norm
    normal_component = float(np.dot(direction_unit, normal_unit))
    tangential_norm = float(np.linalg.norm(direction_unit - normal_component * normal_unit))
    return float(np.arctan2(normal_component, tangential_norm))


def arm_deviation_from_frames(
    frame_trajectories: dict[str, dict[str, np.ndarray]],
    somersault_angles: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the signed arm deviations relative to the best tilting plane."""

    somersault_angles = np.asarray(somersault_angles, dtype=float)
    left = np.zeros(somersault_angles.shape[0], dtype=float)
    right = np.zeros_like(left)

    for frame_index, somersault_angle in enumerate(somersault_angles):
        normal = best_tilting_plane_normal(float(somersault_angle))
        left_vector = (
            frame_trajectories["forearm_left"]["origin"][frame_index]
            - frame_trajectories["upper_arm_left"]["origin"][frame_index]
        )
        right_vector = (
            frame_trajectories["forearm_right"]["origin"][frame_index]
            - frame_trajectories["upper_arm_right"]["origin"][frame_index]
        )
        left[frame_index] = signed_deviation_from_plane(left_vector, normal)
        right[frame_index] = signed_deviation_from_plane(right_vector, normal)

    return {"left": left, "right": right}


def arm_deviation_trajectories(
    model_path: str | Path,
    q_history: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the signed arm deviations relative to the best tilting plane."""

    frames = segment_frame_trajectories(
        model_path,
        q_history,
        ("upper_arm_left", "forearm_left", "upper_arm_right", "forearm_right"),
    )
    q_history = np.asarray(q_history, dtype=float)
    return arm_deviation_from_frames(frames, q_history[:, 3])

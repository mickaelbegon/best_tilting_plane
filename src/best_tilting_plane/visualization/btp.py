"""Best-tilting-plane helpers."""

from __future__ import annotations

import numpy as np


def _rotation_x(angle: float) -> np.ndarray:
    """Return the rotation matrix associated with the somersault angle."""

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_angle, -sin_angle],
            [0.0, sin_angle, cos_angle],
        ],
        dtype=float,
    )


def best_tilting_plane_axes(somersault_angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the two unit directions spanning the best tilting plane."""

    somersault_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    twist_axis = _rotation_x(somersault_angle) @ np.array([0.0, 0.0, 1.0], dtype=float)
    return somersault_axis, twist_axis


def best_tilting_plane_normal(somersault_angle: float) -> np.ndarray:
    """Return the unit normal to the best tilting plane."""

    somersault_axis, twist_axis = best_tilting_plane_axes(somersault_angle)
    normal = np.cross(somersault_axis, twist_axis)
    return normal / np.linalg.norm(normal)


def best_tilting_plane_corners(
    origin: np.ndarray,
    somersault_angle: float,
    *,
    half_width: float = 0.6,
    half_height: float = 0.8,
) -> np.ndarray:
    """Return four corners describing the best tilting plane.

    The plane is spanned by the somersault axis and the twist axis rotated by the somersault angle only.
    """

    origin = np.asarray(origin, dtype=float).reshape(3)
    somersault_axis, twist_axis = best_tilting_plane_axes(somersault_angle)

    return np.vstack(
        [
            origin - half_width * somersault_axis - half_height * twist_axis,
            origin + half_width * somersault_axis - half_height * twist_axis,
            origin + half_width * somersault_axis + half_height * twist_axis,
            origin - half_width * somersault_axis + half_height * twist_axis,
        ]
    )

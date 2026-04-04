"""Visualization helpers for marker extraction, animation, and the best tilting plane."""

from .arm_deviation import (
    arm_deviation_from_frames,
    arm_deviation_trajectories,
    signed_deviation_from_plane,
)
from .btp import best_tilting_plane_axes, best_tilting_plane_corners, best_tilting_plane_normal
from .dynamics import system_observables
from .frames import segment_frame_trajectories
from .markers import SKELETON_CONNECTIONS, marker_trajectories

__all__ = [
    "SKELETON_CONNECTIONS",
    "arm_deviation_from_frames",
    "arm_deviation_trajectories",
    "best_tilting_plane_axes",
    "best_tilting_plane_corners",
    "best_tilting_plane_normal",
    "marker_trajectories",
    "system_observables",
    "segment_frame_trajectories",
    "signed_deviation_from_plane",
]

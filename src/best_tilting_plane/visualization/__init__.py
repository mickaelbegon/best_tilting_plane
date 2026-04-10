"""Visualization helpers for marker extraction, animation, and the best tilting plane."""

from .arm_btp_view import arm_btp_reference_trajectories
from .arm_deviation import (
    arm_deviation_from_frames,
    arm_deviation_trajectories,
    signed_deviation_from_plane,
)
from .arm_top_view import ARM_TOP_VIEW_MARKERS, arm_top_view_trajectories
from .btp import best_tilting_plane_axes, best_tilting_plane_corners, best_tilting_plane_normal
from .dynamics import system_observables
from .external_figure import present_external_figure
from .frames import segment_frame_trajectories
from .markers import SKELETON_CONNECTIONS, marker_trajectories

__all__ = [
    "ARM_TOP_VIEW_MARKERS",
    "SKELETON_CONNECTIONS",
    "arm_btp_reference_trajectories",
    "arm_deviation_from_frames",
    "arm_deviation_trajectories",
    "arm_top_view_trajectories",
    "best_tilting_plane_axes",
    "best_tilting_plane_corners",
    "best_tilting_plane_normal",
    "marker_trajectories",
    "present_external_figure",
    "system_observables",
    "segment_frame_trajectories",
    "signed_deviation_from_plane",
]

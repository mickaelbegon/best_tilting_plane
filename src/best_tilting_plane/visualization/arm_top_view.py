"""Top-view helpers for visualizing arm trajectories relative to the pelvis."""

from __future__ import annotations

import numpy as np

ARM_TOP_VIEW_MARKERS = (
    "shoulder_left",
    "elbow_left",
    "wrist_left",
    "hand_left",
    "shoulder_right",
    "elbow_right",
    "wrist_right",
    "hand_right",
)


def arm_top_view_trajectories(
    marker_trajectories: dict[str, np.ndarray],
    *,
    reference_marker: str = "pelvis_origin",
    reference_axes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return pelvis-relative arm trajectories in either global or pelvis coordinates."""

    reference_positions = np.asarray(marker_trajectories[reference_marker], dtype=float)
    if reference_axes is not None:
        local_axes = np.asarray(reference_axes, dtype=float)
        return {
            marker_name: np.einsum(
                "nij,nj->ni",
                np.transpose(local_axes, (0, 2, 1)),
                np.asarray(marker_trajectories[marker_name], dtype=float) - reference_positions,
            )[:, :2]
            for marker_name in ARM_TOP_VIEW_MARKERS
        }

    reference = reference_positions[:, :2]
    return {
        marker_name: np.asarray(marker_trajectories[marker_name], dtype=float)[:, :2] - reference
        for marker_name in ARM_TOP_VIEW_MARKERS
    }

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
) -> dict[str, np.ndarray]:
    """Return pelvis-relative arm trajectories projected in the global top view."""

    reference = np.asarray(marker_trajectories[reference_marker], dtype=float)[:, :2]
    return {
        marker_name: np.asarray(marker_trajectories[marker_name], dtype=float)[:, :2] - reference
        for marker_name in ARM_TOP_VIEW_MARKERS
    }

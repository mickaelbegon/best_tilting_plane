"""Marker extraction utilities for plotting the reduced whole-body model."""

from __future__ import annotations

from pathlib import Path

import biorbd
import numpy as np

SKELETON_CONNECTIONS = (
    ("pelvis_origin", "head_top"),
    ("shoulder_left", "elbow_left"),
    ("elbow_left", "wrist_left"),
    ("wrist_left", "hand_left"),
    ("shoulder_right", "elbow_right"),
    ("elbow_right", "wrist_right"),
    ("wrist_right", "hand_right"),
    ("hip_left", "knee_left"),
    ("knee_left", "ankle_left"),
    ("ankle_left", "toe_left"),
    ("hip_right", "knee_right"),
    ("knee_right", "ankle_right"),
    ("ankle_right", "toe_right"),
    ("shoulder_left", "shoulder_right"),
    ("hip_left", "hip_right"),
    ("shoulder_left", "hip_left"),
    ("shoulder_right", "hip_right"),
)


def marker_trajectories(model_path: str | Path, q_history: np.ndarray) -> dict[str, np.ndarray]:
    """Return marker trajectories with shape `(n_frames, 3)` for each marker name."""

    model = biorbd.Model(str(model_path))
    names = [name.to_string() for name in model.markerNames()]
    trajectories = {name: np.zeros((q_history.shape[0], 3), dtype=float) for name in names}

    for frame_index, q in enumerate(np.asarray(q_history, dtype=float)):
        marker_values = model.markers(biorbd.GeneralizedCoordinates(q))
        for name, marker in zip(names, marker_values):
            trajectories[name][frame_index, :] = marker.to_array()

    return trajectories

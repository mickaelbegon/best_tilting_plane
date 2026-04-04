"""Segment-frame extraction utilities for 3D visualization."""

from __future__ import annotations

from pathlib import Path

import biorbd
import numpy as np


def segment_frame_trajectories(
    model_path: str | Path,
    q_history: np.ndarray,
    segment_names: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Return segment origins and axes over time for the requested segment names."""

    model = biorbd.Model(str(model_path))
    index_by_name = {}
    for index in range(model.nbSegment()):
        segment = model.segment(index)
        name = (
            segment.name().to_string()
            if hasattr(segment.name(), "to_string")
            else str(segment.name())
        )
        index_by_name[name] = index

    missing_names = [name for name in segment_names if name not in index_by_name]
    if missing_names:
        raise ValueError(f"Unknown segment names: {missing_names}")

    trajectories = {
        name: {
            "origin": np.zeros((q_history.shape[0], 3), dtype=float),
            "axes": np.zeros((q_history.shape[0], 3, 3), dtype=float),
        }
        for name in segment_names
    }

    for frame_index, q in enumerate(np.asarray(q_history, dtype=float)):
        q_biorbd = biorbd.GeneralizedCoordinates(q)
        for name in segment_names:
            jcs = model.globalJCS(q_biorbd, index_by_name[name]).to_array()
            trajectories[name]["origin"][frame_index, :] = jcs[:3, 3]
            trajectories[name]["axes"][frame_index, :, :] = jcs[:3, :3]

    return trajectories

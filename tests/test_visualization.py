"""Tests for marker extraction and best-tilting-plane geometry."""

from pathlib import Path

import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.visualization import best_tilting_plane_corners, marker_trajectories


def test_marker_trajectories_follow_the_simulation_shape(tmp_path: Path) -> None:
    """Marker extraction should preserve the frame count and expose named trajectories."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    q_history = np.zeros((5, 10))
    trajectories = marker_trajectories(model_path, q_history)

    assert "pelvis_origin" in trajectories
    assert "head_top" in trajectories
    assert trajectories["pelvis_origin"].shape == (5, 3)


def test_best_tilting_plane_corners_stay_centered_on_the_origin() -> None:
    """The generated plane corners should stay centered on the requested origin."""

    origin = np.array([0.2, -0.1, 0.4])
    corners = best_tilting_plane_corners(origin, somersault_angle=np.pi / 3.0)

    np.testing.assert_allclose(np.mean(corners, axis=0), origin)

"""Tests for marker extraction and best-tilting-plane geometry."""

from pathlib import Path

import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.visualization import (
    arm_btp_reference_trajectories,
    arm_top_view_trajectories,
    best_tilting_plane_corners,
    marker_trajectories,
)


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


def test_arm_top_view_trajectories_are_relative_to_the_pelvis_and_keep_xy() -> None:
    """The top-view helper should return pelvis-relative `(x, y)` arm coordinates."""

    trajectories = {
        "pelvis_origin": np.array([[1.0, 2.0, 5.0], [1.5, 2.5, 5.5]]),
        "shoulder_left": np.array([[0.8, 2.2, 5.0], [1.3, 2.8, 5.5]]),
        "elbow_left": np.array([[0.6, 2.4, 5.1], [1.1, 3.0, 5.6]]),
        "wrist_left": np.array([[0.4, 2.6, 5.2], [0.9, 3.1, 5.7]]),
        "hand_left": np.array([[0.2, 2.8, 5.3], [0.7, 3.2, 5.8]]),
        "shoulder_right": np.array([[1.2, 2.2, 5.0], [1.7, 2.8, 5.5]]),
        "elbow_right": np.array([[1.4, 2.4, 5.1], [1.9, 3.0, 5.6]]),
        "wrist_right": np.array([[1.6, 2.6, 5.2], [2.1, 3.1, 5.7]]),
        "hand_right": np.array([[1.8, 2.8, 5.3], [2.3, 3.2, 5.8]]),
    }

    top_view = arm_top_view_trajectories(trajectories)

    np.testing.assert_allclose(top_view["hand_left"], np.array([[-0.8, 0.8], [-0.8, 0.7]]))
    np.testing.assert_allclose(top_view["hand_right"], np.array([[0.8, 0.8], [0.8, 0.7]]))


def test_arm_btp_reference_trajectories_project_markers_in_the_btp_basis() -> None:
    """The BTP helper should express all markers in the moving `(somersault, twist, normal)` frame."""

    trajectories = {
        "pelvis_origin": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "head_top": np.array([[1.0, 2.3, 3.8], [1.0, 1.2, 3.3]]),
        "shoulder_left": np.array([[0.8, 2.0, 2.6], [0.8, 2.4, 3.0]]),
        "elbow_left": np.array([[0.8, 2.0, 2.6], [0.8, 2.4, 3.0]]),
        "wrist_left": np.array([[0.8, 2.0, 2.6], [0.8, 2.4, 3.0]]),
        "hand_left": np.array([[0.8, 2.0, 2.6], [0.8, 2.4, 3.0]]),
        "shoulder_right": np.array([[1.2, 2.0, 3.4], [1.2, 1.6, 3.0]]),
        "elbow_right": np.array([[1.2, 2.0, 3.4], [1.2, 1.6, 3.0]]),
        "wrist_right": np.array([[1.2, 2.0, 3.4], [1.2, 1.6, 3.0]]),
        "hand_right": np.array([[1.2, 2.0, 3.4], [1.2, 1.6, 3.0]]),
    }

    projected = arm_btp_reference_trajectories(trajectories, np.array([0.0, np.pi / 2.0]))

    np.testing.assert_allclose(projected["pelvis_origin"], np.zeros((2, 3)), atol=1e-12)
    np.testing.assert_allclose(projected["hand_left"][0], np.array([-0.2, -0.4, 0.0]), atol=1e-12)
    np.testing.assert_allclose(projected["hand_right"][0], np.array([0.2, 0.4, 0.0]), atol=1e-12)
    np.testing.assert_allclose(projected["hand_left"][1], np.array([-0.2, -0.4, 0.0]), atol=1e-12)
    np.testing.assert_allclose(projected["hand_right"][1], np.array([0.2, 0.4, 0.0]), atol=1e-12)
    np.testing.assert_allclose(projected["head_top"][0], np.array([0.0, 0.8, -0.3]), atol=1e-12)

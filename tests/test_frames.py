"""Tests for segment-frame extraction utilities and optimization bounds."""

from pathlib import Path

import numpy as np
import pytest

from best_tilting_plane.modeling import (
    ARM_ELEVATION_SEQUENCE,
    ARM_PLANE_SEQUENCE,
    ARM_SEGMENTS_FOR_DEVIATION,
    ARM_SEGMENTS_FOR_VISUALIZATION,
    GLOBAL_AXIS_LABELS,
    LEFT_ARM_ELEVATION_BOUNDS_DEG,
    LEFT_ARM_PLANE_BOUNDS_DEG,
    ROOT_ROTATION_SEQUENCE,
    RIGHT_ARM_ELEVATION_BOUNDS_DEG,
    RIGHT_ARM_PLANE_BOUNDS_DEG,
    ReducedAerialBiomod,
)
from best_tilting_plane.gui.app import SLIDER_DEFINITIONS
from best_tilting_plane.optimization.ipopt import (
    LEFT_ARM_PLANE_BOUNDS,
    RIGHT_ARM_START_BOUNDS,
    RIGHT_ARM_PLANE_BOUNDS,
    TwistStrategyOptimizer,
)
from best_tilting_plane.visualization import segment_frame_trajectories


def test_default_optimization_bounds_match_requested_right_arm_start_constraint() -> None:
    """The right-arm start should be explicitly constrained between 0.0 and 0.7 s."""

    bounds = TwistStrategyOptimizer.default_bounds()

    assert RIGHT_ARM_START_BOUNDS == (0.0, 0.7)
    assert LEFT_ARM_PLANE_BOUNDS_DEG == (-135.0, 0.0)
    assert RIGHT_ARM_PLANE_BOUNDS_DEG == (0.0, 135.0)
    assert LEFT_ARM_ELEVATION_BOUNDS_DEG == (-180.0, 0.0)
    assert RIGHT_ARM_ELEVATION_BOUNDS_DEG == (0.0, 180.0)
    assert bounds.lower[0] == pytest.approx(0.0)
    assert bounds.upper[0] == pytest.approx(0.7)
    np.testing.assert_allclose(
        LEFT_ARM_PLANE_BOUNDS,
        np.deg2rad(np.array(LEFT_ARM_PLANE_BOUNDS_DEG)),
    )
    np.testing.assert_allclose(
        RIGHT_ARM_PLANE_BOUNDS,
        np.deg2rad(np.array(RIGHT_ARM_PLANE_BOUNDS_DEG)),
    )
    np.testing.assert_allclose(
        bounds.lower[1:],
        np.deg2rad(np.array([-135.0, -135.0, 0.0, 0.0])),
    )
    np.testing.assert_allclose(
        bounds.upper[1:],
        np.deg2rad(np.array([0.0, 0.0, 135.0, 135.0])),
    )


def test_rotation_sequences_are_exposed_for_root_and_arms() -> None:
    """The model should expose the sequences used by the biomod."""

    assert ROOT_ROTATION_SEQUENCE == "xyz"
    assert ARM_PLANE_SEQUENCE == ("z",)
    assert ARM_ELEVATION_SEQUENCE == ("y",)
    assert GLOBAL_AXIS_LABELS == ("x_mediolateral", "y_anteroposterior", "z_longitudinal")


def test_gui_slider_bounds_match_the_validated_constraints() -> None:
    """The GUI should expose the start-time slider and keep the validated plane bounds in the model."""

    slider_by_name = {definition.name: definition for definition in SLIDER_DEFINITIONS}

    assert slider_by_name["right_arm_start"].minimum == RIGHT_ARM_START_BOUNDS[0]
    assert slider_by_name["right_arm_start"].maximum == RIGHT_ARM_START_BOUNDS[1]
    assert set(slider_by_name) == {"right_arm_start"}
    assert LEFT_ARM_PLANE_BOUNDS_DEG == (-135.0, 0.0)
    assert RIGHT_ARM_PLANE_BOUNDS_DEG == (0.0, 135.0)


def test_segment_frame_trajectories_extract_trunk_and_arm_frames(tmp_path: Path) -> None:
    """Frame extraction should return origins and orthonormal axes for trunk and arms."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    q_history = np.zeros((3, 10))
    trajectories = segment_frame_trajectories(model_path, q_history, ARM_SEGMENTS_FOR_VISUALIZATION)

    assert set(trajectories) == set(ARM_SEGMENTS_FOR_VISUALIZATION)
    for segment_name in ARM_SEGMENTS_FOR_VISUALIZATION:
        assert trajectories[segment_name]["origin"].shape == (3, 3)
        assert trajectories[segment_name]["axes"].shape == (3, 3, 3)
        np.testing.assert_allclose(
            trajectories[segment_name]["axes"][0] @ trajectories[segment_name]["axes"][0].T,
            np.eye(3),
            atol=1e-10,
        )

    deviation_frames = segment_frame_trajectories(model_path, q_history, ARM_SEGMENTS_FOR_DEVIATION)
    assert set(deviation_frames) == set(ARM_SEGMENTS_FOR_DEVIATION)

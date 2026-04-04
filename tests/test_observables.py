"""Tests for whole-system observables and corrected foot orientation."""

from pathlib import Path

import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.visualization import marker_trajectories, system_observables


def test_system_observables_return_com_and_angular_momentum(tmp_path: Path) -> None:
    """The whole-system observable helper should return finite CoM and angular momentum histories."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    q_history = np.zeros((4, 10))
    qdot_history = np.zeros((4, 10))

    observables = system_observables(model_path, q_history, qdot_history)

    assert observables["center_of_mass"].shape == (4, 3)
    assert observables["angular_momentum"].shape == (4, 3)
    np.testing.assert_allclose(observables["angular_momentum"], 0.0, atol=1e-12)


def test_foot_marker_now_extends_along_anteroposterior_axis(tmp_path: Path) -> None:
    """The toe marker should extend along the y axis, not the mediolateral x axis."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    q_history = np.zeros((1, 10))
    trajectories = marker_trajectories(model_path, q_history)

    left_ankle = trajectories["ankle_left"][0]
    left_toe = trajectories["toe_left"][0]
    displacement = left_toe - left_ankle

    assert displacement[1] > 0.0
    assert displacement[0] == 0.0

"""Tests for arm deviation relative to the best tilting plane."""

from pathlib import Path

import numpy as np
import pytest

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.visualization import (
    arm_deviation_trajectories,
    best_tilting_plane_normal,
    signed_deviation_from_plane,
)


def test_signed_deviation_from_plane_is_zero_inside_the_plane() -> None:
    """A vector lying in the plane should have zero deviation."""

    normal = best_tilting_plane_normal(0.0)
    deviation = signed_deviation_from_plane(np.array([0.0, 0.0, 1.0]), normal)

    assert deviation == pytest.approx(0.0)


def test_signed_deviation_from_plane_reaches_ninety_degrees_on_the_normal() -> None:
    """A vector aligned with the plane normal should deviate by ninety degrees."""

    normal = best_tilting_plane_normal(0.0)
    deviation = signed_deviation_from_plane(normal, normal)

    assert deviation == pytest.approx(np.pi / 2.0)


def test_arm_deviation_trajectories_return_two_finite_series(tmp_path: Path) -> None:
    """The full-model helper should return finite left/right deviation histories."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    q_history = np.zeros((5, 10))

    deviations = arm_deviation_trajectories(model_path, q_history)

    assert set(deviations) == {"left", "right"}
    assert deviations["left"].shape == (5,)
    assert deviations["right"].shape == (5,)
    assert np.all(np.isfinite(deviations["left"]))
    assert np.all(np.isfinite(deviations["right"]))

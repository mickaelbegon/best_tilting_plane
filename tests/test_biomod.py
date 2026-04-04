"""Tests for the reduced whole-body `bioMod` generator."""

from pathlib import Path

import biorbd
import numpy as np

from best_tilting_plane.modeling.biomod import ReducedAerialBiomod


def test_generated_biomod_loads_in_biorbd(tmp_path: Path) -> None:
    """The generated `bioMod` should load and expose the expected DoF count."""

    biomod_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    model = biorbd.Model(str(biomod_path))

    assert model.nbQ() == 10
    assert model.nbQdot() == 10
    assert model.nbRoot() == 6
    assert model.nbMarkers() >= 10


def test_elevation_signs_raise_each_arm_outward_from_the_body(tmp_path: Path) -> None:
    """Negative left and positive right elevation should produce mirrored outward arm poses."""

    biomod_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    model = biorbd.Model(str(biomod_path))
    marker_names = [name.to_string() for name in model.markerNames()]
    marker_index = {name: index for index, name in enumerate(marker_names)}

    q = np.zeros(model.nbQ())
    q[7] = -np.pi / 2.0
    q[9] = np.pi / 2.0
    markers = model.markers(biorbd.GeneralizedCoordinates(q))
    hand_left = markers[marker_index["hand_left"]].to_array()
    hand_right = markers[marker_index["hand_right"]].to_array()
    shoulder_left = markers[marker_index["shoulder_left"]].to_array()
    shoulder_right = markers[marker_index["shoulder_right"]].to_array()

    assert hand_left[0] > shoulder_left[0]
    assert hand_right[0] < shoulder_right[0]
    np.testing.assert_allclose(hand_left[2], hand_right[2], atol=1e-10)

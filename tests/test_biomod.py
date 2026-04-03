"""Tests for the reduced whole-body `bioMod` generator."""

from pathlib import Path

import biorbd

from best_tilting_plane.modeling.biomod import ReducedAerialBiomod


def test_generated_biomod_loads_in_biorbd(tmp_path: Path) -> None:
    """The generated `bioMod` should load and expose the expected DoF count."""

    biomod_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    model = biorbd.Model(str(biomod_path))

    assert model.nbQ() == 10
    assert model.nbQdot() == 10
    assert model.nbRoot() == 6
    assert model.nbMarkers() >= 10

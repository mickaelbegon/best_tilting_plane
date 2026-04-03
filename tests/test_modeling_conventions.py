"""Tests for the root DoF conventions of the aerial model."""

from best_tilting_plane.modeling import (
    ROOT_ROTATION_DOF_NAMES,
    ROOT_ROTATION_SEQUENCE,
    ROOT_TRANSLATION_DOF_NAMES,
    root_twist_index,
)


def test_root_translation_and_rotation_order_matches_project_convention() -> None:
    """The root must follow translation then somersault/tilt/twist rotations."""

    assert ROOT_TRANSLATION_DOF_NAMES == ("pelvis_tx", "pelvis_ty", "pelvis_tz")
    assert ROOT_ROTATION_DOF_NAMES == ("somersault", "tilt", "twist")
    assert ROOT_ROTATION_SEQUENCE == "xyz"
    assert root_twist_index() == 2

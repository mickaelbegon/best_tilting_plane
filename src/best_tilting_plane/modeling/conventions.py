"""Kinematic conventions shared by the reduced whole-body model and simulation."""

ROOT_TRANSLATION_DOF_NAMES = ("pelvis_tx", "pelvis_ty", "pelvis_tz")
ROOT_ROTATION_DOF_NAMES = ("somersault", "tilt", "twist")
ROOT_ROTATION_SEQUENCE = "xyz"


def root_twist_index() -> int:
    """Return the index of the root twist DoF inside the root rotation block."""

    return ROOT_ROTATION_DOF_NAMES.index("twist")

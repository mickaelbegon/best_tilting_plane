"""Kinematic conventions shared by the reduced whole-body model and simulation."""

ROOT_TRANSLATION_DOF_NAMES = ("pelvis_tx", "pelvis_ty", "pelvis_tz")
ROOT_ROTATION_DOF_NAMES = ("somersault", "tilt", "twist")
ROOT_ROTATION_SEQUENCE = "xyz"
GLOBAL_AXIS_LABELS = ("x_mediolateral", "y_anteroposterior", "z_longitudinal")
ARM_PLANE_SEQUENCE = ("z",)
ARM_ELEVATION_SEQUENCE = ("y",)
ARM_SEGMENTS_FOR_VISUALIZATION = ("pelvis", "upper_arm_left", "upper_arm_right")


def root_twist_index() -> int:
    """Return the index of the root twist DoF inside the root rotation block."""

    return ROOT_ROTATION_DOF_NAMES.index("twist")

"""Modeling conventions for the floating-base aerial model."""

from .biomod import ReducedAerialBiomod
from .conventions import (
    ARM_ELEVATION_SEQUENCE,
    ARM_PLANE_SEQUENCE,
    ARM_SEGMENTS_FOR_VISUALIZATION,
    GLOBAL_AXIS_LABELS,
    ROOT_ROTATION_DOF_NAMES,
    ROOT_ROTATION_SEQUENCE,
    ROOT_TRANSLATION_DOF_NAMES,
    root_twist_index,
)
from .segments import BodyDimensions

__all__ = [
    "BodyDimensions",
    "ReducedAerialBiomod",
    "ARM_ELEVATION_SEQUENCE",
    "ARM_PLANE_SEQUENCE",
    "ARM_SEGMENTS_FOR_VISUALIZATION",
    "GLOBAL_AXIS_LABELS",
    "ROOT_ROTATION_DOF_NAMES",
    "ROOT_ROTATION_SEQUENCE",
    "ROOT_TRANSLATION_DOF_NAMES",
    "root_twist_index",
]

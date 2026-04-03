"""Modeling conventions for the floating-base aerial model."""

from .biomod import ReducedAerialBiomod
from .conventions import (
    ROOT_ROTATION_DOF_NAMES,
    ROOT_ROTATION_SEQUENCE,
    ROOT_TRANSLATION_DOF_NAMES,
    root_twist_index,
)
from .segments import BodyDimensions

__all__ = [
    "BodyDimensions",
    "ReducedAerialBiomod",
    "ROOT_ROTATION_DOF_NAMES",
    "ROOT_ROTATION_SEQUENCE",
    "ROOT_TRANSLATION_DOF_NAMES",
    "root_twist_index",
]

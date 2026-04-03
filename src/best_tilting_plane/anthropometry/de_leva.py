"""Subset of De Leva anthropometric coefficients used to build the reduced whole-body model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeLevaSex(Enum):
    """Sex-specific coefficient set available in the De Leva table."""

    MALE = "male"
    FEMALE = "female"


@dataclass(frozen=True)
class SegmentInertialParameters:
    """Mass fraction and normalized inertial coefficients for one segment."""

    mass_fraction: float
    center_of_mass_fraction: float
    radii_of_gyration: tuple[float, float, float]


_DE_LEVA_TABLE = {
    DeLevaSex.MALE: {
        "head": SegmentInertialParameters(0.0694, 0.5002, (0.303, 0.315, 0.261)),
        "trunk": SegmentInertialParameters(0.4346, 0.5138, (0.328, 0.306, 0.169)),
        "upper_arm": SegmentInertialParameters(0.0271, 0.4228, (0.285, 0.269, 0.158)),
        "forearm": SegmentInertialParameters(0.0162, 0.5426, (0.276, 0.265, 0.121)),
        "hand": SegmentInertialParameters(0.0061, 0.3624, (0.288, 0.235, 0.184)),
        "thigh": SegmentInertialParameters(0.1416, 0.4095, (0.329, 0.329, 0.149)),
        "shank": SegmentInertialParameters(0.0433, 0.4459, (0.255, 0.249, 0.103)),
        "foot": SegmentInertialParameters(0.0137, 0.4415, (0.257, 0.245, 0.124)),
    },
    DeLevaSex.FEMALE: {
        "head": SegmentInertialParameters(0.0669, 0.4841, (0.271, 0.295, 0.261)),
        "trunk": SegmentInertialParameters(0.4257, 0.4964, (0.307, 0.292, 0.147)),
        "upper_arm": SegmentInertialParameters(0.0255, 0.4246, (0.278, 0.260, 0.148)),
        "forearm": SegmentInertialParameters(0.0138, 0.5441, (0.261, 0.257, 0.094)),
        "hand": SegmentInertialParameters(0.0056, 0.3427, (0.244, 0.208, 0.184)),
        "thigh": SegmentInertialParameters(0.1478, 0.3612, (0.369, 0.364, 0.162)),
        "shank": SegmentInertialParameters(0.0481, 0.4416, (0.271, 0.267, 0.093)),
        "foot": SegmentInertialParameters(0.0129, 0.4014, (0.299, 0.279, 0.124)),
    },
}


def de_leva_segment_table(sex: DeLevaSex = DeLevaSex.MALE) -> dict[str, SegmentInertialParameters]:
    """Return the De Leva coefficients for the requested sex."""

    return _DE_LEVA_TABLE[sex].copy()


def total_body_fraction(
    sex: DeLevaSex = DeLevaSex.MALE,
    *,
    bilateral_segments: tuple[str, ...] = (
        "upper_arm",
        "forearm",
        "hand",
        "thigh",
        "shank",
        "foot",
    ),
) -> float:
    """Return the total mass fraction represented by the reduced whole-body segment set."""

    table = de_leva_segment_table(sex)
    unilateral = {"head", "trunk"}
    unilateral_fraction = sum(table[name].mass_fraction for name in unilateral)
    bilateral_fraction = sum(table[name].mass_fraction for name in bilateral_segments)
    return unilateral_fraction + 2.0 * bilateral_fraction

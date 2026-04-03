"""Reduced full-body segment definitions used to generate the aerial `bioMod`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BodyDimensions:
    """Nominal body dimensions expressed in meters."""

    height: float = 1.75
    shoulder_half_width: float = 0.19
    hip_half_width: float = 0.14
    trunk_length: float = 0.62
    head_length: float = 0.24
    upper_arm_length: float = 0.29
    forearm_length: float = 0.26
    hand_length: float = 0.19
    thigh_length: float = 0.42
    shank_length: float = 0.43
    foot_length: float = 0.25

    @classmethod
    def from_height(cls, height: float) -> "BodyDimensions":
        """Build a coherent reduced set of dimensions from body height."""

        return cls(
            height=height,
            shoulder_half_width=0.1086 * height,
            hip_half_width=0.0800 * height,
            trunk_length=0.3543 * height,
            head_length=0.1371 * height,
            upper_arm_length=0.1860 * height,
            forearm_length=0.1463 * height,
            hand_length=0.1086 * height,
            thigh_length=0.2457 * height,
            shank_length=0.2514 * height,
            foot_length=0.1430 * height,
        )

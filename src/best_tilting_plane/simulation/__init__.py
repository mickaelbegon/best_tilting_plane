"""Simulation configuration and prescribed-motion helpers."""

from .arm_motion import ArmJointKinematics, PrescribedArmMotion, TwistOptimizationVariables
from .dynamics import (
    AerialSimulationResult,
    IntegratorSelection,
    PredictiveAerialTwistSimulator,
    SimulationConfiguration,
)
from .jerk_motion import (
    approximate_first_arm_elevation_motion,
    PiecewiseConstantJerkArmMotion,
    PiecewiseConstantJerkTrajectory,
    approximate_first_arm_plane_motion,
    approximate_quintic_segment_with_piecewise_constant_jerk,
    create_first_arm_piecewise_constant_comparison_figure,
    first_arm_piecewise_constant_comparison_data,
    show_first_arm_piecewise_constant_comparison,
)

__all__ = [
    "AerialSimulationResult",
    "ArmJointKinematics",
    "IntegratorSelection",
    "PiecewiseConstantJerkArmMotion",
    "PiecewiseConstantJerkTrajectory",
    "PredictiveAerialTwistSimulator",
    "PrescribedArmMotion",
    "SimulationConfiguration",
    "TwistOptimizationVariables",
    "approximate_first_arm_elevation_motion",
    "approximate_first_arm_plane_motion",
    "approximate_quintic_segment_with_piecewise_constant_jerk",
    "create_first_arm_piecewise_constant_comparison_figure",
    "first_arm_piecewise_constant_comparison_data",
    "show_first_arm_piecewise_constant_comparison",
]

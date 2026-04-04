"""Prescribed arm kinematics used by the predictive aerial simulation."""

from __future__ import annotations

from dataclasses import dataclass

from best_tilting_plane.trajectories import QuinticBoundaryTrajectory


@dataclass(frozen=True)
class TwistOptimizationVariables:
    """Decision variables that define the asymmetric arm strategy.

    All angles are expressed in radians and times in seconds.
    """

    right_arm_start: float
    left_plane_initial: float
    left_plane_final: float
    right_plane_initial: float
    right_plane_final: float


@dataclass(frozen=True)
class ArmJointKinematics:
    """One joint state with position, velocity, and acceleration."""

    position: float
    velocity: float
    acceleration: float


@dataclass(frozen=True)
class ArmKinematics:
    """State of one arm for the elevation plane and elevation joints."""

    elevation_plane: ArmJointKinematics
    elevation: ArmJointKinematics


class PrescribedArmMotion:
    """Generate prescribed left/right arm kinematics from the optimization variables.

    The left arm always moves from 0.0 s to 0.3 s.
    The right arm starts at the decision-variable time and lasts 0.3 s.
    Elevation is imposed from -180 to 0 degrees on the left arm and from +180 to 0 degrees
    on the right arm so that the sign convention matches the biomod.
    """

    def __init__(
        self,
        variables: TwistOptimizationVariables,
        *,
        left_start: float = 0.0,
        duration: float = 0.3,
        left_elevation_initial: float = -3.141592653589793,
        left_elevation_final: float = 0.0,
        right_elevation_initial: float = 3.141592653589793,
        right_elevation_final: float = 0.0,
    ) -> None:
        """Build the prescribed left/right arm trajectories."""

        if duration <= 0.0:
            raise ValueError("duration must be strictly positive.")

        self.variables = variables
        self.left_start = left_start
        self.duration = duration
        self.left_elevation_initial = left_elevation_initial
        self.left_elevation_final = left_elevation_final
        self.right_elevation_initial = right_elevation_initial
        self.right_elevation_final = right_elevation_final

        self._left_plane = QuinticBoundaryTrajectory(
            t0=left_start,
            t1=left_start + duration,
            q0=variables.left_plane_initial,
            q1=variables.left_plane_final,
        )
        self._left_elevation = QuinticBoundaryTrajectory(
            t0=left_start,
            t1=left_start + duration,
            q0=left_elevation_initial,
            q1=left_elevation_final,
        )
        self._right_plane = QuinticBoundaryTrajectory(
            t0=variables.right_arm_start,
            t1=variables.right_arm_start + duration,
            q0=variables.right_plane_initial,
            q1=variables.right_plane_final,
        )
        self._right_elevation = QuinticBoundaryTrajectory(
            t0=variables.right_arm_start,
            t1=variables.right_arm_start + duration,
            q0=right_elevation_initial,
            q1=right_elevation_final,
        )

    @property
    def left_end(self) -> float:
        """Return the end time of the left arm motion."""

        return self.left_start + self.duration

    @property
    def right_end(self) -> float:
        """Return the end time of the right arm motion."""

        return self.variables.right_arm_start + self.duration

    def left(self, time: float) -> ArmKinematics:
        """Return the left-arm kinematics at a given time."""

        return ArmKinematics(
            elevation_plane=ArmJointKinematics(
                position=float(self._left_plane.position(time)),
                velocity=float(self._left_plane.velocity(time)),
                acceleration=float(self._left_plane.acceleration(time)),
            ),
            elevation=ArmJointKinematics(
                position=float(self._left_elevation.position(time)),
                velocity=float(self._left_elevation.velocity(time)),
                acceleration=float(self._left_elevation.acceleration(time)),
            ),
        )

    def right(self, time: float) -> ArmKinematics:
        """Return the right-arm kinematics at a given time."""

        return ArmKinematics(
            elevation_plane=ArmJointKinematics(
                position=float(self._right_plane.position(time)),
                velocity=float(self._right_plane.velocity(time)),
                acceleration=float(self._right_plane.acceleration(time)),
            ),
            elevation=ArmJointKinematics(
                position=float(self._right_elevation.position(time)),
                velocity=float(self._right_elevation.velocity(time)),
                acceleration=float(self._right_elevation.acceleration(time)),
            ),
        )

"""Piecewise-constant-jerk arm kinematics and comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from best_tilting_plane.simulation.arm_motion import (
    ArmJointKinematics,
    ArmKinematics,
    PrescribedArmMotion,
    TwistOptimizationVariables,
)
from best_tilting_plane.trajectories import QuinticBoundaryTrajectory
from best_tilting_plane.visualization.external_figure import present_external_figure

DEFAULT_ARM_MOTION_DURATION = 0.3


def _advance_constant_acceleration(
    q: float,
    qdot: float,
    qddot: float,
    duration: float,
) -> tuple[float, float, float]:
    """Advance one scalar state over a zero-jerk interval."""

    return (
        q + duration * qdot + 0.5 * duration * duration * qddot,
        qdot + duration * qddot,
        qddot,
    )


def _advance_constant_jerk(
    q: float,
    qdot: float,
    qddot: float,
    jerk: float,
    duration: float,
) -> tuple[float, float, float]:
    """Advance one scalar state over a constant-jerk interval."""

    duration2 = duration * duration
    duration3 = duration2 * duration
    return (
        q + duration * qdot + 0.5 * duration2 * qddot + duration3 * jerk / 6.0,
        qdot + duration * qddot + 0.5 * duration2 * jerk,
        qddot + duration * jerk,
    )


@dataclass(frozen=True)
class PiecewiseConstantJerkTrajectory:
    """One scalar trajectory driven by a piecewise-constant jerk control."""

    q0: float
    qdot0: float
    qddot0: float
    step: float
    jerks: np.ndarray
    active_start: float
    active_end: float
    total_duration: float | None = None

    def __post_init__(self) -> None:
        """Validate the discretization inputs."""

        if self.step <= 0.0:
            raise ValueError("The jerk discretization step must be strictly positive.")
        if self.active_end < self.active_start:
            raise ValueError("The active jerk window must satisfy `active_end >= active_start`.")
        if self.total_duration is not None and self.total_duration <= 0.0:
            raise ValueError("The total duration must be strictly positive.")
        if self.total_duration is not None and self.total_duration + 1e-12 < self.control_duration:
            raise ValueError("The total duration cannot be smaller than the jerk-control horizon.")

    @property
    def control_duration(self) -> float:
        """Return the duration covered by the jerk-control grid."""

        return float(len(self.jerks) * self.step)

    @property
    def duration(self) -> float:
        """Return the total duration covered by the discrete control grid."""

        if self.total_duration is not None:
            return float(self.total_duration)
        return self.control_duration

    def state(self, time: float) -> tuple[float, float, float]:
        """Return position, velocity, and acceleration at the requested time."""

        clipped_time = float(np.clip(time, 0.0, self.duration))
        q = float(self.q0)
        qdot = float(self.qdot0)
        qddot = float(self.qddot0)
        current_time = 0.0

        for jerk in np.asarray(self.jerks, dtype=float):
            interval_end = current_time + self.step
            evaluation_end = min(clipped_time, interval_end)
            if evaluation_end <= current_time:
                break
            q, qdot, qddot = self._advance_over_interval(
                q,
                qdot,
                qddot,
                float(jerk),
                current_time,
                evaluation_end,
            )
            current_time = interval_end
            if clipped_time <= interval_end:
                break
        if clipped_time > current_time:
            q, qdot, qddot = _advance_constant_acceleration(
                q,
                qdot,
                qddot,
                clipped_time - current_time,
            )
        return q, qdot, qddot

    def position(self, time: float | np.ndarray) -> np.ndarray | float:
        """Return the trajectory position."""

        return self._evaluate_component(time, component_index=0)

    def velocity(self, time: float | np.ndarray) -> np.ndarray | float:
        """Return the trajectory velocity."""

        return self._evaluate_component(time, component_index=1)

    def acceleration(self, time: float | np.ndarray) -> np.ndarray | float:
        """Return the trajectory acceleration."""

        return self._evaluate_component(time, component_index=2)

    def node_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return position, velocity, and acceleration at all grid nodes."""

        node_times = np.asarray(
            sorted(
                {
                    *np.arange(0.0, self.duration + 0.5 * self.step, self.step, dtype=float).tolist(),
                    float(self.duration),
                }
            ),
            dtype=float,
        )
        positions = np.zeros(node_times.shape[0], dtype=float)
        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)
        for index, time in enumerate(node_times):
            positions[index], velocities[index], accelerations[index] = self.state(float(time))
        return positions, velocities, accelerations

    def _evaluate_component(self, time: float | np.ndarray, *, component_index: int) -> np.ndarray | float:
        """Evaluate one state component for scalar or vectorized input times."""

        if np.isscalar(time):
            return self.state(float(time))[component_index]
        values = np.asarray(time, dtype=float)
        return np.array(
            [self.state(float(sample))[component_index] for sample in values],
            dtype=float,
        )

    def _advance_over_interval(
        self,
        q: float,
        qdot: float,
        qddot: float,
        jerk: float,
        interval_start: float,
        evaluation_end: float,
    ) -> tuple[float, float, float]:
        """Advance the state over one grid interval with optional activation trimming."""

        overlap_start = min(max(interval_start, self.active_start), evaluation_end)
        overlap_end = min(max(interval_start, self.active_end), evaluation_end)
        if overlap_start > interval_start:
            q, qdot, qddot = _advance_constant_acceleration(
                q,
                qdot,
                qddot,
                overlap_start - interval_start,
            )
        if overlap_end > overlap_start:
            q, qdot, qddot = _advance_constant_jerk(
                q,
                qdot,
                qddot,
                jerk,
                overlap_end - overlap_start,
            )
        if evaluation_end > overlap_end:
            q, qdot, qddot = _advance_constant_acceleration(
                q,
                qdot,
                qddot,
                evaluation_end - overlap_end,
            )
        return q, qdot, qddot


class PiecewiseConstantJerkArmMotion:
    """Prescribed arm motion driven by piecewise-constant jerks on the arm planes."""

    def __init__(
        self,
        left_plane: PiecewiseConstantJerkTrajectory,
        right_plane: PiecewiseConstantJerkTrajectory,
        *,
        left_arm_start: float = 0.0,
        right_arm_start: float = 0.0,
        left_elevation: PiecewiseConstantJerkTrajectory | None = None,
        right_elevation: PiecewiseConstantJerkTrajectory | None = None,
        left_elevation_initial: float = -np.pi,
        left_elevation_final: float = 0.0,
        right_elevation_initial: float = np.pi,
        right_elevation_final: float = 0.0,
        duration: float = DEFAULT_ARM_MOTION_DURATION,
    ) -> None:
        """Store the plane trajectories and keep jerk-driven elevation profiles."""

        self.left_plane = left_plane
        self.right_plane = right_plane
        self.left_arm_start = float(left_arm_start)
        self.right_arm_start = float(right_arm_start)
        self.duration = float(duration)
        self.left_elevation = (
            left_elevation
            if left_elevation is not None
            else approximate_quintic_segment_with_piecewise_constant_jerk(
                total_time=self.left_plane.duration,
                step=self.left_plane.step,
                active_start=0.0,
                active_duration=self.duration,
                q0=left_elevation_initial,
                q1=left_elevation_final,
            )
        )
        self.right_elevation = (
            right_elevation
            if right_elevation is not None
            else approximate_quintic_segment_with_piecewise_constant_jerk(
                total_time=self.right_plane.duration,
                step=self.right_plane.step,
                active_start=0.0,
                active_duration=self.duration,
                q0=right_elevation_initial,
                q1=right_elevation_final,
            )
        )

    def left(self, time: float) -> ArmKinematics:
        """Return the left-arm kinematics at a given time."""

        local_time = float(time) - self.left_arm_start
        return ArmKinematics(
            elevation_plane=ArmJointKinematics(
                position=float(self.left_plane.position(local_time)),
                velocity=float(self.left_plane.velocity(local_time)),
                acceleration=float(self.left_plane.acceleration(local_time)),
            ),
            elevation=ArmJointKinematics(
                position=float(self.left_elevation.position(local_time)),
                velocity=float(self.left_elevation.velocity(local_time)),
                acceleration=float(self.left_elevation.acceleration(local_time)),
            ),
        )

    def right(self, time: float) -> ArmKinematics:
        """Return the right-arm kinematics at a given time."""

        local_time = float(time) - self.right_arm_start
        return ArmKinematics(
            elevation_plane=ArmJointKinematics(
                position=float(self.right_plane.position(local_time)),
                velocity=float(self.right_plane.velocity(local_time)),
                acceleration=float(self.right_plane.acceleration(local_time)),
            ),
            elevation=ArmJointKinematics(
                position=float(self.right_elevation.position(local_time)),
                velocity=float(self.right_elevation.velocity(local_time)),
                acceleration=float(self.right_elevation.acceleration(local_time)),
            ),
        )


def build_piecewise_constant_jerk_arm_motion(
    variables: TwistOptimizationVariables,
    *,
    total_time: float = 1.0,
    step: float = 0.02,
    duration: float = DEFAULT_ARM_MOTION_DURATION,
    first_arm_start: float | None = None,
) -> PiecewiseConstantJerkArmMotion:
    """Build the default jerk-driven left/right arm motion used across the project."""

    if first_arm_start is None:
        first_arm_start = float(getattr(variables, "first_arm_start", 0.0))
    else:
        first_arm_start = float(first_arm_start)
    left_total_time = max(float(step), float(total_time - variables.right_arm_start))
    right_total_time = max(float(step), float(total_time - first_arm_start))
    left_plane = approximate_quintic_segment_with_piecewise_constant_jerk(
        total_time=left_total_time,
        step=step,
        active_start=0.0,
        active_duration=duration,
        q0=variables.left_plane_initial,
        q1=variables.left_plane_final,
    )
    right_plane = approximate_quintic_segment_with_piecewise_constant_jerk(
        total_time=right_total_time,
        step=step,
        active_start=0.0,
        active_duration=duration,
        q0=variables.right_plane_initial,
        q1=variables.right_plane_final,
    )
    left_elevation = approximate_quintic_segment_with_piecewise_constant_jerk(
        total_time=left_total_time,
        step=step,
        active_start=0.0,
        active_duration=duration,
        q0=-np.pi,
        q1=0.0,
    )
    right_elevation = approximate_quintic_segment_with_piecewise_constant_jerk(
        total_time=right_total_time,
        step=step,
        active_start=0.0,
        active_duration=duration,
        q0=np.pi,
        q1=0.0,
    )
    return PiecewiseConstantJerkArmMotion(
        left_plane=left_plane,
        right_plane=right_plane,
        left_elevation=left_elevation,
        right_elevation=right_elevation,
        left_arm_start=variables.right_arm_start,
        right_arm_start=first_arm_start,
        duration=duration,
    )


def approximate_quintic_segment_with_piecewise_constant_jerk(
    *,
    total_time: float,
    step: float,
    active_start: float,
    active_duration: float,
    q0: float,
    q1: float,
) -> PiecewiseConstantJerkTrajectory:
    """Approximate one quintic segment with a piecewise-constant jerk control."""

    interval_count = int(round(total_time / step))
    if abs(interval_count * step - total_time) > 1e-12:
        raise ValueError("`total_time` must be a multiple of `step`.")

    jerks = np.zeros(interval_count, dtype=float)
    quintic = QuinticBoundaryTrajectory(
        t0=float(active_start),
        t1=float(active_start + active_duration),
        q0=float(q0),
        q1=float(q1),
    )
    q = float(q0)
    qdot = 0.0
    qddot = 0.0
    current_time = 0.0
    active_end = float(active_start + active_duration)

    for interval_index in range(interval_count):
        interval_end = current_time + step
        overlap_start = max(current_time, active_start)
        overlap_end = min(interval_end, active_end)
        jerk = 0.0
        if overlap_end > overlap_start:
            target_acceleration = float(quintic.acceleration(overlap_end))
            jerk = (target_acceleration - qddot) / (overlap_end - overlap_start)
        jerks[interval_index] = jerk
        local_active_start = min(step, max(0.0, overlap_start - current_time))
        local_active_end = min(step, max(0.0, overlap_end - current_time))
        local_active_end = max(local_active_start, local_active_end)
        q, qdot, qddot = PiecewiseConstantJerkTrajectory(
            q0=q,
            qdot0=qdot,
            qddot0=qddot,
            step=step,
            jerks=np.array([jerk], dtype=float),
            active_start=local_active_start,
            active_end=local_active_end,
        ).state(step)
        current_time = interval_end

    return PiecewiseConstantJerkTrajectory(
        q0=float(q0),
        qdot0=0.0,
        qddot0=0.0,
        step=step,
        jerks=jerks,
        active_start=float(active_start),
        active_end=active_end,
    )


def approximate_first_arm_elevation_motion(
    variables: TwistOptimizationVariables | None = None,
    *,
    total_time: float = 1.0,
    step: float = 0.02,
) -> PiecewiseConstantJerkTrajectory:
    """Approximate the first-arm elevation motion with a piecewise-constant jerk control."""

    return approximate_quintic_segment_with_piecewise_constant_jerk(
        total_time=total_time,
        step=step,
        active_start=0.0,
        active_duration=0.3,
        q0=np.pi,
        q1=0.0,
    )


def approximate_first_arm_plane_motion(
    variables: TwistOptimizationVariables,
    *,
    total_time: float = 1.0,
    step: float = 0.02,
) -> PiecewiseConstantJerkTrajectory:
    """Backward-compatible alias kept for callers that still use the old helper name."""

    return approximate_first_arm_elevation_motion(
        variables,
        total_time=total_time,
        step=step,
    )


def first_arm_piecewise_constant_comparison_data(
    variables: TwistOptimizationVariables,
    *,
    total_time: float = 1.0,
    jerk_step: float = 0.02,
    sample_step: float = 0.005,
) -> dict[str, np.ndarray]:
    """Return the reference and piecewise-constant-jerk first-arm elevation trajectories."""

    motion = PrescribedArmMotion(variables)
    approximation = approximate_first_arm_elevation_motion(
        variables,
        total_time=total_time,
        step=jerk_step,
    )
    samples = np.arange(0.0, total_time + 0.5 * sample_step, sample_step, dtype=float)
    return {
        "time": samples,
        "reference_q": np.array(
            [motion.right(float(sample)).elevation.position for sample in samples],
            dtype=float,
        ),
        "reference_qdot": np.array(
            [motion.right(float(sample)).elevation.velocity for sample in samples],
            dtype=float,
        ),
        "reference_qddot": np.array(
            [motion.right(float(sample)).elevation.acceleration for sample in samples],
            dtype=float,
        ),
        "approximate_q": approximation.position(samples),
        "approximate_qdot": approximation.velocity(samples),
        "approximate_qddot": approximation.acceleration(samples),
        "jerk_nodes": np.asarray(approximation.jerks, dtype=float),
        "jerk_time": np.arange(0.0, total_time, jerk_step, dtype=float),
    }


def create_first_arm_piecewise_constant_comparison_figure(
    variables: TwistOptimizationVariables,
    *,
    total_time: float = 1.0,
    jerk_step: float = 0.02,
    sample_step: float = 0.005,
):
    """Create a comparison figure between quintic and piecewise-constant-jerk kinematics."""

    import matplotlib.pyplot as plt

    data = first_arm_piecewise_constant_comparison_data(
        variables,
        total_time=total_time,
        jerk_step=jerk_step,
        sample_step=sample_step,
    )
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 8.0), tight_layout=True)
    for axis, reference_key, approximate_key, ylabel in (
        (axes[0], "reference_q", "approximate_q", "q (rad)"),
        (axes[1], "reference_qdot", "approximate_qdot", "qdot (rad/s)"),
        (axes[2], "reference_qddot", "approximate_qddot", "qddot (rad/s2)"),
    ):
        axis.plot(data["time"], data[reference_key], color="black", linewidth=2.0, label="Quintique")
        axis.plot(
            data["time"],
            data[approximate_key],
            color="tab:orange",
            linewidth=1.8,
            linestyle="--",
            label="Jerk piecewise constant",
        )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Temps (s)")
    axes[0].set_title("Comparaison bras 1: quintique vs jerk piecewise constant")
    return figure, axes, data


def show_first_arm_piecewise_constant_comparison(
    variables: TwistOptimizationVariables,
    *,
    total_time: float = 1.0,
    jerk_step: float = 0.02,
    sample_step: float = 0.005,
):
    """Open an external matplotlib window comparing the first-arm elevation kinematics."""

    figure, axes, data = create_first_arm_piecewise_constant_comparison_figure(
        variables,
        total_time=total_time,
        jerk_step=jerk_step,
        sample_step=sample_step,
    )
    jerk_axis = axes[-1].twinx()
    jerk_axis.step(
        data["jerk_time"],
        data["jerk_nodes"],
        where="post",
        color="tab:blue",
        linewidth=1.2,
        alpha=0.55,
        label="Jerk discretise",
    )
    jerk_axis.set_ylabel("Jerk (rad/s3)")
    present_external_figure(figure)
    return figure, axes, data

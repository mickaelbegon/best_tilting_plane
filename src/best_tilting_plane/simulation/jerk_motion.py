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

    def __post_init__(self) -> None:
        """Validate the discretization inputs."""

        if self.step <= 0.0:
            raise ValueError("The jerk discretization step must be strictly positive.")
        if self.active_end < self.active_start:
            raise ValueError("The active jerk window must satisfy `active_end >= active_start`.")

    @property
    def duration(self) -> float:
        """Return the total duration covered by the discrete control grid."""

        return float(len(self.jerks) * self.step)

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

        node_times = np.linspace(0.0, self.duration, len(self.jerks) + 1)
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
        right_arm_start: float,
        left_elevation_initial: float = -np.pi,
        left_elevation_final: float = 0.0,
        right_elevation_initial: float = np.pi,
        right_elevation_final: float = 0.0,
        duration: float = 0.3,
    ) -> None:
        """Store the plane trajectories and keep the fixed-elevation quintics."""

        self.left_plane = left_plane
        self.right_plane = right_plane
        self.right_arm_start = float(right_arm_start)
        self.duration = float(duration)
        self._left_elevation = QuinticBoundaryTrajectory(
            t0=0.0,
            t1=self.duration,
            q0=left_elevation_initial,
            q1=left_elevation_final,
        )
        self._right_elevation = QuinticBoundaryTrajectory(
            t0=self.right_arm_start,
            t1=self.right_arm_start + self.duration,
            q0=right_elevation_initial,
            q1=right_elevation_final,
        )

    def left(self, time: float) -> ArmKinematics:
        """Return the left-arm kinematics at a given time."""

        return ArmKinematics(
            elevation_plane=ArmJointKinematics(
                position=float(self.left_plane.position(time)),
                velocity=float(self.left_plane.velocity(time)),
                acceleration=float(self.left_plane.acceleration(time)),
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
                position=float(self.right_plane.position(time)),
                velocity=float(self.right_plane.velocity(time)),
                acceleration=float(self.right_plane.acceleration(time)),
            ),
            elevation=ArmJointKinematics(
                position=float(self._right_elevation.position(time)),
                velocity=float(self._right_elevation.velocity(time)),
                acceleration=float(self._right_elevation.acceleration(time)),
            ),
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


def approximate_first_arm_plane_motion(
    variables: TwistOptimizationVariables,
    *,
    total_time: float = 1.0,
    step: float = 0.02,
) -> PiecewiseConstantJerkTrajectory:
    """Approximate the first-arm plane motion with a piecewise-constant jerk control."""

    return approximate_quintic_segment_with_piecewise_constant_jerk(
        total_time=total_time,
        step=step,
        active_start=0.0,
        active_duration=0.3,
        q0=float(variables.left_plane_initial),
        q1=float(variables.left_plane_final),
    )


def first_arm_piecewise_constant_comparison_data(
    variables: TwistOptimizationVariables,
    *,
    total_time: float = 1.0,
    jerk_step: float = 0.02,
    sample_step: float = 0.005,
) -> dict[str, np.ndarray]:
    """Return the reference and piecewise-constant-jerk first-arm plane trajectories."""

    motion = PrescribedArmMotion(variables)
    approximation = approximate_first_arm_plane_motion(
        variables,
        total_time=total_time,
        step=jerk_step,
    )
    samples = np.arange(0.0, total_time + 0.5 * sample_step, sample_step, dtype=float)
    return {
        "time": samples,
        "reference_q": np.array(
            [motion.left(float(sample)).elevation_plane.position for sample in samples],
            dtype=float,
        ),
        "reference_qdot": np.array(
            [motion.left(float(sample)).elevation_plane.velocity for sample in samples],
            dtype=float,
        ),
        "reference_qddot": np.array(
            [motion.left(float(sample)).elevation_plane.acceleration for sample in samples],
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
    """Open an external matplotlib window comparing the first-arm plane kinematics."""

    import matplotlib.pyplot as plt

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
    figure.canvas.draw_idle()
    plt.show(block=False)
    return figure, axes, data

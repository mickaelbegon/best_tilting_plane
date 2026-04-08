"""Classical jerk-driven direct multiple shooting for the reduced aerial model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import biorbd
import biorbd_casadi as biorbd_ca
import casadi as ca
import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.optimization.solver_options import (
    build_ipopt_solver_options,
    configure_optimization_threads,
)
from best_tilting_plane.simulation import (
    AerialSimulationResult,
    build_piecewise_constant_jerk_arm_motion,
    PiecewiseConstantJerkArmMotion,
    PiecewiseConstantJerkTrajectory,
    PredictiveAerialTwistSimulator,
    SimulationConfiguration,
    TwistOptimizationVariables,
    approximate_first_arm_elevation_motion,
    approximate_quintic_segment_with_piecewise_constant_jerk,
)

LEFT_ARM_ACTIVE_DURATION = 0.3
RIGHT_ARM_ACTIVE_DURATION = 0.3
RIGHT_ARM_START_BOUNDS = (0.0, 0.7)
RIGHT_ARM_SWEEP_BOUNDS = (0.0, 0.7)
MULTISTART_REFERENCE_T1 = 0.30
MULTISTART_START_COUNT = 5
OBJECTIVE_MODE_TWIST = "twist"
OBJECTIVE_MODE_TWIST_BTP = "twist_btp"
PLANE_STATE_SIZE = 3
ROOT_STATE_SIZE = 12
ELEVATION_STAGE_BLOCK_SIZE = 18
DEFAULT_DMS_JERK_REGULARIZATION = 1e-9
DEFAULT_DMS_BTP_DEVIATION_WEIGHT = 10.0
START_TIME_TOLERANCE = 1e-9
JERK_BOUND_SCALE = 4.0


def _result_is_better(candidate, reference) -> bool:
    """Return whether one DMS result should be preferred over another."""

    if reference is None:
        return True
    if bool(candidate.success) != bool(reference.success):
        return bool(candidate.success)
    return float(candidate.objective) < float(reference.objective)


@dataclass(frozen=True)
class DirectMultipleShootingResult:
    """Optimization result for one fixed second-arm start time."""

    variables: TwistOptimizationVariables
    right_arm_start_node_index: int
    left_plane_jerk: np.ndarray
    right_plane_jerk: np.ndarray
    node_times: np.ndarray
    arm_node_times: np.ndarray
    root_state_nodes: np.ndarray
    left_plane_state_nodes: np.ndarray
    right_plane_state_nodes: np.ndarray
    prescribed_motion: PiecewiseConstantJerkArmMotion
    simulation: AerialSimulationResult
    objective: float
    solver_status: str
    success: bool
    btp_deviation_lagrange: float = 0.0
    warm_start_primal: np.ndarray | None = field(default=None, repr=False)
    warm_start_lam_x: np.ndarray | None = field(default=None, repr=False)
    warm_start_lam_g: np.ndarray | None = field(default=None, repr=False)

    @property
    def final_twist_angle(self) -> float:
        """Return the final twist angle in radians."""

        return self.simulation.final_twist_angle

    @property
    def final_twist_turns(self) -> float:
        """Return the final twist count in turns."""

        return self.simulation.final_twist_turns


@dataclass(frozen=True)
class DirectMultipleShootingSweepResult:
    """Result of the discrete sweep over admissible second-arm start nodes."""

    best_result: DirectMultipleShootingResult
    candidate_results: tuple[DirectMultipleShootingResult, ...]

    @property
    def start_times(self) -> np.ndarray:
        """Return the scanned second-arm start times."""

        return np.array([result.variables.right_arm_start for result in self.candidate_results], dtype=float)

    @property
    def objective_values(self) -> np.ndarray:
        """Return the objective value associated with every scanned start time."""

        return np.array([result.objective for result in self.candidate_results], dtype=float)

    @property
    def final_twist_turns(self) -> np.ndarray:
        """Return the final twist count associated with every scanned start time."""

        return np.array([result.final_twist_turns for result in self.candidate_results], dtype=float)

    @property
    def success_mask(self) -> np.ndarray:
        """Return whether each candidate solve reached a successful NLP status."""

        return np.array([result.success for result in self.candidate_results], dtype=bool)

    @property
    def solver_statuses(self) -> tuple[str, ...]:
        """Return the solver status associated with every scanned start time."""

        return tuple(result.solver_status for result in self.candidate_results)


class DirectMultipleShootingOptimizer:
    """Classical jerk-driven multiple shooting with one global graph and fixed known values."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        configuration: SimulationConfiguration | None = None,
        shooting_step: float = 0.02,
        jerk_regularization: float = DEFAULT_DMS_JERK_REGULARIZATION,
        objective_mode: str = OBJECTIVE_MODE_TWIST,
        btp_deviation_weight: float = DEFAULT_DMS_BTP_DEVIATION_WEIGHT,
        model: biorbd.Model | None = None,
    ) -> None:
        """Store the models and the direct multiple-shooting settings."""

        self.model_path = str(model_path)
        self.configuration = configuration or SimulationConfiguration(integrator="rk4", rk4_step=0.005)
        self.shooting_step = float(shooting_step)
        self.jerk_regularization = float(jerk_regularization)
        self.objective_mode = str(objective_mode)
        self.btp_deviation_weight = float(btp_deviation_weight)
        self.model = model if model is not None else biorbd.Model(self.model_path)
        self.symbolic_model = biorbd_ca.Model(self.model_path)
        self._solver = None
        self._solver_options_key: tuple[int, int, bool, str, float] | None = None
        self._constraint_count = 0
        self._interval_parallelization = "serial"
        self._nlpsol_expand = True
        self._elevation_trajectory_cache: dict[
            float,
            tuple[PiecewiseConstantJerkTrajectory, PiecewiseConstantJerkTrajectory],
        ] = {}
        self._segment_index_by_name = self._build_segment_index_by_name()

        if self.shooting_step <= 0.0:
            raise ValueError("The shooting step must be strictly positive.")
        if self.objective_mode not in {OBJECTIVE_MODE_TWIST, OBJECTIVE_MODE_TWIST_BTP}:
            raise ValueError(f"Unsupported DMS objective mode '{self.objective_mode}'.")
        if self.btp_deviation_weight < 0.0:
            raise ValueError("The BTP deviation weight must be non-negative.")

        step_count = int(round(self.configuration.final_time / self.shooting_step))
        if abs(step_count * self.shooting_step - self.configuration.final_time) > 1e-12:
            raise ValueError("The final time must be a multiple of the shooting step.")
        active_control_count = int(round(LEFT_ARM_ACTIVE_DURATION / self.shooting_step))
        if abs(active_control_count * self.shooting_step - LEFT_ARM_ACTIVE_DURATION) > 1e-12:
            raise ValueError("The active arm duration must be a multiple of the shooting step.")

        self.interval_count = step_count
        self.active_control_count = active_control_count
        self.node_times = np.linspace(0.0, self.configuration.final_time, self.interval_count + 1)
        self.arm_node_times = np.linspace(0.0, LEFT_ARM_ACTIVE_DURATION, self.active_control_count + 1)
        elevation_fit = approximate_first_arm_elevation_motion(
            total_time=LEFT_ARM_ACTIVE_DURATION,
            step=self.shooting_step,
        )
        self.jerk_bound = float(JERK_BOUND_SCALE * np.max(np.abs(elevation_fit.jerks)))

    @classmethod
    def from_builder(
        cls,
        output_path: str | Path,
        *,
        model_builder: ReducedAerialBiomod | None = None,
        configuration: SimulationConfiguration | None = None,
        shooting_step: float = 0.02,
        jerk_regularization: float = DEFAULT_DMS_JERK_REGULARIZATION,
        objective_mode: str = OBJECTIVE_MODE_TWIST,
        btp_deviation_weight: float = DEFAULT_DMS_BTP_DEVIATION_WEIGHT,
    ) -> "DirectMultipleShootingOptimizer":
        """Generate the model file and build a DMS optimizer on top of it."""

        builder = model_builder or ReducedAerialBiomod()
        model_path = builder.write(output_path)
        return cls(
            model_path,
            configuration=configuration,
            shooting_step=shooting_step,
            jerk_regularization=jerk_regularization,
            objective_mode=objective_mode,
            btp_deviation_weight=btp_deviation_weight,
        )

    def _build_segment_index_by_name(self) -> dict[str, int]:
        """Return the segment-index map shared by the numeric and symbolic models."""

        index_by_name: dict[str, int] = {}
        for index in range(self.model.nbSegment()):
            segment = self.model.segment(index)
            name = segment.name().to_string() if hasattr(segment.name(), "to_string") else str(segment.name())
            index_by_name[name] = index
        return index_by_name

    @staticmethod
    def _quintic_profile(phase: float) -> tuple[float, float, float]:
        """Return the quintic profile and its first two derivatives."""

        phase2 = phase * phase
        phase3 = phase2 * phase
        phase4 = phase3 * phase
        phase5 = phase4 * phase
        profile = 6.0 * phase5 - 15.0 * phase4 + 10.0 * phase3
        velocity = 30.0 * phase4 - 60.0 * phase3 + 30.0 * phase2
        acceleration = 120.0 * phase3 - 180.0 * phase2 + 60.0 * phase
        return profile, velocity, acceleration

    @classmethod
    def _numeric_quintic_segment(
        cls,
        *,
        time: float,
        start: float,
        duration: float,
        q0: float,
        q1: float,
    ) -> tuple[float, float, float]:
        """Return one known quintic segment and its first two derivatives."""

        if time <= start:
            return q0, 0.0, 0.0
        if time >= start + duration:
            return q1, 0.0, 0.0
        phase = (time - start) / duration
        profile, velocity, acceleration = cls._quintic_profile(phase)
        delta = q1 - q0
        return (
            q0 + delta * profile,
            delta * velocity / duration,
            delta * acceleration / (duration**2),
        )

    def _fixed_elevation_kinematics(
        self,
        *,
        time: float,
        right_arm_start: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the known jerk-driven elevation kinematics at one fixed time."""

        left_trajectory, right_trajectory = self._fixed_elevation_trajectories(
            right_arm_start=right_arm_start
        )
        left_q, left_qdot, left_qddot = left_trajectory.state(time)
        right_local_time = max(0.0, float(time) - float(right_arm_start))
        right_q, right_qdot, right_qddot = right_trajectory.state(right_local_time)
        return (
            np.array([left_q, right_q], dtype=float),
            np.array([left_qdot, right_qdot], dtype=float),
            np.array([left_qddot, right_qddot], dtype=float),
        )

    def _fixed_elevation_trajectories(
        self,
        *,
        right_arm_start: float,
    ) -> tuple[PiecewiseConstantJerkTrajectory, PiecewiseConstantJerkTrajectory]:
        """Return the cached jerk-driven elevation trajectories for one fixed second-arm start."""

        cache_key = round(float(right_arm_start), 10)
        cached = self._elevation_trajectory_cache.get(cache_key)
        if cached is not None:
            return cached

        left_trajectory = approximate_quintic_segment_with_piecewise_constant_jerk(
            total_time=self.configuration.final_time,
            step=self.shooting_step,
            active_start=0.0,
            active_duration=LEFT_ARM_ACTIVE_DURATION,
            q0=-np.pi,
            q1=0.0,
        )
        right_trajectory = approximate_quintic_segment_with_piecewise_constant_jerk(
            total_time=self.configuration.final_time - float(right_arm_start),
            step=self.shooting_step,
            active_start=0.0,
            active_duration=RIGHT_ARM_ACTIVE_DURATION,
            q0=np.pi,
            q1=0.0,
        )
        self._elevation_trajectory_cache[cache_key] = (left_trajectory, right_trajectory)
        return left_trajectory, right_trajectory

    def _elevation_parameter_values(self, *, right_arm_start: float) -> np.ndarray:
        """Return the known elevation values injected into the root dynamics over the whole horizon."""

        values: list[float] = []
        for interval_index in range(self.interval_count):
            interval_start = float(self.node_times[interval_index])
            for stage_time in (
                interval_start,
                interval_start + 0.5 * self.shooting_step,
                interval_start + self.shooting_step,
            ):
                q, qdot, qddot = self._fixed_elevation_kinematics(
                    time=stage_time,
                    right_arm_start=right_arm_start,
                )
                values.extend(q.tolist())
                values.extend(qdot.tolist())
                values.extend(qddot.tolist())
        return np.asarray(values, dtype=float)

    def _symbolic_initial_root_state(self, variables: TwistOptimizationVariables) -> np.ndarray:
        """Return the initial root state."""

        q = np.zeros(self.model.nbQ(), dtype=float)
        qdot = np.zeros(self.model.nbQ(), dtype=float)
        q[6:] = (
            variables.left_plane_initial,
            -np.pi,
            variables.right_plane_initial,
            np.pi,
        )
        qdot[3] = self.configuration.somersault_rate
        qdot[5] = variables.contact_twist_rate
        qdot[:3] = -np.asarray(self.model.CoMdot(q, qdot, True).to_array(), dtype=float).reshape(3)
        return np.array(
            [
                q[0],
                q[1],
                q[2],
                q[3],
                q[4],
                q[5],
                qdot[0],
                qdot[1],
                qdot[2],
                qdot[3],
                qdot[4],
                qdot[5],
            ],
            dtype=float,
        )

    @staticmethod
    def _plane_state_vector(q_value: float) -> np.ndarray:
        """Return one `(q, qdot, qddot)` vector."""

        return np.array([q_value, 0.0, 0.0], dtype=float)

    @staticmethod
    def _advance_symbolic_constant_jerk(
        plane_state: ca.MX,
        jerk: ca.MX,
        duration: float,
    ) -> ca.MX:
        """Advance one plane state over a constant-jerk interval."""

        q, qdot, qddot = plane_state[0], plane_state[1], plane_state[2]
        return ca.vertcat(
            q + duration * qdot + 0.5 * duration * duration * qddot + duration**3 * jerk / 6.0,
            qdot + duration * qddot + 0.5 * duration * duration * jerk,
            qddot + duration * jerk,
        )

    def _symbolic_root_dynamics(
        self,
        root_state: ca.MX,
        left_plane_state: ca.MX,
        right_plane_state: ca.MX,
        elevation_q: ca.MX,
        elevation_qdot: ca.MX,
        elevation_qddot: ca.MX,
    ) -> ca.MX:
        """Return the time derivative of the root state."""

        q_root = root_state[:6]
        qdot_root = root_state[6:12]
        left_q, left_qdot, left_qddot = left_plane_state[0], left_plane_state[1], left_plane_state[2]
        right_q, right_qdot, right_qddot = right_plane_state[0], right_plane_state[1], right_plane_state[2]

        q_joint = ca.vertcat(left_q, elevation_q[0], right_q, elevation_q[1])
        qdot_joint = ca.vertcat(left_qdot, elevation_qdot[0], right_qdot, elevation_qdot[1])
        qddot_joint = ca.vertcat(left_qddot, elevation_qddot[0], right_qddot, elevation_qddot[1])

        q_full = ca.vertcat(q_root, q_joint)
        qdot_full = ca.vertcat(qdot_root, qdot_joint)
        qddot_root = self.symbolic_model.ForwardDynamicsFreeFloatingBase(
            q_full,
            qdot_full,
            qddot_joint,
        ).to_mx()
        return ca.vertcat(qdot_root, qddot_root)

    def _symbolic_segment_origin(self, q_full: ca.MX, segment_name: str) -> ca.MX:
        """Return the symbolic origin of one segment."""

        return self.symbolic_model.globalJCS(q_full, self._segment_index_by_name[segment_name]).to_mx()[:3, 3]

    @staticmethod
    def _symbolic_signed_plane_deviation(vector: ca.MX, somersault_angle: ca.MX) -> ca.MX:
        """Return the symbolic signed deviation angle relative to the BTP."""

        normal = ca.vertcat(0.0, -ca.cos(somersault_angle), -ca.sin(somersault_angle))
        vector_norm = ca.sqrt(ca.sumsqr(vector) + 1e-12)
        direction_unit = vector / vector_norm
        normal_component = ca.dot(direction_unit, normal)
        tangential_norm = ca.sqrt(ca.fmax(0.0, 1.0 - normal_component * normal_component) + 1e-12)
        return ca.atan2(normal_component, tangential_norm)

    def _symbolic_btp_deviation_cost(
        self,
        root_state: ca.MX,
        left_plane_state: ca.MX,
        right_plane_state: ca.MX,
        elevation_q: ca.MX,
    ) -> ca.MX:
        """Return the squared BTP arm-deviation cost at one state."""

        q_full = ca.vertcat(
            root_state[:6],
            ca.vertcat(
                left_plane_state[0],
                elevation_q[0],
                right_plane_state[0],
                elevation_q[1],
            ),
        )
        left_vector = self._symbolic_segment_origin(q_full, "forearm_left") - self._symbolic_segment_origin(
            q_full,
            "upper_arm_left",
        )
        right_vector = self._symbolic_segment_origin(q_full, "forearm_right") - self._symbolic_segment_origin(
            q_full,
            "upper_arm_right",
        )
        somersault_angle = root_state[3]
        left_deviation = self._symbolic_signed_plane_deviation(left_vector, somersault_angle)
        right_deviation = self._symbolic_signed_plane_deviation(right_vector, somersault_angle)
        return left_deviation * left_deviation + right_deviation * right_deviation

    def _integrate_root_interval(
        self,
        root_state: ca.MX,
        left_plane_state: ca.MX,
        left_jerk: ca.MX,
        right_plane_state: ca.MX,
        right_jerk: ca.MX,
        elevation_start_q: ca.MX,
        elevation_start_qdot: ca.MX,
        elevation_start_qddot: ca.MX,
        elevation_mid_q: ca.MX,
        elevation_mid_qdot: ca.MX,
        elevation_mid_qddot: ca.MX,
        elevation_end_q: ca.MX,
        elevation_end_qdot: ca.MX,
        elevation_end_qddot: ca.MX,
    ) -> ca.MX:
        """Integrate one root shooting interval with RK4."""

        dt = self.shooting_step
        left_mid = self._advance_symbolic_constant_jerk(left_plane_state, left_jerk, 0.5 * dt)
        left_end = self._advance_symbolic_constant_jerk(left_plane_state, left_jerk, dt)
        right_mid = self._advance_symbolic_constant_jerk(right_plane_state, right_jerk, 0.5 * dt)
        right_end = self._advance_symbolic_constant_jerk(right_plane_state, right_jerk, dt)

        k1 = self._symbolic_root_dynamics(
            root_state,
            left_plane_state,
            right_plane_state,
            elevation_start_q,
            elevation_start_qdot,
            elevation_start_qddot,
        )
        k2 = self._symbolic_root_dynamics(
            root_state + 0.5 * dt * k1,
            left_mid,
            right_mid,
            elevation_mid_q,
            elevation_mid_qdot,
            elevation_mid_qddot,
        )
        k3 = self._symbolic_root_dynamics(
            root_state + 0.5 * dt * k2,
            left_mid,
            right_mid,
            elevation_mid_q,
            elevation_mid_qdot,
            elevation_mid_qddot,
        )
        k4 = self._symbolic_root_dynamics(
            root_state + dt * k3,
            left_end,
            right_end,
            elevation_end_q,
            elevation_end_qdot,
            elevation_end_qddot,
        )
        return root_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _build_interval_defect_function(self) -> ca.Function:
        """Build one interval defect function that CasADi can map over all shooting nodes."""

        root_state = ca.MX.sym("xk", ROOT_STATE_SIZE, 1)
        root_state_next = ca.MX.sym("xk_next", ROOT_STATE_SIZE, 1)
        left_plane_state = ca.MX.sym("xlk", PLANE_STATE_SIZE, 1)
        left_plane_state_next = ca.MX.sym("xlk_next", PLANE_STATE_SIZE, 1)
        right_plane_state = ca.MX.sym("xrk", PLANE_STATE_SIZE, 1)
        right_plane_state_next = ca.MX.sym("xrk_next", PLANE_STATE_SIZE, 1)
        left_jerk = ca.MX.sym("ulk")
        right_jerk = ca.MX.sym("urk")
        elevation_block = ca.MX.sym("pk", ELEVATION_STAGE_BLOCK_SIZE, 1)

        left_plane_prediction = self._advance_symbolic_constant_jerk(
            left_plane_state,
            left_jerk,
            self.shooting_step,
        )
        right_plane_prediction = self._advance_symbolic_constant_jerk(
            right_plane_state,
            right_jerk,
            self.shooting_step,
        )
        root_state_prediction = self._integrate_root_interval(
            root_state,
            left_plane_state,
            left_jerk,
            right_plane_state,
            right_jerk,
            elevation_block[0:2],
            elevation_block[2:4],
            elevation_block[4:6],
            elevation_block[6:8],
            elevation_block[8:10],
            elevation_block[10:12],
            elevation_block[12:14],
            elevation_block[14:16],
            elevation_block[16:18],
        )
        defect = ca.vertcat(
            root_state_next - root_state_prediction,
            left_plane_state_next - left_plane_prediction,
            right_plane_state_next - right_plane_prediction,
        )
        running_cost = 0.5 * (
            self._symbolic_btp_deviation_cost(
                root_state,
                left_plane_state,
                right_plane_state,
                elevation_block[0:2],
            )
            + self._symbolic_btp_deviation_cost(
                root_state_prediction,
                left_plane_prediction,
                right_plane_prediction,
                elevation_block[12:14],
            )
        )
        return ca.Function(
            "dms_interval_defect",
            [
                root_state,
                root_state_next,
                left_plane_state,
                left_plane_state_next,
                right_plane_state,
                right_plane_state_next,
                left_jerk,
                right_jerk,
                elevation_block,
            ],
            [defect, running_cost],
        )

    def _initial_guess_motion(
        self,
        variables: TwistOptimizationVariables,
        *,
        right_arm_start: float | None = None,
    ) -> PiecewiseConstantJerkArmMotion:
        """Return a piecewise-constant-jerk motion used as a warm start for one fixed-start problem."""

        fixed_right_arm_start = variables.right_arm_start if right_arm_start is None else float(right_arm_start)
        return build_piecewise_constant_jerk_arm_motion(
            TwistOptimizationVariables(
                right_arm_start=fixed_right_arm_start,
                left_plane_initial=variables.left_plane_initial,
                left_plane_final=variables.left_plane_final,
                right_plane_initial=variables.right_plane_initial,
                right_plane_final=variables.right_plane_final,
                contact_twist_rate=variables.contact_twist_rate,
            ),
            total_time=self.configuration.final_time,
            step=self.shooting_step,
        )

    def _initial_guess_root_state_history(
        self,
        variables: TwistOptimizationVariables,
        motion: PiecewiseConstantJerkArmMotion,
    ) -> np.ndarray:
        """Return a warm-start root-state history sampled on the shooting grid."""

        simulation = PredictiveAerialTwistSimulator(
            self.model_path,
            motion,
            configuration=SimulationConfiguration(
                final_time=self.configuration.final_time,
                steps=self.interval_count + 1,
                integrator="rk4",
                rk4_step=self.shooting_step,
                somersault_rate=self.configuration.somersault_rate,
                contact_twist_rate=self.configuration.contact_twist_rate,
            ),
            model=self.model,
        ).simulate()
        states = np.zeros((ROOT_STATE_SIZE, self.interval_count + 1), dtype=float)
        states[:6, :] = simulation.q[:, :6].T
        states[6:12, :] = simulation.qdot[:, :6].T
        return states

    @staticmethod
    def _plane_state_history(trajectory: PiecewiseConstantJerkTrajectory, sample_times: np.ndarray) -> np.ndarray:
        """Return `(q, qdot, qddot)` at arbitrary sample times for one plane trajectory."""

        return np.vstack(
            (
                np.asarray(trajectory.position(sample_times), dtype=float),
                np.asarray(trajectory.velocity(sample_times), dtype=float),
                np.asarray(trajectory.acceleration(sample_times), dtype=float),
            )
        )

    def candidate_start_times(self) -> np.ndarray:
        """Return the admissible discrete second-arm start times on the shooting grid."""

        lower_bound = max(RIGHT_ARM_SWEEP_BOUNDS[0], RIGHT_ARM_START_BOUNDS[0])
        upper_bound = min(
            RIGHT_ARM_SWEEP_BOUNDS[1],
            RIGHT_ARM_START_BOUNDS[1],
            self.configuration.final_time - RIGHT_ARM_ACTIVE_DURATION,
        )
        if lower_bound > upper_bound:
            raise ValueError("The discrete second-arm start-time sweep is empty.")
        first_node = int(round(lower_bound / self.shooting_step))
        last_node = int(round(upper_bound / self.shooting_step))
        return np.round(self.shooting_step * np.arange(first_node, last_node + 1, dtype=float), decimals=10)

    def _snap_start_time_to_grid(self, right_arm_start: float) -> float:
        """Project one start time onto the shooting grid while tolerating floating-point noise."""

        clipped_start = float(
            np.clip(right_arm_start, RIGHT_ARM_START_BOUNDS[0], RIGHT_ARM_START_BOUNDS[1])
        )
        snapped_start = self.shooting_step * round(clipped_start / self.shooting_step)
        return float(np.round(snapped_start, decimals=10))

    def _global_jerk_bounds(
        self,
        *,
        right_start_node_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the lower/upper jerk bounds applied on the 50-node horizon."""

        left_lower = np.zeros(self.interval_count, dtype=float)
        left_upper = np.zeros(self.interval_count, dtype=float)
        left_lower[: self.active_control_count] = -self.jerk_bound
        left_upper[: self.active_control_count] = self.jerk_bound

        right_lower = np.zeros(self.interval_count, dtype=float)
        right_upper = np.zeros(self.interval_count, dtype=float)
        right_terminal_node_index = right_start_node_index + self.active_control_count
        right_lower[right_start_node_index:right_terminal_node_index] = -self.jerk_bound
        right_upper[right_start_node_index:right_terminal_node_index] = self.jerk_bound
        return left_lower, left_upper, right_lower, right_upper

    def _build_solver(
        self,
        *,
        max_iter: int,
        print_level: int,
        print_time: bool,
    ):
        """Build one global DMS solver and cache it."""

        options_key = (
            int(max_iter),
            int(print_level),
            bool(print_time),
            self.objective_mode,
            float(self.btp_deviation_weight),
        )
        if self._solver is not None and self._solver_options_key == options_key:
            return self._solver

        configure_optimization_threads()
        parameters = ca.MX.sym("p", ELEVATION_STAGE_BLOCK_SIZE * self.interval_count, 1)
        root_state_symbols = [
            ca.MX.sym(f"X_root_{index}", ROOT_STATE_SIZE, 1) for index in range(self.interval_count + 1)
        ]
        left_plane_state_symbols = [
            ca.MX.sym(f"X_left_{index}", PLANE_STATE_SIZE, 1) for index in range(self.interval_count + 1)
        ]
        right_plane_state_symbols = [
            ca.MX.sym(f"X_right_{index}", PLANE_STATE_SIZE, 1) for index in range(self.interval_count + 1)
        ]
        left_control_symbols = ca.MX.sym("U_left", self.interval_count, 1)
        right_control_symbols = ca.MX.sym("U_right", self.interval_count, 1)

        objective = root_state_symbols[-1][5] / (2.0 * np.pi)
        objective += self.jerk_regularization * self.shooting_step * (
            ca.sumsqr(left_control_symbols) + ca.sumsqr(right_control_symbols)
        )
        interval_defect = self._build_interval_defect_function()
        parallelization = "openmp" if self.interval_count > 1 else "serial"
        try:
            mapped_interval_defect = interval_defect.map(
                self.interval_count,
                parallelization,
            )
        except RuntimeError:
            parallelization = "serial"
            mapped_interval_defect = interval_defect.map(self.interval_count, parallelization)
        self._interval_parallelization = parallelization

        mapped_defects, mapped_running_costs = mapped_interval_defect(
            ca.horzcat(*root_state_symbols[:-1]),
            ca.horzcat(*root_state_symbols[1:]),
            ca.horzcat(*left_plane_state_symbols[:-1]),
            ca.horzcat(*left_plane_state_symbols[1:]),
            ca.horzcat(*right_plane_state_symbols[:-1]),
            ca.horzcat(*right_plane_state_symbols[1:]),
            ca.reshape(left_control_symbols, 1, self.interval_count),
            ca.reshape(right_control_symbols, 1, self.interval_count),
            ca.reshape(parameters, ELEVATION_STAGE_BLOCK_SIZE, self.interval_count),
        )
        if self.objective_mode == OBJECTIVE_MODE_TWIST_BTP:
            objective += self.btp_deviation_weight * self.shooting_step * ca.sum2(mapped_running_costs)
        constraints = [ca.reshape(mapped_defects, -1, 1)]

        decision_vector = ca.vertcat(
            *root_state_symbols,
            *left_plane_state_symbols,
            *right_plane_state_symbols,
            left_control_symbols,
            right_control_symbols,
        )
        self._constraint_count = int(sum(component.shape[0] for component in constraints))
        self._solver = ca.nlpsol(
            f"dms_solver_{self.objective_mode}",
            "ipopt",
            {
                "x": decision_vector,
                "p": parameters,
                "f": objective,
                "g": ca.vertcat(*constraints),
            },
            build_ipopt_solver_options(
                max_iter=max_iter,
                print_level=print_level,
                print_time=print_time,
                expand=self._nlpsol_expand,
                warm_start=True,
            ),
        )
        self._solver_options_key = options_key
        return self._solver

    def _numeric_segment_origin(self, q_full: np.ndarray, segment_name: str) -> np.ndarray:
        """Return the numeric origin of one segment."""

        q_biorbd = biorbd.GeneralizedCoordinates(np.asarray(q_full, dtype=float))
        return np.asarray(
            self.model.globalJCS(q_biorbd, self._segment_index_by_name[segment_name]).to_array()[:3, 3],
            dtype=float,
        )

    def _btp_deviation_lagrange_from_state_nodes(
        self,
        *,
        root_state_nodes: np.ndarray,
        left_plane_state_nodes: np.ndarray,
        right_plane_state_nodes: np.ndarray,
        right_arm_start: float,
    ) -> float:
        """Return the node-wise BTP deviation Lagrange term used by the 3D-BTP objective."""

        if self.objective_mode != OBJECTIVE_MODE_TWIST_BTP:
            return 0.0

        deviation_costs = np.zeros(self.interval_count, dtype=float)
        for interval_index in range(self.interval_count):
            node_time = float(self.node_times[interval_index])
            elevation_q, _, _ = self._fixed_elevation_kinematics(
                time=node_time,
                right_arm_start=right_arm_start,
            )
            q_full = np.array(
                [
                    *root_state_nodes[:6, interval_index],
                    left_plane_state_nodes[0, interval_index],
                    elevation_q[0],
                    right_plane_state_nodes[0, interval_index],
                    elevation_q[1],
                ],
                dtype=float,
            )
            left_vector = self._numeric_segment_origin(q_full, "forearm_left") - self._numeric_segment_origin(
                q_full,
                "upper_arm_left",
            )
            right_vector = self._numeric_segment_origin(q_full, "forearm_right") - self._numeric_segment_origin(
                q_full,
                "upper_arm_right",
            )
            somersault_angle = float(root_state_nodes[3, interval_index])
            left_deviation = float(
                self._symbolic_signed_plane_deviation(ca.DM(left_vector), ca.DM(somersault_angle))
            )
            right_deviation = float(
                self._symbolic_signed_plane_deviation(ca.DM(right_vector), ca.DM(somersault_angle))
            )
            deviation_costs[interval_index] = left_deviation * left_deviation + right_deviation * right_deviation
        return float(self.btp_deviation_weight * self.shooting_step * np.sum(deviation_costs))

    @staticmethod
    def _project_initial_guess_to_bounds(
        initial_guess: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> np.ndarray:
        """Project one warm start inside the finite NLP bounds."""

        projected = np.asarray(initial_guess, dtype=float).copy()
        finite_lower = np.isfinite(lower_bounds)
        finite_upper = np.isfinite(upper_bounds)
        projected[finite_lower] = np.maximum(projected[finite_lower], lower_bounds[finite_lower])
        projected[finite_upper] = np.minimum(projected[finite_upper], upper_bounds[finite_upper])
        return projected

    def solve_fixed_start(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        right_arm_start: float,
        previous_result: DirectMultipleShootingResult | None = None,
        warm_start_override: np.ndarray | None = None,
        max_iter: int = 100,
        print_level: int = 0,
        print_time: bool = False,
        show_jerk_diagnostics: bool = False,
    ) -> DirectMultipleShootingResult:
        """Solve one direct multiple-shooting problem with a fixed second-arm start time."""

        if (
            right_arm_start < RIGHT_ARM_START_BOUNDS[0] - START_TIME_TOLERANCE
            or right_arm_start > RIGHT_ARM_START_BOUNDS[1] + START_TIME_TOLERANCE
        ):
            raise ValueError("The fixed second-arm start time is outside the admissible bounds.")
        right_arm_start = self._snap_start_time_to_grid(float(right_arm_start))

        solver = self._build_solver(max_iter=max_iter, print_level=print_level, print_time=print_time)
        fixed_variables = TwistOptimizationVariables(
            right_arm_start=float(right_arm_start),
            left_plane_initial=initial_guess.left_plane_initial,
            left_plane_final=initial_guess.left_plane_final,
            right_plane_initial=initial_guess.right_plane_initial,
            right_plane_final=initial_guess.right_plane_final,
            contact_twist_rate=initial_guess.contact_twist_rate,
        )
        right_start_node_index = int(round(right_arm_start / self.shooting_step))
        right_terminal_node_index = right_start_node_index + self.active_control_count

        initial_motion = self._initial_guess_motion(fixed_variables, right_arm_start=right_arm_start)
        initial_root_states = self._initial_guess_root_state_history(fixed_variables, initial_motion)
        initial_left_states = self._plane_state_history(initial_motion.left_plane, self.node_times)
        right_local_times = np.maximum(0.0, self.node_times - right_arm_start)
        initial_right_states = self._plane_state_history(initial_motion.right_plane, right_local_times)

        initial_root_state = self._symbolic_initial_root_state(fixed_variables)
        initial_left_plane_state = self._plane_state_vector(fixed_variables.left_plane_initial)
        initial_right_plane_state = self._plane_state_vector(fixed_variables.right_plane_initial)
        terminal_left_plane_state = self._plane_state_vector(fixed_variables.left_plane_final)
        terminal_right_plane_state = self._plane_state_vector(fixed_variables.right_plane_final)
        left_lower_bounds, left_upper_bounds, right_lower_bounds, right_upper_bounds = (
            self._global_jerk_bounds(right_start_node_index=right_start_node_index)
        )

        x0: list[float] = []
        lbx: list[float] = []
        ubx: list[float] = []
        for state_index in range(self.interval_count + 1):
            root_guess = initial_root_states[:, state_index]
            x0.extend(root_guess.tolist())
            if state_index == 0:
                lbx.extend(initial_root_state.tolist())
                ubx.extend(initial_root_state.tolist())
            else:
                lbx.extend([-float("inf")] * ROOT_STATE_SIZE)
                ubx.extend([float("inf")] * ROOT_STATE_SIZE)

        for state_index in range(self.interval_count + 1):
            left_guess = initial_left_states[:, state_index]
            x0.extend(left_guess.tolist())
            if state_index == 0:
                lbx.extend(initial_left_plane_state.tolist())
                ubx.extend(initial_left_plane_state.tolist())
            elif state_index == self.active_control_count:
                lbx.extend(terminal_left_plane_state.tolist())
                ubx.extend(terminal_left_plane_state.tolist())
            else:
                lbx.extend([-float("inf")] * PLANE_STATE_SIZE)
                ubx.extend([float("inf")] * PLANE_STATE_SIZE)

        for state_index in range(self.interval_count + 1):
            right_guess = initial_right_states[:, state_index]
            x0.extend(right_guess.tolist())
            if state_index == 0:
                lbx.extend(initial_right_plane_state.tolist())
                ubx.extend(initial_right_plane_state.tolist())
            elif state_index == right_terminal_node_index:
                lbx.extend(terminal_right_plane_state.tolist())
                ubx.extend(terminal_right_plane_state.tolist())
            else:
                lbx.extend([-float("inf")] * PLANE_STATE_SIZE)
                ubx.extend([float("inf")] * PLANE_STATE_SIZE)

        for control_index in range(self.interval_count):
            jerk_guess = (
                initial_motion.left_plane.jerks[control_index]
                if control_index < self.active_control_count
                else 0.0
            )
            x0.append(float(jerk_guess))
            lbx.append(float(left_lower_bounds[control_index]))
            ubx.append(float(left_upper_bounds[control_index]))

        for control_index in range(self.interval_count):
            active = right_start_node_index <= control_index < right_terminal_node_index
            jerk_guess = (
                initial_motion.right_plane.jerks[control_index - right_start_node_index]
                if active
                else 0.0
            )
            x0.append(float(jerk_guess))
            lbx.append(float(right_lower_bounds[control_index]))
            ubx.append(float(right_upper_bounds[control_index]))

        x0_array = np.asarray(x0, dtype=float)
        lbx_array = np.asarray(lbx, dtype=float)
        ubx_array = np.asarray(ubx, dtype=float)
        solver_inputs = {
            "x0": x0_array,
            "lbx": lbx_array,
            "ubx": ubx_array,
            "lbg": np.zeros(self._constraint_count, dtype=float),
            "ubg": np.zeros(self._constraint_count, dtype=float),
            "p": self._elevation_parameter_values(right_arm_start=right_arm_start),
        }
        if warm_start_override is not None:
            solver_inputs["x0"] = self._project_initial_guess_to_bounds(
                np.asarray(warm_start_override, dtype=float),
                lbx_array,
                ubx_array,
            )
        elif previous_result is not None and previous_result.warm_start_primal is not None:
            solver_inputs["x0"] = self._project_initial_guess_to_bounds(
                previous_result.warm_start_primal,
                lbx_array,
                ubx_array,
            )
            if previous_result.warm_start_lam_x is not None:
                solver_inputs["lam_x0"] = np.asarray(previous_result.warm_start_lam_x, dtype=float)
            if previous_result.warm_start_lam_g is not None:
                solver_inputs["lam_g0"] = np.asarray(previous_result.warm_start_lam_g, dtype=float)

        print(
            f"DMS solve: t1={right_arm_start:.2f} s "
            f"(node {right_start_node_index}/{self.interval_count})"
        )
        solution = solver(**solver_inputs)
        status = solver.stats()["return_status"]
        raw_solution = np.asarray(solution["x"].full(), dtype=float).reshape(-1)
        offset = 0
        root_state_values = raw_solution[offset : offset + ROOT_STATE_SIZE * (self.interval_count + 1)]
        offset += ROOT_STATE_SIZE * (self.interval_count + 1)
        left_plane_global_values = raw_solution[offset : offset + PLANE_STATE_SIZE * (self.interval_count + 1)]
        offset += PLANE_STATE_SIZE * (self.interval_count + 1)
        right_plane_global_values = raw_solution[offset : offset + PLANE_STATE_SIZE * (self.interval_count + 1)]
        offset += PLANE_STATE_SIZE * (self.interval_count + 1)
        left_control_global_values = raw_solution[offset : offset + self.interval_count]
        offset += self.interval_count
        right_control_global_values = raw_solution[offset : offset + self.interval_count]
        root_state_nodes = root_state_values.reshape(self.interval_count + 1, ROOT_STATE_SIZE).T
        twist_objective_value = float(root_state_nodes[5, -1] / (2.0 * np.pi))
        jerk_regularization_value = float(
            self.jerk_regularization
            * self.shooting_step
            * (np.sum(left_control_global_values**2) + np.sum(right_control_global_values**2))
        )
        if show_jerk_diagnostics:
            show_dms_jerk_bounds_figure(
                node_times=self.node_times,
                left_jerk=np.asarray(left_control_global_values, dtype=float),
                right_jerk=np.asarray(right_control_global_values, dtype=float),
                left_lower_bounds=left_lower_bounds,
                left_upper_bounds=left_upper_bounds,
                right_lower_bounds=right_lower_bounds,
                right_upper_bounds=right_upper_bounds,
                right_arm_start=right_arm_start,
            )

        left_control_values = left_control_global_values[: self.active_control_count]
        right_control_values = right_control_global_values[
            right_start_node_index : right_start_node_index + self.active_control_count
        ]
        left_plane = PiecewiseConstantJerkTrajectory(
            q0=fixed_variables.left_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=left_control_values,
            active_start=0.0,
            active_end=LEFT_ARM_ACTIVE_DURATION,
            total_duration=self.configuration.final_time,
        )
        right_plane = PiecewiseConstantJerkTrajectory(
            q0=fixed_variables.right_plane_initial,
            qdot0=0.0,
            qddot0=0.0,
            step=self.shooting_step,
            jerks=right_control_values,
            active_start=0.0,
            active_end=RIGHT_ARM_ACTIVE_DURATION,
            total_duration=self.configuration.final_time - right_arm_start,
        )
        motion = PiecewiseConstantJerkArmMotion(
            left_plane=left_plane,
            right_plane=right_plane,
            right_arm_start=right_arm_start,
        )
        simulation = PredictiveAerialTwistSimulator(
            self.model_path,
            motion,
            configuration=self.configuration,
            model=self.model,
        ).simulate()
        left_plane_global_states = left_plane_global_values.reshape(self.interval_count + 1, PLANE_STATE_SIZE).T
        right_plane_global_states = right_plane_global_values.reshape(self.interval_count + 1, PLANE_STATE_SIZE).T
        btp_deviation_lagrange_value = self._btp_deviation_lagrange_from_state_nodes(
            root_state_nodes=root_state_nodes,
            left_plane_state_nodes=left_plane_global_states,
            right_plane_state_nodes=right_plane_global_states,
            right_arm_start=right_arm_start,
        )
        total_objective_value = (
            twist_objective_value + jerk_regularization_value + btp_deviation_lagrange_value
        )
        print(
            "DMS objective terms: "
            f"twist={twist_objective_value:.6f}, "
            f"jerk_reg={jerk_regularization_value:.6e}, "
            f"btp_lagrange={btp_deviation_lagrange_value:.6e}, "
            f"total={total_objective_value:.6f}, "
            f"ipopt_f={float(solution['f']):.6f}"
        )

        normalized_status = status.lower().replace("_", " ")
        return DirectMultipleShootingResult(
            variables=fixed_variables,
            right_arm_start_node_index=right_start_node_index,
            left_plane_jerk=left_control_values.copy(),
            right_plane_jerk=right_control_values.copy(),
            node_times=self.node_times.copy(),
            arm_node_times=self.arm_node_times.copy(),
            root_state_nodes=root_state_nodes,
            left_plane_state_nodes=left_plane_global_states[:, : self.active_control_count + 1].copy(),
            right_plane_state_nodes=right_plane_global_states[
                :,
                right_start_node_index : right_terminal_node_index + 1,
            ].copy(),
            prescribed_motion=motion,
            simulation=simulation,
            objective=float(solution["f"]),
            btp_deviation_lagrange=btp_deviation_lagrange_value,
            solver_status=status,
            success=(
                "success" in normalized_status
                or "succeeded" in normalized_status
                or "solved" in normalized_status
            ),
            warm_start_primal=raw_solution.copy(),
            warm_start_lam_x=(
                np.asarray(solution["lam_x"].full(), dtype=float).reshape(-1)
                if "lam_x" in solution
                else None
            ),
            warm_start_lam_g=(
                np.asarray(solution["lam_g"].full(), dtype=float).reshape(-1)
                if "lam_g" in solution
                else None
            ),
        )

    def _randomized_warm_start(
        self,
        base_primal: np.ndarray,
        *,
        right_start_node_index: int,
        generator: np.random.Generator,
    ) -> np.ndarray:
        """Return one randomized warm start by perturbing only the active jerk controls."""

        randomized = np.asarray(base_primal, dtype=float).copy()
        left_control_offset = (
            ROOT_STATE_SIZE * (self.interval_count + 1)
            + 2 * PLANE_STATE_SIZE * (self.interval_count + 1)
        )
        right_control_offset = left_control_offset + self.interval_count

        left_slice = slice(left_control_offset, left_control_offset + self.active_control_count)
        right_slice = slice(
            right_control_offset + right_start_node_index,
            right_control_offset + right_start_node_index + self.active_control_count,
        )
        perturbation_scale = 0.35 * self.jerk_bound
        randomized[left_slice] += generator.uniform(
            low=-perturbation_scale,
            high=perturbation_scale,
            size=self.active_control_count,
        )
        randomized[right_slice] += generator.uniform(
            low=-perturbation_scale,
            high=perturbation_scale,
            size=self.active_control_count,
        )
        return randomized

    def solve_fixed_start_multistart(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        right_arm_start: float,
        start_count: int = MULTISTART_START_COUNT,
        previous_result: DirectMultipleShootingResult | None = None,
        max_iter: int = 100,
        print_level: int = 0,
        print_time: bool = False,
        show_jerk_diagnostics: bool = False,
    ) -> DirectMultipleShootingResult:
        """Run several fixed-start solves and keep the best result for the requested `t1`."""

        if start_count <= 1:
            return self.solve_fixed_start(
                initial_guess,
                right_arm_start=right_arm_start,
                previous_result=previous_result,
                max_iter=max_iter,
                print_level=print_level,
                print_time=print_time,
                show_jerk_diagnostics=show_jerk_diagnostics,
            )

        generator = np.random.default_rng(1234)
        right_start_node_index = int(round(right_arm_start / self.shooting_step))
        best_result = self.solve_fixed_start(
            initial_guess,
            right_arm_start=right_arm_start,
            previous_result=previous_result,
            max_iter=max_iter,
            print_level=print_level,
            print_time=print_time,
            show_jerk_diagnostics=False,
        )
        selection_pool: list[DirectMultipleShootingResult] = [best_result]
        best_primal = best_result.warm_start_primal

        for start_index in range(1, start_count):
            print(
                f"DMS multistart: t1={right_arm_start:.2f} s "
                f"({start_index + 1}/{start_count})"
            )
            randomized_warm_start = None
            if best_primal is not None:
                randomized_warm_start = self._randomized_warm_start(
                    best_primal,
                    right_start_node_index=right_start_node_index,
                    generator=generator,
                )
            current_result = self.solve_fixed_start(
                initial_guess,
                right_arm_start=right_arm_start,
                warm_start_override=randomized_warm_start,
                max_iter=max_iter,
                print_level=print_level,
                print_time=print_time,
                show_jerk_diagnostics=False,
            )
            selection_pool.append(current_result)
            best_candidates = [result for result in selection_pool if result.success]
            if not best_candidates:
                best_candidates = selection_pool
            best_result = min(best_candidates, key=lambda result: result.objective)
            if best_result.warm_start_primal is not None:
                best_primal = best_result.warm_start_primal

        if show_jerk_diagnostics:
            show_dms_jerk_bounds_figure(
                node_times=best_result.node_times,
                left_jerk=np.pad(
                    np.asarray(best_result.left_plane_jerk, dtype=float),
                    (0, self.interval_count - best_result.left_plane_jerk.size),
                    mode="constant",
                ),
                right_jerk=np.pad(
                    np.asarray(best_result.right_plane_jerk, dtype=float),
                    (right_start_node_index, self.interval_count - right_start_node_index - best_result.right_plane_jerk.size),
                    mode="constant",
                ),
                left_lower_bounds=self._global_jerk_bounds(right_start_node_index=right_start_node_index)[0],
                left_upper_bounds=self._global_jerk_bounds(right_start_node_index=right_start_node_index)[1],
                right_lower_bounds=self._global_jerk_bounds(right_start_node_index=right_start_node_index)[2],
                right_upper_bounds=self._global_jerk_bounds(right_start_node_index=right_start_node_index)[3],
                right_arm_start=right_arm_start,
            )
        return best_result

    def solve(
        self,
        initial_guess: TwistOptimizationVariables,
        *,
        max_iter: int = 100,
        print_level: int = 0,
        print_time: bool = False,
    ) -> DirectMultipleShootingSweepResult:
        """Solve one fixed-start OCP per admissible node and keep the best solution."""

        candidate_results_list: list[DirectMultipleShootingResult] = []
        previous_result: DirectMultipleShootingResult | None = None
        best_warm_start_result: DirectMultipleShootingResult | None = None
        candidate_start_times = np.asarray(self.candidate_start_times(), dtype=float)
        if candidate_start_times.size == 0:
            raise ValueError("No admissible second-arm start time is available for the DMS sweep.")
        for start_time in candidate_start_times:
            current_result = self.solve_fixed_start(
                initial_guess,
                right_arm_start=float(start_time),
                previous_result=(best_warm_start_result if best_warm_start_result is not None else previous_result),
                max_iter=max_iter,
                print_level=print_level,
                print_time=print_time,
            )
            candidate_results_list.append(current_result)
            previous_result = current_result
            if current_result.warm_start_primal is not None and _result_is_better(current_result, best_warm_start_result):
                best_warm_start_result = current_result
        candidate_results = tuple(candidate_results_list)
        successful_results = tuple(result for result in candidate_results if result.success)
        selection_pool = successful_results if successful_results else candidate_results
        best_result = min(selection_pool, key=lambda result: result.objective)
        return DirectMultipleShootingSweepResult(
            best_result=best_result,
            candidate_results=candidate_results,
        )


def create_dms_start_time_sweep_figure(
    *,
    start_times: np.ndarray,
    final_twist_turns: np.ndarray,
    objective_values: np.ndarray,
    success_mask: np.ndarray,
    best_start_time: float,
):
    """Create one simple figure summarizing the discrete sweep over the second-arm start time."""

    import matplotlib.pyplot as plt

    del objective_values
    times = np.asarray(start_times, dtype=float)
    twists = np.asarray(final_twist_turns, dtype=float)
    successes = np.asarray(success_mask, dtype=bool)
    best_index = int(np.argmin(np.abs(times - float(best_start_time))))

    figure, axis = plt.subplots(1, 1, figsize=(7.5, 4.5), tight_layout=True)
    axis.plot(times, twists, color="tab:blue", linewidth=1.6, marker="o", markersize=4.0)
    if np.any(~successes):
        axis.scatter(times[~successes], twists[~successes], color="tab:red", marker="x", s=36, label="Echec NLP")
    axis.scatter(
        [times[best_index]],
        [twists[best_index]],
        color="tab:orange",
        s=64,
        zorder=3,
        label="Meilleure solution",
    )
    axis.set_ylabel("Vrille finale (tours)")
    axis.set_xlabel("Debut bras droit t1 (s)")
    axis.set_title("Balayage DMS sur les noeuds de t1")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    return figure, axis


def show_dms_start_time_sweep_figure(
    *,
    start_times: np.ndarray,
    final_twist_turns: np.ndarray,
    objective_values: np.ndarray,
    success_mask: np.ndarray,
    best_start_time: float,
):
    """Open an external figure summarizing the discrete sweep over the second-arm start time."""

    import matplotlib.pyplot as plt

    figure, axis = create_dms_start_time_sweep_figure(
        start_times=start_times,
        final_twist_turns=final_twist_turns,
        objective_values=objective_values,
        success_mask=success_mask,
        best_start_time=best_start_time,
    )
    figure.canvas.draw_idle()
    plt.show(block=False)
    return figure, axis


def create_dms_jerk_bounds_figure(
    *,
    node_times: np.ndarray,
    left_jerk: np.ndarray,
    right_jerk: np.ndarray,
    left_lower_bounds: np.ndarray,
    left_upper_bounds: np.ndarray,
    right_lower_bounds: np.ndarray,
    right_upper_bounds: np.ndarray,
    right_arm_start: float,
):
    """Create one temporary diagnostic figure showing jerk controls and their interval bounds."""

    import matplotlib.pyplot as plt

    interval_times = np.asarray(node_times[:-1], dtype=float)
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9.0, 6.0), tight_layout=True)

    for axis, jerks, lower, upper, title in (
        (
            axes[0],
            np.asarray(left_jerk, dtype=float),
            np.asarray(left_lower_bounds, dtype=float),
            np.asarray(left_upper_bounds, dtype=float),
            f"Bras gauche | borne = +/-{float(np.max(np.abs(left_upper_bounds))):.2f} rad/s^3",
        ),
        (
            axes[1],
            np.asarray(right_jerk, dtype=float),
            np.asarray(right_lower_bounds, dtype=float),
            np.asarray(right_upper_bounds, dtype=float),
            f"Bras droit | t1 = {right_arm_start:.2f} s | borne = +/-{float(np.max(np.abs(right_upper_bounds))):.2f} rad/s^3",
        ),
    ):
        axis.step(interval_times, lower, where="post", color="0.55", linestyle="--", linewidth=1.2)
        axis.step(interval_times, upper, where="post", color="0.55", linestyle="--", linewidth=1.2)
        axis.step(interval_times, jerks, where="post", color="tab:blue", linewidth=1.8)
        axis.axhline(0.0, color="0.8", linewidth=1.0)
        axis.set_ylabel("Jerk (rad/s^3)")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Temps (s)")
    figure.suptitle("Diagnostic DMS: jerks piecewise constants et bornes", fontsize=12)
    return figure, axes


def show_dms_jerk_bounds_figure(
    *,
    node_times: np.ndarray,
    left_jerk: np.ndarray,
    right_jerk: np.ndarray,
    left_lower_bounds: np.ndarray,
    left_upper_bounds: np.ndarray,
    right_lower_bounds: np.ndarray,
    right_upper_bounds: np.ndarray,
    right_arm_start: float,
):
    """Open one external diagnostic figure right after one IPOPT solve."""

    import matplotlib.pyplot as plt

    figure, axes = create_dms_jerk_bounds_figure(
        node_times=node_times,
        left_jerk=left_jerk,
        right_jerk=right_jerk,
        left_lower_bounds=left_lower_bounds,
        left_upper_bounds=left_upper_bounds,
        right_lower_bounds=right_lower_bounds,
        right_upper_bounds=right_upper_bounds,
        right_arm_start=right_arm_start,
    )
    figure.canvas.draw_idle()
    if "agg" not in plt.get_backend().lower():
        plt.show(block=False)
    return figure, axes

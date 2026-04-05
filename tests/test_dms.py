"""Tests for the direct multiple-shooting optimizer."""

from __future__ import annotations

from pathlib import Path

import casadi as ca
import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.optimization import DirectMultipleShootingOptimizer
from best_tilting_plane.optimization import dms as dms_module
from best_tilting_plane.simulation import (
    PiecewiseConstantJerkArmMotion,
    PiecewiseConstantJerkTrajectory,
    SimulationConfiguration,
    TwistOptimizationVariables,
)


def test_direct_multiple_shooting_initial_guess_motion_respects_activation_windows(
    tmp_path: Path,
) -> None:
    """The warm start should activate the first arm immediately and the second arm at `t1`."""

    optimizer = DirectMultipleShootingOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=1.0, steps=201, integrator="rk4", rk4_step=0.005),
        shooting_step=0.02,
    )

    motion = optimizer._initial_guess_motion(
        TwistOptimizationVariables(
            right_arm_start=0.16,
            left_plane_initial=-0.2,
            left_plane_final=0.1,
            right_plane_initial=0.3,
            right_plane_final=-0.1,
        )
    )

    assert motion.left_plane.active_start == 0.0
    assert motion.left_plane.active_end == dms_module.LEFT_ARM_ACTIVE_DURATION
    assert motion.right_plane.active_start == 0.16
    assert motion.right_plane.active_end == 0.16 + dms_module.RIGHT_ARM_ACTIVE_DURATION
    assert motion.left_plane.jerks.shape == (optimizer.interval_count,)
    assert motion.right_plane.jerks.shape == (optimizer.interval_count,)


def test_direct_multiple_shooting_solve_builds_float_bounds_and_returns_motion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The DMS solve path should assemble a float-bounded NLP and rebuild the jerk-controlled motion."""

    optimizer = DirectMultipleShootingOptimizer.from_builder(
        tmp_path / "reduced.bioMod",
        model_builder=ReducedAerialBiomod(),
        configuration=SimulationConfiguration(final_time=0.4, steps=81, integrator="rk4", rk4_step=0.005),
        shooting_step=0.02,
    )
    initial_guess = TwistOptimizationVariables(
        right_arm_start=0.1,
        left_plane_initial=0.0,
        left_plane_final=0.0,
        right_plane_initial=0.0,
        right_plane_final=0.0,
    )
    warm_start_motion = PiecewiseConstantJerkArmMotion(
        left_plane=PiecewiseConstantJerkTrajectory(
            q0=0.0,
            qdot0=0.0,
            qddot0=0.0,
            step=optimizer.shooting_step,
            jerks=np.zeros(optimizer.interval_count, dtype=float),
            active_start=0.0,
            active_end=dms_module.LEFT_ARM_ACTIVE_DURATION,
        ),
        right_plane=PiecewiseConstantJerkTrajectory(
            q0=0.0,
            qdot0=0.0,
            qddot0=0.0,
            step=optimizer.shooting_step,
            jerks=np.zeros(optimizer.interval_count, dtype=float),
            active_start=initial_guess.right_arm_start,
            active_end=initial_guess.right_arm_start + dms_module.RIGHT_ARM_ACTIVE_DURATION,
        ),
        right_arm_start=initial_guess.right_arm_start,
    )
    initial_state = np.asarray(optimizer._symbolic_initial_state(initial_guess), dtype=float).reshape(-1)
    state_history = np.tile(initial_state.reshape(-1, 1), (1, optimizer.interval_count + 1))
    captured: dict[str, object] = {}

    monkeypatch.setattr(optimizer, "_initial_guess_motion", lambda _variables: warm_start_motion)
    monkeypatch.setattr(
        optimizer,
        "_initial_guess_state_history",
        lambda _variables, _motion: state_history,
    )

    class FakeSimulator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def simulate():
            return type(
                "SimulationResult",
                (),
                {
                    "final_twist_angle": -0.5,
                    "final_twist_turns": -0.08,
                },
            )()

    class FakeSolver:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"x": ca.DM(np.asarray(kwargs["x0"], dtype=float)), "f": ca.DM([[-0.5]])}

        @staticmethod
        def stats():
            return {"return_status": "Solve_Succeeded"}

    def fake_nlpsol(_name, _solver_name, _problem, options):
        captured["options"] = options
        return FakeSolver()

    monkeypatch.setattr(dms_module, "PredictiveAerialTwistSimulator", FakeSimulator)
    monkeypatch.setattr(dms_module.ca, "nlpsol", fake_nlpsol)

    result = optimizer.solve(initial_guess, max_iter=3, print_level=5, print_time=True)

    assert result.success
    assert result.solver_status == "Solve_Succeeded"
    assert result.variables == initial_guess
    assert result.left_plane_jerk.shape == (optimizer.interval_count,)
    assert result.right_plane_jerk.shape == (optimizer.interval_count,)
    assert result.prescribed_motion.right_arm_start == initial_guess.right_arm_start
    assert result.objective == -0.5
    assert captured["options"]["ipopt.max_iter"] == 3
    assert captured["options"]["ipopt.print_level"] == 5
    assert captured["options"]["print_time"] == 1
    assert np.asarray(captured["x0"], dtype=float).shape == (
        1 + (optimizer.interval_count + 1) * dms_module.FULL_STATE_SIZE + optimizer.interval_count * 2,
    )
    assert np.asarray(captured["lbx"], dtype=float).dtype == float
    assert np.asarray(captured["ubx"], dtype=float).dtype == float

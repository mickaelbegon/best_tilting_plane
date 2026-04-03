"""Tests for the predictive floating-base simulation."""

from pathlib import Path

import numpy as np

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.simulation import PrescribedArmMotion, TwistOptimizationVariables
from best_tilting_plane.simulation.dynamics import (
    PredictiveAerialTwistSimulator,
    SimulationConfiguration,
)


def _default_motion() -> PrescribedArmMotion:
    return PrescribedArmMotion(
        TwistOptimizationVariables(
            right_arm_start=0.1,
            left_plane_initial=0.0,
            left_plane_final=0.0,
            right_plane_initial=0.0,
            right_plane_final=0.0,
        )
    )


def test_initial_state_cancels_center_of_mass_velocity(tmp_path: Path) -> None:
    """The initial translational velocity should be corrected to enforce `v_COM = 0`."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    simulator = PredictiveAerialTwistSimulator(model_path, _default_motion())

    q0, qdot0 = simulator.initial_state()

    np.testing.assert_allclose(simulator.center_of_mass_velocity(q0, qdot0), 0.0, atol=1e-10)


def test_simulation_returns_consistent_shapes(tmp_path: Path) -> None:
    """A short simulation should return consistent state histories and a finite twist count."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    simulator = PredictiveAerialTwistSimulator(
        model_path,
        _default_motion(),
        configuration=SimulationConfiguration(final_time=0.1, steps=11),
    )

    result = simulator.simulate()

    assert result.time.shape == (11,)
    assert result.q.shape == (11, 10)
    assert result.qdot.shape == (11, 10)
    assert result.qddot.shape == (11, 10)
    assert np.isfinite(result.final_twist_angle)
    assert np.isfinite(result.final_twist_turns)

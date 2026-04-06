"""Tests for the predictive floating-base simulation."""

from pathlib import Path

import numpy as np
import pytest

from best_tilting_plane.modeling import ReducedAerialBiomod
from best_tilting_plane.simulation import PrescribedArmMotion, TwistOptimizationVariables
from best_tilting_plane.simulation.dynamics import (
    IntegratorSelection,
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


def test_initial_state_applies_the_contact_twist_rate(tmp_path: Path) -> None:
    """The configured contact twist should populate the initial root-twist velocity."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    simulator = PredictiveAerialTwistSimulator(
        model_path,
        _default_motion(),
        configuration=SimulationConfiguration(contact_twist_rate=-2.0 * np.pi),
    )

    _q0, qdot0 = simulator.initial_state()

    assert qdot0[5] == pytest.approx(-2.0 * np.pi)


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


def test_fixed_step_rk4_returns_consistent_shapes(tmp_path: Path) -> None:
    """A short RK4 simulation should return consistent state histories."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    simulator = PredictiveAerialTwistSimulator(
        model_path,
        _default_motion(),
        configuration=SimulationConfiguration(
            final_time=0.1,
            steps=11,
            integrator="rk4",
            rk4_step=0.01,
        ),
    )

    result = simulator.simulate()

    assert result.time.shape == (11,)
    assert result.q.shape == (11, 10)
    assert result.qdot.shape == (11, 10)
    assert result.qddot.shape == (11, 10)
    assert result.integrator_method == "rk4"
    assert result.rk4_step == 0.01


def test_auto_integrator_prefers_faster_accurate_rk4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auto selector should choose RK4 only when it is both accurate enough and faster."""

    model_path = ReducedAerialBiomod().write(tmp_path / "reduced.bioMod")
    simulator = PredictiveAerialTwistSimulator(
        model_path,
        _default_motion(),
        configuration=SimulationConfiguration(integrator="auto"),
    )
    evaluation_times = np.linspace(
        0.0, simulator.configuration.final_time, simulator.configuration.steps
    )
    reference_state = np.zeros((12, evaluation_times.size))

    def fake_rk45(_initial_state, _t_eval):
        return evaluation_times, reference_state, 0.10

    def fake_rk4(_initial_state, _t_eval, step):
        states = reference_state.copy()
        if step == 0.02:
            states[0, -1] = 1e-2
            return evaluation_times, states, 0.02
        if step == 0.01:
            states[0, -1] = 1e-6
            return evaluation_times, states, 0.03
        return evaluation_times, states, 0.05

    monkeypatch.setattr(simulator, "_integrate_rk45", fake_rk45)
    monkeypatch.setattr(simulator, "_integrate_rk4", fake_rk4)

    selection = simulator.select_integrator(
        initial_state=np.zeros(12),
        t_eval=evaluation_times,
    )

    assert selection == IntegratorSelection(
        method="rk4",
        rk4_step=0.01,
        elapsed_seconds=0.03,
        max_state_error=1e-6,
        final_twist_error=0.0,
        reference_elapsed_seconds=0.10,
    )

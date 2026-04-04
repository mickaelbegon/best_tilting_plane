"""Tests for the prescribed asymmetric arm-motion helper."""

import math

import pytest

from best_tilting_plane.simulation import PrescribedArmMotion, TwistOptimizationVariables


def test_left_arm_uses_the_fixed_motion_window() -> None:
    """The left arm should always move between 0.0 s and 0.3 s."""

    motion = PrescribedArmMotion(
        TwistOptimizationVariables(
            right_arm_start=0.15,
            left_plane_initial=math.radians(-20.0),
            left_plane_final=math.radians(10.0),
            right_plane_initial=math.radians(30.0),
            right_plane_final=math.radians(40.0),
        )
    )

    initial = motion.left(0.0)
    final = motion.left(0.3)

    assert motion.left_end == pytest.approx(0.3)
    assert initial.elevation.position == pytest.approx(-math.pi)
    assert final.elevation.position == pytest.approx(0.0)
    assert initial.elevation_plane.position == pytest.approx(math.radians(-20.0))
    assert final.elevation_plane.position == pytest.approx(math.radians(10.0))


def test_right_arm_uses_the_decision_variable_start_time() -> None:
    """The right arm should stay still before its start time and finish after 0.3 s."""

    motion = PrescribedArmMotion(
        TwistOptimizationVariables(
            right_arm_start=0.2,
            left_plane_initial=0.0,
            left_plane_final=0.0,
            right_plane_initial=math.radians(-45.0),
            right_plane_final=math.radians(15.0),
        )
    )

    before = motion.right(0.1)
    start = motion.right(0.2)
    end = motion.right(0.5)

    assert motion.right_end == pytest.approx(0.5)
    assert before.elevation.position == pytest.approx(math.pi)
    assert before.elevation.velocity == pytest.approx(0.0)
    assert start.elevation_plane.position == pytest.approx(math.radians(-45.0))
    assert end.elevation_plane.position == pytest.approx(math.radians(15.0))
    assert end.elevation.position == pytest.approx(0.0)


def test_prescribed_arm_motion_rejects_non_positive_duration() -> None:
    """The motion helper should validate the prescribed movement duration."""

    with pytest.raises(ValueError, match="strictly positive"):
        PrescribedArmMotion(
            TwistOptimizationVariables(
                right_arm_start=0.0,
                left_plane_initial=0.0,
                left_plane_final=0.0,
                right_plane_initial=0.0,
                right_plane_final=0.0,
            ),
            duration=0.0,
        )

"""Tests for the reduced De Leva anthropometric table."""

import pytest

from best_tilting_plane.anthropometry import DeLevaSex, de_leva_segment_table
from best_tilting_plane.anthropometry.de_leva import total_body_fraction


@pytest.mark.parametrize("sex", [DeLevaSex.MALE, DeLevaSex.FEMALE])
def test_reduced_de_leva_table_covers_full_body_mass(sex: DeLevaSex) -> None:
    """The reduced segment set should still represent the full body mass."""

    assert total_body_fraction(sex) == pytest.approx(1.0)


def test_de_leva_table_exposes_expected_segments() -> None:
    """The reduced table should expose all segments needed by the future model builder."""

    table = de_leva_segment_table()

    assert set(table) == {
        "head",
        "trunk",
        "upper_arm",
        "forearm",
        "hand",
        "thigh",
        "shank",
        "foot",
    }

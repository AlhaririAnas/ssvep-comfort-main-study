from __future__ import annotations

import pytest

from core import conditions


def test_condition_markers_follow_formula() -> None:
    """Condition markers should use the documented 100/150 bases."""

    catalog = conditions.assign_condition_ids(
        [
            conditions.build_frequency_pretest_condition_spec(
                frequency_hz=12.0,
                luminance_mod_kind="square",
                luminance_depth=1.0,
            ),
            conditions.build_frequency_pretest_condition_spec(
                frequency_hz=8.57,
                luminance_mod_kind="square",
                luminance_depth=1.0,
            ),
        ]
    )

    for condition in catalog["conditions"]:
        condition_id = condition["condition_id"]
        assert condition["start_marker"] == 100 + condition_id
        assert condition["stop_marker"] == 150 + condition_id


def test_condition_ids_are_deterministic_after_sorting() -> None:
    """Input order should not change assigned condition IDs."""

    spec_a = conditions.build_frequency_pretest_condition_spec(
        frequency_hz=8.57,
        luminance_mod_kind="square",
        luminance_depth=1.0,
    )
    spec_b = conditions.build_frequency_pretest_condition_spec(
        frequency_hz=12.0,
        luminance_mod_kind="square",
        luminance_depth=1.0,
    )

    first = conditions.assign_condition_ids([spec_a, spec_b])
    second = conditions.assign_condition_ids([spec_b, spec_a])

    assert [row["condition_key"] for row in first["conditions"]] == [
        row["condition_key"] for row in second["conditions"]
    ]
    assert [row["condition_id"] for row in first["conditions"]] == [1, 2]


def test_multi_target_cue_markers_use_201_to_204() -> None:
    """Four target cue markers should be 201 through 204."""

    assert [conditions.multi_target_cue_marker(i) for i in range(4)] == [201, 202, 203, 204]


def test_multi_target_cue_marker_rejects_out_of_range_index() -> None:
    """Cue marker helper should reject invalid target indices."""

    with pytest.raises(ValueError, match="target_idx"):
        conditions.multi_target_cue_marker(4)


def test_idle_angle_markers_start_at_211() -> None:
    """Idle markers should start after the cue marker range."""

    assert [conditions.idle_angle_marker(i) for i in range(3)] == [211, 212, 213]


def test_attach_condition_fields_adds_backward_compatible_aliases() -> None:
    """Condition fields should include current names and legacy aliases."""

    catalog = conditions.assign_condition_ids(
        [
            conditions.build_frequency_pretest_condition_spec(
                frequency_hz=15.0,
                luminance_mod_kind="sine",
                luminance_depth=0.5,
            )
        ]
    )
    row: dict[str, object] = {}

    conditions.attach_condition_fields(row, catalog["conditions"][0])

    assert row["frequency_hz"] == 15.0
    assert row["stim_type"] == row["stimulus_type"]
    assert row["stim_shape"] == row["stimulus_shape"]
    assert row["lum_mod_kind"] == row["luminance_mod_kind"]
    assert row["lum_depth"] == row["luminance_depth"]

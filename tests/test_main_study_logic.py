from __future__ import annotations

from pathlib import Path

from core.conditions import CONDITION_MARKER_START_BASE, CONDITION_MARKER_STOP_BASE
from experiments.main_study import (
    _build_refresh_frequency_audit,
    _compute_remaining_break_s,
    _load_target_positions,
    _main_study_block_marker,
    _participant_order_index,
    _resolve_active_block_order,
)


def test_main_study_block_markers_are_disjoint_from_trial_markers() -> None:
    block_markers = {
        _main_study_block_marker(block_idx, role=role)
        for block_idx in range(1, 9)
        for role in ("start", "stop")
    }
    trial_markers = {
        CONDITION_MARKER_START_BASE + block_idx
        for block_idx in range(1, 9)
    } | {
        CONDITION_MARKER_STOP_BASE + block_idx
        for block_idx in range(1, 9)
    }

    assert block_markers.isdisjoint(trial_markers)


def test_remaining_break_is_computed_from_block_end() -> None:
    elapsed_s, remaining_s = _compute_remaining_break_s(
        break_total_s=45.0,
        block_end_monotonic=100.0,
        now_monotonic=120.0,
    )

    assert elapsed_s == 20.0
    assert remaining_s == 25.0


def test_remaining_break_clamps_to_zero_after_budget_is_used() -> None:
    elapsed_s, remaining_s = _compute_remaining_break_s(
        break_total_s=45.0,
        block_end_monotonic=100.0,
        now_monotonic=160.0,
    )

    assert elapsed_s == 60.0
    assert remaining_s == 0.0


def test_participant_order_index_uses_numeric_suffix_zero_based() -> None:
    assert _participant_order_index("P01", 7) == 0
    assert _participant_order_index("P07", 7) == 6
    assert _participant_order_index("P08", 7) == 0


def test_balanced_rotation_keeps_condition_ids_but_changes_presentation_order() -> None:
    blocks = [
        {"block_id": "B1", "stimulus_id": 101},
        {"block_id": "B2", "stimulus_id": 102},
        {"block_id": "B3", "stimulus_id": 103},
    ]

    ordered, meta = _resolve_active_block_order(
        blocks,
        proband_id="P02",
        strategy="balanced_rotation_by_proband",
        seed_base=42,
    )

    assert [block["block_id"] for block in ordered] == ["B2", "B3", "B1"]
    assert [block["presentation_order_idx"] for block in ordered] == [1, 2, 3]
    assert meta["canonical_block_ids"] == ["B1", "B2", "B3"]
    assert meta["presented_block_ids"] == ["B2", "B3", "B1"]


def test_shuffle_first_n_keep_tail_leaves_reference_last() -> None:
    blocks = [
        {"block_id": "B1", "stimulus_id": 101},
        {"block_id": "B2", "stimulus_id": 102},
        {"block_id": "B3", "stimulus_id": 103},
        {"block_id": "B7", "stimulus_id": 299},
    ]

    ordered, meta = _resolve_active_block_order(
        blocks,
        proband_id="P02",
        strategy="shuffle_first_n_keep_tail",
        seed_base=42,
        lock_tail_count=1,
    )

    assert ordered[-1]["block_id"] == "B7"
    assert set(block["block_id"] for block in ordered[:-1]) == {"B1", "B2", "B3"}
    assert meta["locked_tail_count"] == 1


def test_standard_target_positions_follow_runtime_spacing() -> None:
    session_cfg = {
        "target_positions_deg": [
            [-9.0, 0.0],
            [0.0, 9.0],
            [9.0, 0.0],
            [0.0, -9.0],
        ]
    }

    positions = _load_target_positions(session_cfg, spacing_deg=7.5)

    assert positions == [(-7.5, 0.0), (0.0, 7.5), (7.5, 0.0), (0.0, -7.5)]


def test_custom_target_positions_are_preserved() -> None:
    session_cfg = {
        "target_positions_deg": [
            [-8.0, 1.0],
            [1.0, 8.0],
            [8.0, -1.0],
            [-1.0, -8.0],
        ]
    }

    positions = _load_target_positions(session_cfg, spacing_deg=7.5)

    assert positions == [(-8.0, 1.0), (1.0, 8.0), (8.0, -1.0), (-1.0, -8.0)]


def test_main_study_sine_depths_are_eighty_percent() -> None:
    stimuli = Path("configs/main_study_stimuli.json").read_text(encoding="utf-8")

    assert "sine, 90%" not in stimuli
    assert '"luminance_depth": 0.9' not in stimuli
    assert "sine, 80%" in stimuli


def test_refresh_frequency_audit_flags_odd_square_frame_periods() -> None:
    audit = _build_refresh_frequency_audit([12.0, 15.0, 20.0], refresh_hz=60.0, trial_duration_s=5.0)
    by_freq = {row["frequency_hz"]: row for row in audit["per_frequency"]}

    assert by_freq[12.0]["nearest_integer_period_frames"] == 5
    assert by_freq[12.0]["square_cycle_balance"] == "uneven_square_duty_possible"
    assert by_freq[15.0]["nearest_integer_period_frames"] == 4
    assert by_freq[15.0]["square_cycle_balance"] == "balanced_even_frame_cycle"
    assert by_freq[20.0]["nearest_integer_period_frames"] == 3


def test_german_questionnaire_text_keeps_umlauts() -> None:
    source = Path("src/questionnaires/main_study.py").read_text(encoding="utf-8")

    assert "Blöcken" in source
    assert "Qualitätsschwelle" in source
    assert "weiß" in source
    assert "Frühere" in source

from __future__ import annotations

from experiments.frequency_pretest import _estimate_session_duration_min as estimate_freqtest_min
from experiments.main_study import _estimate_session_duration_min as estimate_mainstudy_min
from experiments.stimulus_screening import _estimate_session_duration_min as estimate_screening_min


def test_frequency_pretest_duration_estimate_uses_dynamic_ranges_and_buffer() -> None:
    estimated = estimate_freqtest_min(
        baseline_open_s=30.0,
        total_trials=12,
        stim_duration_s=5.0,
        fix_range_s=(1.0, 3.0),
        iti_range_s=(1.0, 2.0),
        block_count=4,
    )

    expected = (30.0 + 12 * (5.0 + 2.0 + 1.5) + 4 * 5.0 + 5.0 * 60.0) / 60.0
    assert estimated == expected


def test_stimulus_screening_duration_estimate_counts_long_breaks_once() -> None:
    estimated = estimate_screening_min(
        baseline_open_s=20.0,
        baseline_closed_s=15.0,
        include_eyes_closed_baseline=True,
        total_trials=20,
        total_trial_stimulus_s=100.0,
        fix_range_s=(1.0, 3.0),
        iti_range_s=(2.0, 4.0),
        block_break_s=30.0,
        long_break_s=90.0,
        long_break_after_blocks=(2,),
        block_count=4,
    )

    expected_breaks = 30.0 + 90.0 + 30.0
    expected = (20.0 + 15.0 + 100.0 + 20 * (2.0 + 3.0) + expected_breaks + 4 * 60.0 + 5.0 * 60.0) / 60.0
    assert estimated == expected


def test_main_study_duration_estimate_is_dynamic_and_includes_safety_buffer() -> None:
    estimated = estimate_mainstudy_min(
        baseline_open_s=30.0,
        baseline_closed_s=20.0,
        include_eyes_closed_baseline=True,
        active_block_count=3,
        trials_per_block=8,
        cue_duration_s=1.0,
        trial_duration_s=5.0,
        fix_range_s=(1.0, 3.0),
        iti_range_s=(1.0, 2.0),
        block_break_s=45.0,
        long_break_s=120.0,
        long_break_after_blocks=(2,),
        idle_block_count=1,
        idle_trials_total=6,
        idle_post_stop_hold_s=0.1,
        impedance_stable_s_initial=5.0,
        impedance_stable_s_between=3.0,
    )

    active_trials = 3 * 8 * (1.0 + 5.0 + 2.0 + 1.5)
    breaks = 45.0 + 120.0
    between_impedance = 2 * 3.0
    idle = 6 * 5.0 + 0.1
    expected = (30.0 + 20.0 + 5.0 + active_trials + idle + breaks + between_impedance + 3 * 60.0 + 180.0 + 5.0 * 60.0) / 60.0
    assert estimated == expected

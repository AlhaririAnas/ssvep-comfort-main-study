from __future__ import annotations

import pytest

from stimuli.timing import build_continuous_trial_frame_plan


def test_continuous_idle_plan_has_five_second_boundaries_at_144_hz() -> None:
    """Six idle trials should mark 0, 5, 10, ..., 30 s."""

    plan = build_continuous_trial_frame_plan(
        trial_duration_s=5.0,
        trial_count=6,
        refresh_hz=144.0,
        post_stop_hold_s=0.1,
    )

    assert plan.trial_frames == 720
    assert plan.final_marker_frame == 4320
    assert plan.post_stop_frames == 14
    assert plan.events_by_frame[0] == (("start", 1),)
    assert plan.events_by_frame[720] == (("stop", 1), ("start", 2))
    assert plan.events_by_frame[3600] == (("stop", 5), ("start", 6))
    assert plan.events_by_frame[4320] == (("stop", 6),)
    assert plan.last_frame == 4334


def test_continuous_idle_plan_rejects_invalid_values() -> None:
    """Invalid timing values should fail before the visual loop starts."""

    with pytest.raises(ValueError, match="trial_duration_s"):
        build_continuous_trial_frame_plan(
            trial_duration_s=0.0,
            trial_count=6,
            refresh_hz=144.0,
        )
    with pytest.raises(ValueError, match="trial_count"):
        build_continuous_trial_frame_plan(
            trial_duration_s=5.0,
            trial_count=0,
            refresh_hz=144.0,
        )

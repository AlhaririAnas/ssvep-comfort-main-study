from __future__ import annotations

"""Frame plans for continuous multi-target stimulus runs."""

from dataclasses import dataclass
from typing import Dict, Tuple


MarkerEvent = Tuple[str, int]


@dataclass(frozen=True)
class ContinuousTrialFramePlan:
    """Frame-exact marker plan for visual trials without gaps."""

    trial_frames: int
    trial_count: int
    post_stop_frames: int
    final_marker_frame: int
    last_frame: int
    events_by_frame: Dict[int, Tuple[MarkerEvent, ...]]


def build_continuous_trial_frame_plan(
    *,
    trial_duration_s: float,
    trial_count: int,
    refresh_hz: float,
    post_stop_hold_s: float = 0.1,
) -> ContinuousTrialFramePlan:
    """Return marker frames for continuous trials.

    Trial 1 starts on frame 0. For every boundary, the stop marker of the
    previous trial is sent first, followed by the start marker of the next
    trial on the same flip. The final stop marker is sent on the exact trial
    boundary, while the stimulus keeps running for a short visual hold.
    """

    trial_duration = float(trial_duration_s)
    n_trials = int(trial_count)
    hz = float(refresh_hz)
    post_hold = max(0.0, float(post_stop_hold_s))
    if trial_duration <= 0.0:
        raise ValueError("trial_duration_s must be > 0.")
    if n_trials <= 0:
        raise ValueError("trial_count must be >= 1.")
    if hz <= 1.0:
        raise ValueError("refresh_hz must be > 1.")

    trial_frames = max(1, int(round(trial_duration * hz)))
    post_stop_frames = max(0, int(round(post_hold * hz)))
    final_marker_frame = trial_frames * n_trials
    last_frame = final_marker_frame + post_stop_frames

    mutable_events: Dict[int, list[MarkerEvent]] = {0: [("start", 1)]}
    for trial_idx in range(1, n_trials):
        boundary_frame = trial_frames * trial_idx
        mutable_events.setdefault(boundary_frame, []).append(("stop", trial_idx))
        mutable_events[boundary_frame].append(("start", trial_idx + 1))
    mutable_events.setdefault(final_marker_frame, []).append(("stop", n_trials))

    events_by_frame = {
        frame: tuple(events)
        for frame, events in sorted(mutable_events.items())
    }
    return ContinuousTrialFramePlan(
        trial_frames=trial_frames,
        trial_count=n_trials,
        post_stop_frames=post_stop_frames,
        final_marker_frame=final_marker_frame,
        last_frame=last_frame,
        events_by_frame=events_by_frame,
    )

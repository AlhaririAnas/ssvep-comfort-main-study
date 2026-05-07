from __future__ import annotations

"""
PsychoPy display helpers and frame-timing utilities.

This module contains small display helpers only. It does not create trials,
does not choose stimuli, and does not define experiment conditions.
"""

import time
import sys
from pathlib import Path
from typing import Any, Sequence

from psychopy import event, visual

try:
    from core.config import MONITOR_REFRESH_RATE, TEXT_COLOR, TEXT_HEIGHT_DEG, TEXT_WRAP_DEG
except ModuleNotFoundError:  # pragma: no cover - supports direct module demos
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.config import MONITOR_REFRESH_RATE, TEXT_COLOR, TEXT_HEIGHT_DEG, TEXT_WRAP_DEG


def seconds_to_frames(seconds: float, refresh_hz: float) -> int:
    """Convert seconds to the nearest frame count with a minimum of one."""

    return max(1, int(round(float(seconds) * float(refresh_hz))))


def estimate_realized_frequency(target_hz: float, refresh_hz: float) -> dict[str, Any]:
    """Estimate the realised flicker frequency for frame-based presentation."""

    frames_per_cycle = max(1, int(round(float(refresh_hz) / float(target_hz))))
    realized = float(refresh_hz) / frames_per_cycle
    return {
        "target_hz": round(float(target_hz), 6),
        "frames_per_cycle": frames_per_cycle,
        "realized_hz": round(realized, 6),
        "error_hz": round(realized - float(target_hz), 6),
    }


def measure_refresh_rate(
    win: visual.Window,
    n_identical: int = 60,
    n_max: int = 200,
    expected_hz: float = MONITOR_REFRESH_RATE,
    max_deviation_hz: float = 2.0,
    measured_out: list[float | None] | None = None,
) -> float:
    """Measure refresh rate, but keep the configured rate if measurement disagrees."""

    for _ in range(5):
        win.flip()
    hz = win.getActualFrameRate(nIdentical=n_identical, nMaxFrames=n_max)
    if measured_out is not None:
        measured_out.append(float(hz) if hz else None)
    if hz is None or hz <= 1.0:
        return float(expected_hz)
    measured_hz = float(hz)
    if abs(measured_hz - float(expected_hz)) > float(max_deviation_hz):
        return float(expected_hz)
    return measured_hz


def check_escape() -> bool:
    """Return True if Escape was pressed since the last call."""

    return bool(event.getKeys(keyList=["escape"]))


def draw_all(stimuli: Sequence[visual.BaseVisualStim]) -> None:
    """Draw all stimuli in order."""

    for stimulus in stimuli:
        stimulus.draw()


def show_fixation(
    win: visual.Window,
    fixation: Sequence[visual.BaseVisualStim],
    n_frames: int,
) -> bool:
    """Show fixation for a fixed number of frames."""

    for _ in range(n_frames):
        if check_escape():
            return True
        draw_all(fixation)
        win.flip()
    return False


def show_blank(win: visual.Window, n_frames: int) -> bool:
    """Show a blank screen for a fixed number of frames."""

    for _ in range(n_frames):
        if check_escape():
            return True
        win.flip()
    return False


def show_message(
    win: visual.Window,
    info_stim: visual.TextStim,
    text: str,
    wait_keys: list[str] | None = None,
) -> str | None:
    """Display text and wait for one accepted key."""

    if wait_keys is None:
        wait_keys = ["space", "escape"]
    info_stim.text = text
    info_stim.draw()
    win.flip()
    keys = event.waitKeys(keyList=wait_keys)
    return keys[0] if keys else None


def make_info_stim(win: visual.Window) -> visual.TextStim:
    """Create the standard instruction text stimulus."""

    return visual.TextStim(
        win,
        text="",
        color=TEXT_COLOR,
        height=TEXT_HEIGHT_DEG,
        wrapWidth=TEXT_WRAP_DEG,
        units="deg",
    )


def schedule_marker_on_next_flip(
    win: visual.Window,
    marker: Any,
    *,
    value: int,
    label: str,
    flip_time_out: list[float] | None = None,
    marker_time_ms_out: list[float] | None = None,
) -> None:
    """
    Schedule marker injection on the next PsychoPy flip.

    The callback captures time.monotonic() first and then converts it through
    marker.capture_flip_time_ms(). This preserves the existing timing design.
    """

    def _callback() -> None:
        flip_time = time.monotonic()
        marker_time_ms = marker.capture_flip_time_ms(flip_time)

        if flip_time_out is not None:
            flip_time_out[0] = flip_time
        if marker_time_ms_out is not None:
            marker_time_ms_out[0] = marker_time_ms

        marker.inject_marker(
            value=str(value),
            label=label,
            timestamp_ms=marker_time_ms,
            capture_method="flip_callback",
        )

    win.callOnFlip(_callback)


if __name__ == "__main__":
    print("Display module demo")
    print("This demo does not open a PsychoPy window.")
    print(f"Frames for 5 s at 60 Hz: {seconds_to_frames(5.0, 60.0)}")
    print(f"Frequency estimate: {estimate_realized_frequency(12.0, 60.0)}")

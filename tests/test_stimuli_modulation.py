from __future__ import annotations

import numpy as np

from stimuli.modulation import clip_unit, shifted_dark_state


def test_shifted_dark_state_keeps_bright_state_and_reduces_difference() -> None:
    """Depth 0.6 should keep 60% of the original dark-bright difference."""

    dark = np.asarray([26, 26, 26], dtype=np.float32)
    bright = np.asarray([191, 255, 191], dtype=np.float32)

    shifted = shifted_dark_state(dark, bright, 0.6)

    np.testing.assert_allclose(shifted, [92, 117.6, 92])
    np.testing.assert_allclose(bright - shifted, (bright - dark) * 0.6)


def test_shifted_dark_state_matches_project_example_after_rounding() -> None:
    """The user's 60% project example rounds to [92, 118, 92]."""

    shifted = shifted_dark_state(
        np.asarray([26, 26, 26], dtype=np.float32),
        np.asarray([191, 255, 191], dtype=np.float32),
        0.6,
    )

    assert [round(float(value)) for value in shifted] == [92, 118, 92]


def test_clip_unit_keeps_alpha_values_valid() -> None:
    """Invalid intermediate values should not produce invalid opacity."""

    assert clip_unit(-1.0) == 0.0
    assert clip_unit(0.6) == 0.6
    assert clip_unit(2.0) == 1.0

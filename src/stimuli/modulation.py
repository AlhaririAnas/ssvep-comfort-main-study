from __future__ import annotations

"""Small math helpers for visual modulation depth."""

import numpy as np


def clip_unit(value: float) -> float:
    """Clip one scalar to the valid 0..1 alpha range."""

    return float(np.clip(float(value), 0.0, 1.0))


def shifted_dark_state(
    dark: np.ndarray,
    bright: np.ndarray,
    depth: float,
) -> np.ndarray:
    """Move the dark state toward the bright state by modulation depth.

    Here, `depth` means the remaining proportion of the original dark-bright
    difference. The bright state stays unchanged.

    Example in 8-bit RGB:
    dark=[26, 26, 26], bright=[191, 255, 191], depth=0.6
    -> dark=[92, 118, 92], bright=[191, 255, 191].
    """

    depth = clip_unit(depth)
    return np.asarray(bright, dtype=np.float32) - (
        np.asarray(bright, dtype=np.float32) - np.asarray(dark, dtype=np.float32)
    ) * depth

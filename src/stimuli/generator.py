from __future__ import annotations

"""
stimulus_generator.py
=====================
Visual stimulus renderer for SSVEP, SSMVEP, and hybrid BCI experiments.

Main design rules
-----------------
- Background is fixed dark gray.
- Timing is frame-based.
- The refresh rate is measured once at startup and then treated as fixed.
- SSMVEP motion uses sine modulation and full depth.
- Comments are simple and in English.
"""

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image
from psychopy import core, event, monitors, visual

try:
    from core.config import (
        DEFAULT_SCREEN_ID,
        DIST_CM,
        MONITOR_NAME,
        MONITOR_REFRESH_RATE,
        PROJECT_ROOT,
        SCREEN_SIZE_PX,
        SCREEN_W_CM,
    )
    from stimuli.modulation import clip_unit, shifted_dark_state
    from stimuli.timing import build_continuous_trial_frame_plan
except ModuleNotFoundError:  # pragma: no cover - supports direct module demos
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.config import (
        DEFAULT_SCREEN_ID,
        DIST_CM,
        MONITOR_NAME,
        MONITOR_REFRESH_RATE,
        PROJECT_ROOT,
        SCREEN_SIZE_PX,
        SCREEN_W_CM,
    )
    from stimuli.modulation import clip_unit, shifted_dark_state
    from stimuli.timing import build_continuous_trial_frame_plan

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
FPS = MONITOR_REFRESH_RATE
SIZE_DEG = 5.0
TEX_SIZE = 512
V2_TEX_SIZE = 1024
V2_PHASE_STEPS = 96
DEFAULT_DURATION_S = 5.0
DEFAULT_FREQ_HZ = 15.0
DEFAULT_ISI_S = 0.5
DEFAULT_SCREEN_INDEX = DEFAULT_SCREEN_ID

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
BG_COL = (-0.8, -0.8, -0.8)
STIM_ON_COL = (0.5, 1.0, 0.5)
STIM_OFF_COL = BG_COL

ON_NP = np.asarray(STIM_ON_COL, dtype=np.float32)
OFF_NP = np.asarray(STIM_OFF_COL, dtype=np.float32)
BG_NP = np.asarray(BG_COL, dtype=np.float32)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _mix_rgb(c0: Sequence[float], c1: Sequence[float], t: float) -> Tuple[float, float, float]:
    arr = (1.0 - t) * np.asarray(c0, dtype=np.float32) + t * np.asarray(c1, dtype=np.float32)
    return float(arr[0]), float(arr[1]), float(arr[2])


def _cfg_float(cfg: Mapping[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        value = cfg.get(key)
        if value is not None:
            return float(value)
    return float(default)


def _cfg_int(cfg: Mapping[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        value = cfg.get(key)
        if value is not None:
            return int(value)
    return int(default)


def _cfg_str(cfg: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = cfg.get(key)
        if value is not None:
            return str(value)
    return default


def _cfg_xy(
    cfg: Mapping[str, Any],
    *keys: str,
    default: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    for key in keys:
        value = cfg.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            raise ValueError(f"Position field {key!r} must be a string or a 2-item list.")
        if len(parts) != 2:
            raise ValueError(f"Position field {key!r} must contain exactly two values.")
        return float(parts[0]), float(parts[1])
    return float(default[0]), float(default[1])


def _coerce_rgb(
    value: Any,
    *,
    default: Sequence[float],
) -> Tuple[float, float, float]:
    if value is None:
        parts = list(default)
    elif isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError("RGB values must be a 3-item list, tuple, or comma-separated string.")
    if len(parts) != 3:
        raise ValueError("RGB values must contain exactly three numbers.")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _circle_mask(size: int = TEX_SIZE) -> np.ndarray:
    h = size // 2
    y, x = np.mgrid[-h:h, -h:h].astype(np.float32)
    return (np.hypot(x, y) <= h).astype(np.float32)


def _to_pil(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    u8 = ((rgb + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    alpha = (mask * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([u8, alpha]), "RGBA")


def _resolve_asset_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    rel_path = Path(str(path_str).lstrip("./\\"))
    cwd_path = (Path.cwd() / rel_path)
    if cwd_path.exists():
        return cwd_path
    return PROJECT_ROOT / rel_path


def _image_with_white_background_alpha(path: Path, threshold: float = 0.96) -> Image.Image:
    """Load a bitmap and make near-white background pixels transparent."""

    img = Image.open(path).convert("RGBA")
    rgba = np.asarray(img, dtype=np.float32) / 255.0
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    white_background = np.mean(rgb, axis=2) >= float(threshold)
    alpha = np.where(white_background, 0.0, alpha)
    out = np.dstack([rgb, alpha])
    return Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8), "RGBA")


def _polar_grid(size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) / ((size - 1) / 2.0)
    x, y = np.meshgrid(coords, coords)
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    mask = (radius <= 1.0).astype(np.float32)
    return radius, theta, mask


def _blend_to_on(
    weight: np.ndarray,
    *,
    off_np: np.ndarray = OFF_NP,
    on_np: np.ndarray = ON_NP,
) -> np.ndarray:
    clipped = np.clip(weight, 0.0, 1.0).astype(np.float32)
    return (1.0 - clipped[:, :, None]) * off_np + clipped[:, :, None] * on_np


def _binary_pattern_imgs(
    flag: np.ndarray,
    mask: np.ndarray,
    *,
    on_np: np.ndarray = ON_NP,
    off_np: np.ndarray = OFF_NP,
    bg_np: np.ndarray = BG_NP,
) -> Tuple[Image.Image, Image.Image]:
    flag = np.asarray(flag, dtype=bool)
    rgb_a = np.where(flag[:, :, None], on_np, off_np)
    rgb_b = np.where(flag[:, :, None], off_np, on_np)
    return (
        _to_pil(np.where(mask[:, :, None] > 0, rgb_a, bg_np), mask),
        _to_pil(np.where(mask[:, :, None] > 0, rgb_b, bg_np), mask),
    )



def _spiral_checker_imgs_v2(
    n_radial: int = 16,
    n_angular: int = 24,
    twist: float = 0.35,
    spiral_depth: float = 0.25,
    min_radius_frac: float = 0.05,
    tex_size: int = V2_TEX_SIZE,
    edge_sigma_px: float = 1.5,
    on_np: np.ndarray = ON_NP,
    off_np: np.ndarray = OFF_NP,
    bg_np: np.ndarray = BG_NP,
) -> Tuple[Image.Image, Image.Image]:
    """Log-polar spiral checkerboard with controlled centre and AA.

    Key fixes over the original:
    - `min_radius_frac` default raised from 0.01 to 0.05 to avoid the
      numerical singularity at the centre.  Pixels inside this radius are
      assigned to the background, eliminating the discontinuity artefact.
    - Edge softening via Gaussian blur on the binary flag suppresses
      high-frequency Moiré when the texture is down-scaled on screen.
    - Both polarities are returned for full contrast reversal.
    """
    radius, theta, mask = _polar_grid(tex_size)
    min_radius_frac = float(np.clip(min_radius_frac, 0.02, 0.5))
    spiral_depth = float(np.clip(spiral_depth, 0.0, 1.0))
    # -- mask out the singular centre --
    centre_mask = (radius >= min_radius_frac).astype(np.float32)
    combined_mask = mask * centre_mask
    radius_eval = np.clip(radius, min_radius_frac, 1.0)
    log_radius = np.log(radius_eval / min_radius_frac) / math.log(1.0 / min_radius_frac)
    log_radius = np.clip(log_radius, 0.0, 0.999999)
    linear_radius = np.clip((radius_eval - min_radius_frac) / (1.0 - min_radius_frac), 0.0, 0.999999)
    radial_coord = (1.0 - spiral_depth) * linear_radius + spiral_depth * log_radius
    theta_spiral = theta + spiral_depth * twist * np.log(radius_eval)
    angular_norm = (theta_spiral / (2.0 * math.pi)) % 1.0
    radial_idx = np.floor(radial_coord * max(1, n_radial)).astype(int)
    angular_idx = np.floor(angular_norm * max(1, n_angular)).astype(int)
    flag = ((radial_idx + angular_idx) % 2) == 0

    if edge_sigma_px > 0.0:
        from scipy.ndimage import gaussian_filter
        flag_f = gaussian_filter(flag.astype(np.float32), sigma=edge_sigma_px)
        flag_f = np.clip(flag_f, 0.0, 1.0)
        rgb_a = (1.0 - flag_f[:, :, None]) * off_np + flag_f[:, :, None] * on_np
        rgb_b = flag_f[:, :, None] * off_np + (1.0 - flag_f[:, :, None]) * on_np
        return (
            _to_pil(np.where(combined_mask[:, :, None] > 0, rgb_a, bg_np), combined_mask),
            _to_pil(np.where(combined_mask[:, :, None] > 0, rgb_b, bg_np), combined_mask),
        )
    return _binary_pattern_imgs(flag, combined_mask, on_np=on_np, off_np=off_np, bg_np=bg_np)



def _checker_imgs(
    n: int = 8,
    *,
    on_np: np.ndarray = ON_NP,
    off_np: np.ndarray = OFF_NP,
    bg_np: np.ndarray = BG_NP,
) -> Tuple[Image.Image, Image.Image]:
    s = TEX_SIZE
    i, j = np.mgrid[0:s, 0:s]
    grid = ((i * n // s) + (j * n // s)) % 2
    mask = _circle_mask(s)
    rgb_a = np.where(grid[:, :, None] == 0, off_np, on_np)
    rgb_b = np.where(grid[:, :, None] == 0, on_np, off_np)
    return (
        _to_pil(np.where(mask[:, :, None] > 0, rgb_a, bg_np), mask),
        _to_pil(np.where(mask[:, :, None] > 0, rgb_b, bg_np), mask),
    )


def _spoke_imgs(
    n: int = 8,
    *,
    on_np: np.ndarray = ON_NP,
    off_np: np.ndarray = OFF_NP,
    bg_np: np.ndarray = BG_NP,
) -> Tuple[Image.Image, Image.Image]:
    s = TEX_SIZE
    h = s // 2
    y, x = np.mgrid[-h:h, -h:h].astype(np.float32)
    sector = ((np.arctan2(y, x) + math.pi) / (2.0 * math.pi) * n).astype(int) % n
    mask = _circle_mask(s)
    flag = (sector % 2) == 0
    rgb_a = np.where(flag[:, :, None], off_np, on_np)
    rgb_b = np.where(flag[:, :, None], on_np, off_np)
    return (
        _to_pil(np.where(mask[:, :, None] > 0, rgb_a, bg_np), mask),
        _to_pil(np.where(mask[:, :, None] > 0, rgb_b, bg_np), mask),
    )


def _ring_imgs(
    n: int = 4,
    *,
    on_np: np.ndarray = ON_NP,
    off_np: np.ndarray = OFF_NP,
    bg_np: np.ndarray = BG_NP,
) -> Tuple[Image.Image, Image.Image]:
    s = TEX_SIZE
    h = s // 2
    y, x = np.mgrid[-h:h, -h:h].astype(np.float32)
    idx = (np.hypot(x, y) / h * n).astype(int)
    mask = _circle_mask(s)
    flag = (idx % 2) == 0
    rgb_a = np.where(flag[:, :, None], off_np, on_np)
    rgb_b = np.where(flag[:, :, None], on_np, off_np)
    return (
        _to_pil(np.where(mask[:, :, None] > 0, rgb_a, bg_np), mask),
        _to_pil(np.where(mask[:, :, None] > 0, rgb_b, bg_np), mask),
    )


def _newton_frames(
    n: int = 4,
    n_frames: int = 6,
    *,
    on_np: np.ndarray = ON_NP,
    off_np: np.ndarray = OFF_NP,
    bg_np: np.ndarray = BG_NP,
) -> List[Image.Image]:
    s = TEX_SIZE
    h = s // 2
    y, x = np.mgrid[-h:h, -h:h].astype(np.float32)
    radius = np.hypot(x, y) / h
    mask = _circle_mask(s)
    frames: List[Image.Image] = []
    for step in range(n_frames):
        phase = step / float(n_frames)
        idx = np.floor(radius * n - phase * n).astype(int)
        flag = (idx % 2) == 0
        rgb = np.where(flag[:, :, None], off_np, on_np)
        frames.append(_to_pil(np.where(mask[:, :, None] > 0, rgb, bg_np), mask))
    return frames


def _sample_circle_vertices(radius: float, n_vertices: int = 128) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * math.pi, n_vertices, endpoint=False)
    return np.c_[radius * np.cos(ang), radius * np.sin(ang)]


def _sample_square_vertices(radius: float, n_vertices: int = 128) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * math.pi, n_vertices, endpoint=False)
    denom = np.maximum(np.abs(np.cos(ang)), np.abs(np.sin(ang)))
    rr = radius / np.maximum(denom, 1e-6)
    return np.c_[rr * np.cos(ang), rr * np.sin(ang)]


def make_window(
    fullscr: bool = True,
    screen: int = DEFAULT_SCREEN_INDEX,
    background_color: Sequence[float] = BG_COL,
) -> visual.Window:
    mon = monitors.Monitor(MONITOR_NAME, width=SCREEN_W_CM, distance=DIST_CM)
    mon.setSizePix(SCREEN_SIZE_PX)
    window_size = SCREEN_SIZE_PX
    if not fullscr:
        aspect = SCREEN_SIZE_PX[0] / max(float(SCREEN_SIZE_PX[1]), 1.0)
        width = min(int(SCREEN_SIZE_PX[0]), 1280)
        height = int(round(width / aspect))
        window_size = (width, height)
    win = visual.Window(
        size=window_size,
        monitor=mon,
        fullscr=fullscr,
        screen=screen,
        color=_coerce_rgb(background_color, default=BG_COL),
        colorSpace="rgb",
        units="deg",
        allowGUI=not fullscr,
        winType="pyglet",
        waitBlanking=True,
    )
    win.mouseVisible = False
    return win


def measure_refresh_rate(
    win: visual.Window,
    fallback_hz: float = FPS,
    max_deviation_hz: float = 2.0,
    measured_out: list[float | None] | None = None,
) -> float:
    """Measure refresh rate, but keep the configured rate if measurement disagrees."""

    measured = win.getActualFrameRate(nIdentical=20, nMaxFrames=240, nWarmUpFrames=20, threshold=1.0)
    if measured_out is not None:
        measured_out.append(float(measured) if measured else None)
    if measured is None or measured <= 1.0:
        return float(fallback_hz)
    measured_hz = float(measured)
    if abs(measured_hz - float(fallback_hz)) > float(max_deviation_hz):
        return float(fallback_hz)
    return measured_hz


def build_fixation_stimuli(win: visual.Window) -> List[visual.Rect]:
    kw = dict(lineColor=None, units="deg")
    return [
        visual.Rect(win, width=0.50, height=0.10, fillColor="white", **kw),
        visual.Rect(win, width=0.10, height=0.50, fillColor="white", **kw),
        visual.Rect(win, width=0.38, height=0.06, fillColor="black", **kw),
        visual.Rect(win, width=0.06, height=0.38, fillColor="black", **kw),
    ]


def normalize_stimulus_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    if "duration" not in out and "duration_s" in out:
        out["duration"] = out["duration_s"]
    if "freq" not in out and "frequency_hz" in out:
        out["freq"] = out["frequency_hz"]
    if "type" not in out and "category" in out:
        out["type"] = out["category"]
    return out


def validate_stimulus_cfg(
    cfg: Mapping[str, Any],
    supported_shapes: Sequence[str] | None = None,
) -> Dict[str, Any]:
    out = normalize_stimulus_cfg(cfg)

    shape = _cfg_str(out, "shape", default="").strip()
    if not shape:
        raise ValueError("Stimulus config is missing 'shape'.")
    if supported_shapes is not None and shape not in supported_shapes:
        raise ValueError(f"Unsupported stimulus shape: {shape!r}")

    duration = _cfg_float(out, "duration", default=DEFAULT_DURATION_S)
    freq = _cfg_float(out, "freq", default=DEFAULT_FREQ_HZ)
    if duration <= 0.0:
        raise ValueError(f"Stimulus duration must be > 0. Got {duration}.")
    if freq <= 0.0:
        raise ValueError(f"Stimulus frequency must be > 0. Got {freq}.")

    out["shape"] = shape
    out["duration"] = duration
    out["freq"] = freq
    return out


def load_stimuli(
    path: Path,
    supported_shapes: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError("stimuli.json must contain a list of stimulus configs.")

    stimuli = [validate_stimulus_cfg(cfg, supported_shapes) for cfg in raw]
    stimuli.sort(key=lambda x: (float(x.get("freq", DEFAULT_FREQ_HZ)), int(x.get("id", 0))))
    return stimuli


class StimulusGenerator:
    _DISPATCH = {
        "flicker": "_run_flicker",
        "rectangle_flicker": "_run_rectangle_flicker",
        "checkerboard": "_run_checkerboard",
        "checkerboard_v2": "_run_checkerboard_v2",
        "circle_rings": "_run_circle_rings",
        "spokes": "_run_spokes",
        "image_alternation": "_run_image_alternation",
        "radial_zoom": "_run_radial_zoom",
        "rotation": "_run_rotation",
        "newtons_rings": "_run_newtons_rings",
        "zoom_chromatic_flicker": "_run_zoom_chromatic_flicker",
        "rotation_center_flicker": "_run_rotation_center_flicker",
        "chromatic_morph_hybrid": "_run_chromatic_morph_hybrid",
        "oscillating_prs": "_run_oscillating_prs",
        "center_surround": "_run_center_surround",
    }

    # Shapes supported by the 4-target simultaneous renderer used in the
    # main study. These shapes have an
    # independent per-target updater (no shared-state assumptions inside
    # the trial loop).
    _MULTI_TARGET_SHAPES: Tuple[str, ...] = (
        "flicker",
        "rectangle_flicker",
        "checkerboard",
        "checkerboard_v2",
        "newtons_rings",
        "image_alternation",
        "radial_zoom",
    )

    def __init__(
        self,
        win: visual.Window,
        monitor_refresh_rate: float = FPS,
        luminance_depth: float = 0.75,
        luminance_mod_kind: str = "sine",
        background_color: Sequence[float] = BG_COL,
        stim_on_color: Sequence[float] = STIM_ON_COL,
        stim_off_color: Sequence[float] | None = None,
        abort_check: Callable[[], bool] | None = None,
    ) -> None:
        if monitor_refresh_rate <= 1.0:
            raise ValueError("monitor_refresh_rate must be > 1 Hz.")
        if not 0.0 <= luminance_depth <= 1.0:
            raise ValueError("luminance_depth must be in [0, 1].")
        if luminance_mod_kind not in {"square", "sine", "triangle", "sawtooth"}:
            raise ValueError(
                "luminance_mod_kind must be one of: square, sine, triangle, sawtooth."
            )

        self.win = win
        self.monitor_refresh_rate = float(monitor_refresh_rate)
        self.luminance_depth = float(luminance_depth)
        self.luminance_mod_kind = str(luminance_mod_kind)
        self.background_color = _coerce_rgb(background_color, default=BG_COL)
        self.stim_on_color = _coerce_rgb(stim_on_color, default=STIM_ON_COL)
        self.stim_off_color = _coerce_rgb(stim_off_color, default=self.background_color)
        self._background_np = np.asarray(self.background_color, dtype=np.float32)
        self._stim_on_np = np.asarray(self.stim_on_color, dtype=np.float32)
        self._stim_off_np = np.asarray(self.stim_off_color, dtype=np.float32)
        self._fixation = build_fixation_stimuli(win)
        self.abort_check = abort_check

    def set_colors(
        self,
        stim_on_color: Sequence[float] | None = None,
        stim_off_color: Sequence[float] | None = None,
        background_color: Sequence[float] | None = None,
    ) -> None:
        """Update stimulus colors in-place (takes effect on the next trial build)."""
        if stim_on_color is not None:
            self.stim_on_color = _coerce_rgb(stim_on_color, default=self.stim_on_color)
            self._stim_on_np = np.asarray(self.stim_on_color, dtype=np.float32)
        if stim_off_color is not None:
            self.stim_off_color = _coerce_rgb(stim_off_color, default=self.stim_off_color)
            self._stim_off_np = np.asarray(self.stim_off_color, dtype=np.float32)
        if background_color is not None:
            self.background_color = _coerce_rgb(background_color, default=self.background_color)
            self._background_np = np.asarray(self.background_color, dtype=np.float32)

    def _luminance_off_np(self) -> np.ndarray:
        return shifted_dark_state(self._stim_off_np, self._stim_on_np, self.luminance_depth)

    def _luminance_off_color(self) -> Tuple[float, float, float]:
        arr = self._luminance_off_np()
        return float(arr[0]), float(arr[1]), float(arr[2])

    @classmethod
    def supported_shapes(cls) -> Tuple[str, ...]:
        return tuple(cls._DISPATCH.keys())

    def _draw_fix(self) -> None:
        for stim in self._fixation:
            stim.draw()

    def _escape_pressed(self) -> bool:
        return bool(event.getKeys(keyList=["escape"]))

    def _abort_requested(self) -> bool:
        return bool(self.abort_check is not None and self.abort_check())

    def _circle(
        self,
        radius: float,
        fill_color: Any,
        pos: Tuple[float, float] = (0.0, 0.0),
        edges: int = 128,
    ) -> visual.Circle:
        return visual.Circle(
            self.win,
            radius=radius,
            pos=pos,
            edges=edges,
            fillColor=fill_color,
            lineColor=None,
            units="deg",
        )

    def _rect(
        self,
        size: float,
        fill_color: Any,
        pos: Tuple[float, float] = (0.0, 0.0),
    ) -> visual.Rect:
        return visual.Rect(
            self.win,
            width=size,
            height=size,
            pos=pos,
            fillColor=fill_color,
            lineColor=None,
            units="deg",
        )

    def _shape(
        self,
        vertices: np.ndarray,
        fill_color: Any,
        pos: Tuple[float, float] = (0.0, 0.0),
    ) -> visual.ShapeStim:
        return visual.ShapeStim(
            self.win,
            vertices=vertices,
            pos=pos,
            fillColor=fill_color,
            lineColor=None,
            units="deg",
            interpolate=True,
            closeShape=True,
        )

    def _img_stim(
        self,
        image: Any,
        size: float = SIZE_DEG,
        pos: Tuple[float, float] = (0.0, 0.0),
        ori: float = 0.0,
        interpolate: bool = False,
    ) -> visual.ImageStim:
        return visual.ImageStim(
            self.win,
            image=image,
            size=size,
            pos=pos,
            ori=ori,
            units="deg",
            interpolate=interpolate,
        )

    def _single_image_stim(
        self,
        cfg: Mapping[str, Any],
        *,
        interpolate: bool = True,
    ) -> visual.BaseVisualStim:
        size = self._size(cfg)
        pos = self._pos(cfg)
        path_a = _resolve_asset_path(_cfg_str(cfg, "image_a", default="./images/emoji_A.png"))
        if path_a.exists():
            img = _image_with_white_background_alpha(path_a)
            return self._img_stim(img, size=size, pos=pos, interpolate=interpolate)
        return self._circle(radius=size / 2.0, fill_color=self.stim_on_color, pos=pos)

    def _duration(self, cfg: Mapping[str, Any]) -> float:
        return _cfg_float(cfg, "duration", default=DEFAULT_DURATION_S)

    def _freq(self, cfg: Mapping[str, Any]) -> float:
        return _cfg_float(cfg, "freq", default=DEFAULT_FREQ_HZ)

    def _size(self, cfg: Mapping[str, Any]) -> float:
        return _cfg_float(cfg, "size_deg", default=SIZE_DEG)

    def _stimulus_footprint(self, cfg: Mapping[str, Any]) -> float:
        size_deg = _cfg_float(cfg, "size_deg", default=0.0)
        zoom_size_deg = _cfg_float(cfg, "size_max_deg", "size_min_deg", default=0.0)
        surround_size_deg = 2.0 * _cfg_float(
            cfg,
            "surround_max_deg",
            "surround_min_deg",
            "center_radius_deg",
            default=0.0,
        )
        return max(size_deg, zoom_size_deg, surround_size_deg, SIZE_DEG)

    def _screen_size_deg(self) -> Tuple[float, float]:
        width_px = float(self.win.size[0]) if self.win.size is not None else float(SCREEN_SIZE_PX[0])
        height_px = float(self.win.size[1]) if self.win.size is not None else float(SCREEN_SIZE_PX[1])
        width_deg = math.degrees(2.0 * math.atan(SCREEN_W_CM / (2.0 * DIST_CM)))
        height_cm = SCREEN_W_CM * (height_px / max(width_px, 1.0))
        height_deg = math.degrees(2.0 * math.atan(height_cm / (2.0 * DIST_CM)))
        return width_deg, height_deg

    def _position_from_preset(self, cfg: Mapping[str, Any], preset: str) -> Tuple[float, float]:
        width_deg, height_deg = self._screen_size_deg()
        margin_deg = _cfg_float(cfg, "position_margin_deg", default=1.0)
        half_size_deg = self._stimulus_footprint(cfg) / 2.0
        x_edge = max(0.0, width_deg / 2.0 - half_size_deg - margin_deg)
        y_edge = max(0.0, height_deg / 2.0 - half_size_deg - margin_deg)
        mapping = {
            "center": (0.0, 0.0),
            "upper_right": (x_edge, y_edge),
            "upper_left": (-x_edge, y_edge),
            "lower_right": (x_edge, -y_edge),
            "lower_left": (-x_edge, -y_edge),
        }
        try:
            return mapping[preset]
        except KeyError as exc:
            raise ValueError(
                "position_preset must be one of: center, upper_right, upper_left, lower_right, lower_left."
            ) from exc

    def _pos(self, cfg: Mapping[str, Any]) -> Tuple[float, float]:
        if any(cfg.get(key) is not None for key in ("position_deg", "pos_deg")):
            return _cfg_xy(cfg, "position_deg", "pos_deg", default=(0.0, 0.0))
        preset = _cfg_str(cfg, "position_preset", default="center").strip().lower().replace("-", "_")
        return self._position_from_preset(cfg, preset or "center")

    def _phase_cycles(self, frame_idx: int, freq_hz: float, phase_cycles: float = 0.0) -> float:
        return (frame_idx * freq_hz / self.monitor_refresh_rate + phase_cycles) % 1.0

    def _wave(self, frame_idx: int, freq_hz: float, mod_kind: str, phase_cycles: float = 0.0) -> float:
        phase = self._phase_cycles(frame_idx, freq_hz, phase_cycles)

        if mod_kind == "square":
            return 1.0 if phase < 0.5 else 0.0
        if mod_kind == "sine":
            return 0.5 * (math.sin(2.0 * math.pi * phase) + 1.0)
        if mod_kind == "triangle":
            return 1.0 - abs(2.0 * phase - 1.0)
        if mod_kind == "sawtooth":
            return phase
        raise ValueError(f"Unsupported mod_kind={mod_kind!r}")

    def _alpha_lum(self, frame_idx: int, freq_hz: float) -> float:
        raw = self._wave(frame_idx, freq_hz, self.luminance_mod_kind)
        return clip_unit(raw)

    def _image_opacity_lum(self, frame_idx: int, freq_hz: float) -> float:
        raw = self._alpha_lum(frame_idx, freq_hz)
        depth = clip_unit(self.luminance_depth)
        return float((1.0 - depth) + depth * raw)

    def _alpha_motion(self, frame_idx: int, freq_hz: float) -> float:
        return float(self._wave(frame_idx, freq_hz, "sine"))

    def _state_lum(self, frame_idx: int, freq_hz: float) -> bool:
        return self._wave(frame_idx, freq_hz, self.luminance_mod_kind) >= 0.5

    def _index_motion(self, frame_idx: int, freq_hz: float, n_states: int) -> int:
        if n_states <= 1:
            return 0
        alpha = self._alpha_motion(frame_idx, freq_hz)
        return int(min(n_states - 1, max(0, round(alpha * (n_states - 1)))))

    def _progress(self, frame_idx: float, n_frames: int) -> float:
        if n_frames <= 1:
            return 1.0
        return float(np.clip(frame_idx / float(n_frames - 1), 0.0, 1.0))

    def _signed_motion(self, frame_idx: int, freq_hz: float) -> float:
        return 2.0 * self._alpha_motion(frame_idx, freq_hz) - 1.0

    def _draw_modulated_pair(
        self,
        stim_a: visual.BaseVisualStim,
        stim_b: visual.BaseVisualStim,
        alpha: float,
        *,
        binary_ok: bool = False,
    ) -> None:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if binary_ok and self.luminance_mod_kind == "square" and self.luminance_depth >= 0.999:
            if alpha >= 0.5:
                stim_b.draw()
            else:
                stim_a.draw()
            return
        stim_a.opacity = 1.0 - alpha
        stim_b.opacity = alpha
        stim_a.draw()
        stim_b.draw()

    def _run_trial_loop(self, duration_s: float, update_fn: Callable[[float, int], None]) -> None:
        n_frames = max(1, int(round(duration_s * self.monitor_refresh_rate)))
        event.clearEvents(eventType="keyboard")
        if hasattr(self.win, "frameIntervals"):
            self.win.frameIntervals = []
        self.win.recordFrameIntervals = True
        start_flip_time: float | None = None
        last_flip_time: float | None = None
        try:
            while True:
                if self._escape_pressed() or self._abort_requested():
                    break
                if start_flip_time is None or last_flip_time is None:
                    elapsed_s = 0.0
                else:
                    elapsed_s = max(0.0, last_flip_time - start_flip_time)
                sample_idx = elapsed_s * self.monitor_refresh_rate
                update_fn(sample_idx, n_frames)
                self._draw_fix()
                flip_time = self.win.flip()
                if flip_time is None:
                    flip_time = core.getTime()
                if start_flip_time is None:
                    start_flip_time = flip_time
                last_flip_time = flip_time
                if last_flip_time - start_flip_time >= duration_s:
                    break
        finally:
            self.win.recordFrameIntervals = False

    def run_stimulus(self, cfg: Dict[str, Any]) -> None:
        cfg = validate_stimulus_cfg(cfg, self.supported_shapes())
        method_name = self._DISPATCH[cfg["shape"]]
        getattr(self, method_name)(cfg)

    # ------------------------------------------------------------------
    # Multi-target (4 simultaneous stimuli, one frequency per target)
    # ------------------------------------------------------------------

    @classmethod
    def supported_multi_target_shapes(cls) -> Tuple[str, ...]:
        return cls._MULTI_TARGET_SHAPES

    def _build_multi_target_updater(
        self,
        base_cfg: Mapping[str, Any],
        freq_hz: float,
        position_deg: Tuple[float, float],
    ) -> Callable[[float], None]:
        """Build an `update(frame_idx)` closure for one of four simultaneous
        targets. The returned function updates state and draws the stimulus;
        callers issue a single `win.flip()` after calling all four.
        """
        shape = str(base_cfg.get("shape", "")).strip().lower()
        if shape not in self._MULTI_TARGET_SHAPES:
            raise ValueError(
                f"Shape {shape!r} is not supported in multi-target mode. "
                f"Supported: {self._MULTI_TARGET_SHAPES}"
            )

        cfg = dict(base_cfg)
        cfg["freq"] = float(freq_hz)
        cfg["position_deg"] = (float(position_deg[0]), float(position_deg[1]))

        if shape == "flicker":
            stim_a = self._circle(
                radius=self._size(cfg) / 2.0,
                fill_color=self._luminance_off_color(),
                pos=cfg["position_deg"],
            )
            stim_b = self._circle(
                radius=self._size(cfg) / 2.0,
                fill_color=self.stim_on_color,
                pos=cfg["position_deg"],
            )

            def update(frame_idx: float) -> None:
                alpha = self._alpha_lum(frame_idx, freq_hz)
                stim_a.opacity = 1.0 - alpha
                stim_b.opacity = alpha
                stim_a.draw()
                stim_b.draw()

            return update

        if shape == "rectangle_flicker":
            stim_a = self._rect(
                size=self._size(cfg),
                fill_color=self._luminance_off_color(),
                pos=cfg["position_deg"],
            )
            stim_b = self._rect(
                size=self._size(cfg),
                fill_color=self.stim_on_color,
                pos=cfg["position_deg"],
            )

            def update(frame_idx: float) -> None:
                alpha = self._alpha_lum(frame_idx, freq_hz)
                stim_a.opacity = 1.0 - alpha
                stim_b.opacity = alpha
                stim_a.draw()
                stim_b.draw()

            return update

        if shape == "checkerboard_v2":
            size = self._size(cfg)
            n_radial = _cfg_int(cfg, "n_radial", "n_rings", default=16)
            n_angular = _cfg_int(cfg, "n_angular", "n_spokes", default=24)
            twist = _cfg_float(cfg, "spiral_twist", default=1.2)
            spiral_depth = _cfg_float(cfg, "spiral_depth", default=1.0)
            min_radius_frac = _cfg_float(cfg, "min_radius_frac", default=0.01)
            tex_size = _cfg_int(cfg, "tex_size", default=V2_TEX_SIZE)
            edge_sigma = _cfg_float(cfg, "edge_sigma_px", default=0.0)
            img_a, img_b = _spiral_checker_imgs_v2(
                n_radial=n_radial,
                n_angular=n_angular,
                twist=twist,
                spiral_depth=spiral_depth,
                min_radius_frac=min_radius_frac,
                tex_size=tex_size,
                edge_sigma_px=edge_sigma,
                on_np=self._stim_on_np,
                off_np=self._luminance_off_np(),
                bg_np=self._background_np,
            )
            stim_a = self._img_stim(img_a, size=size, pos=cfg["position_deg"], interpolate=True)
            stim_b = self._img_stim(img_b, size=size, pos=cfg["position_deg"], interpolate=True)

            def update(frame_idx: float) -> None:
                alpha = self._alpha_lum(frame_idx, freq_hz)
                self._draw_modulated_pair(stim_a, stim_b, alpha, binary_ok=True)

            return update

        if shape == "newtons_rings":
            n_rings = _cfg_int(cfg, "n_rings", default=4)
            # Fixed, high state count decoupled from monitor/frequency ratio.
            # A low n_frames (e.g. 4 at 15 Hz / 60 Hz) combined with the sine
            # envelope's plateaus produces uneven dwell times and visible
            # "hanging". 24 states yield fine-grained quantization for every
            # supported frequency with negligible memory cost.
            n_frames = 24
            frames = [
                self._img_stim(img, size=self._size(cfg), pos=cfg["position_deg"])
                for img in _newton_frames(
                    n_rings,
                    n_frames=n_frames,
                    on_np=self._stim_on_np,
                    off_np=self._stim_off_np,
                    bg_np=self._background_np,
                )
            ]
            n_states = len(frames)

            def update(frame_idx: float) -> None:
                frames[self._index_motion(frame_idx, freq_hz, n_states)].draw()

            return update

        if shape == "image_alternation":
            # Uses a single image with luminance modulation.
            cfg_with_pos = dict(cfg)
            cfg_with_pos["position_deg"] = cfg["position_deg"]
            stim = self._single_image_stim(cfg_with_pos, interpolate=True)

            def update(frame_idx: float) -> None:
                stim.opacity = self._image_opacity_lum(frame_idx, freq_hz)
                stim.draw()

            return update

        if shape == "checkerboard":
            size = self._size(cfg)
            checks = _cfg_int(cfg, "checks_per_row", default=8)
            img_a, img_b = _checker_imgs(
                checks,
                on_np=self._stim_on_np,
                off_np=self._luminance_off_np(),
                bg_np=self._background_np,
            )
            stim_a = self._img_stim(img_a, size=size, pos=cfg["position_deg"])
            stim_b = self._img_stim(img_b, size=size, pos=cfg["position_deg"])

            def update(frame_idx: float) -> None:
                alpha = self._alpha_lum(frame_idx, freq_hz)
                stim_a.opacity = 1.0 - alpha
                stim_b.opacity = alpha
                stim_a.draw()
                stim_b.draw()

            return update

        if shape == "radial_zoom":
            r_min = _cfg_float(cfg, "size_min_deg", default=3.6) / 2.0
            r_max = _cfg_float(cfg, "size_max_deg", default=5.2) / 2.0
            circle = self._circle(
                radius=r_min,
                fill_color=self.stim_on_color,
                pos=cfg["position_deg"],
            )

            def update(frame_idx: float) -> None:
                circle.radius = _lerp(r_min, r_max, self._alpha_motion(frame_idx, freq_hz))
                circle.draw()

            return update

        # Should have been caught above.
        raise AssertionError(f"Unhandled multi-target shape: {shape}")

    def run_multi_target_trial(
        self,
        base_cfg: Mapping[str, Any],
        frequencies_hz: Sequence[float],
        positions_deg: Sequence[Tuple[float, float]],
        duration_s: float,
        *,
        on_first_flip: Callable[[float], None] | None = None,
        cue_target_idx: int | None = None,
        cue_marker_radius_deg: float = 0.35,
    ) -> Dict[str, Any]:
        """Render four stimuli simultaneously for `duration_s` seconds.

        Parameters
        ----------
        base_cfg : Mapping
            Shape definition shared across all four targets (e.g. same
            `shape`, `size_deg`, shape-specific texture parameters).
            The `luminance_mod_kind` and `luminance_depth` come from the
            generator instance (set at construction).
        frequencies_hz : sequence of float
            One frequency per target. Length = number of targets.
        positions_deg : sequence of (x, y)
            Screen positions per target (same length as `frequencies_hz`).
        duration_s : float
            Stimulation duration in seconds.
        on_first_flip : callable, optional
            Invoked exactly once, on the window's first flip timestamp, with
            the flip time in seconds (core.getTime). Use this for marker
            injection without adding callOnFlip plumbing on the call site.
        cue_target_idx : int, optional
            If set, a small white dot is drawn near the cued target for the
            entire trial to support gaze-on-target feedback. Typically the
            session runner shows a dedicated cue phase BEFORE the stimulation
            and leaves this parameter None during the stimulation itself.
        """
        freqs = [float(f) for f in frequencies_hz]
        positions = [(float(p[0]), float(p[1])) for p in positions_deg]
        if len(freqs) != len(positions):
            raise ValueError("frequencies_hz and positions_deg must have equal length.")
        if not freqs:
            raise ValueError("run_multi_target_trial needs at least one target.")

        updaters: List[Callable[[float], None]] = [
            self._build_multi_target_updater(base_cfg, f, p)
            for f, p in zip(freqs, positions)
        ]

        cue_dot = None
        if cue_target_idx is not None:
            if cue_target_idx < 0 or cue_target_idx >= len(positions):
                raise ValueError(
                    f"cue_target_idx {cue_target_idx} out of range [0, {len(positions)-1}]"
                )
            cue_dot = self._circle(
                radius=float(cue_marker_radius_deg),
                fill_color="white",
                pos=positions[cue_target_idx],
            )

        n_frames = max(1, int(round(float(duration_s) * self.monitor_refresh_rate)))
        event.clearEvents(eventType="keyboard")
        if hasattr(self.win, "frameIntervals"):
            self.win.frameIntervals = []
        self.win.recordFrameIntervals = True

        first_flip_time: float | None = None
        start_flip_time: float | None = None
        last_flip_time: float | None = None
        aborted = False
        try:
            while True:
                if self._escape_pressed() or self._abort_requested():
                    aborted = True
                    break
                if start_flip_time is None or last_flip_time is None:
                    elapsed_s = 0.0
                else:
                    elapsed_s = max(0.0, last_flip_time - start_flip_time)
                sample_idx = elapsed_s * self.monitor_refresh_rate

                for update in updaters:
                    update(sample_idx)
                if cue_dot is not None:
                    cue_dot.draw()
                self._draw_fix()

                if first_flip_time is None and on_first_flip is not None:
                    try:
                        self.win.callOnFlip(on_first_flip, core.getTime())
                    except Exception:
                        # callOnFlip signature expects no-arg in older PsychoPy; retry
                        self.win.callOnFlip(on_first_flip)

                flip_time = self.win.flip()
                if flip_time is None:
                    flip_time = core.getTime()
                if first_flip_time is None:
                    first_flip_time = flip_time
                if start_flip_time is None:
                    start_flip_time = flip_time
                last_flip_time = flip_time
                if last_flip_time - start_flip_time >= float(duration_s):
                    break
        finally:
            self.win.recordFrameIntervals = False
            # Explicit framebuffer clearing: the last stimulus frame otherwise
            # persists for one monitor flip after the trial loop exits, which
            # with high-contrast textures (e.g. Newton's rings at 15 Hz) leaves
            # a visible on-screen ring imprint due to LCD pixel response. Two
            # flips guarantee the fresh blank frame actually reaches the panel.
            self._draw_fix()
            self.win.flip()
            self._draw_fix()
            self.win.flip()

        return {
            "aborted": aborted,
            "first_flip_time": first_flip_time,
            "last_flip_time": last_flip_time,
            "duration_s_actual": (last_flip_time - first_flip_time) if (first_flip_time and last_flip_time) else 0.0,
            "frames_requested": n_frames,
            "frequencies_hz": freqs,
            "positions_deg": positions,
            "cue_target_idx": cue_target_idx,
        }

    def run_multi_target_continuous_trials(
        self,
        base_cfg: Mapping[str, Any],
        frequencies_hz: Sequence[float],
        positions_deg: Sequence[Tuple[float, float]],
        trial_duration_s: float,
        trial_count: int,
        *,
        on_trial_start: Callable[[int, float], None] | None = None,
        on_trial_stop: Callable[[int, float], None] | None = None,
        post_stop_hold_s: float = 0.1,
    ) -> Dict[str, Any]:
        """Render separate trials without visual gaps between them.

        This is used for idle data: every 5 s epoch is a real trial with its
        own start and stop marker, but the flicker stream does not reset and no
        blank/ITI is inserted between consecutive trials. A short hold after
        the final stop marker keeps the stimulus visible while the marker is
        sent on the exact final trial boundary.
        """

        freqs = [float(f) for f in frequencies_hz]
        positions = [(float(p[0]), float(p[1])) for p in positions_deg]
        if len(freqs) != len(positions):
            raise ValueError("frequencies_hz and positions_deg must have equal length.")
        if not freqs:
            raise ValueError("run_multi_target_continuous_trials needs at least one target.")
        trial_duration = float(trial_duration_s)
        n_trials = int(trial_count)
        plan = build_continuous_trial_frame_plan(
            trial_duration_s=trial_duration,
            trial_count=n_trials,
            refresh_hz=self.monitor_refresh_rate,
            post_stop_hold_s=post_stop_hold_s,
        )

        updaters: List[Callable[[float], None]] = [
            self._build_multi_target_updater(base_cfg, f, p)
            for f, p in zip(freqs, positions)
        ]

        event.clearEvents(eventType="keyboard")
        if hasattr(self.win, "frameIntervals"):
            self.win.frameIntervals = []
        self.win.recordFrameIntervals = True

        first_flip_time: float | None = None
        start_flip_time: float | None = None
        last_flip_time: float | None = None
        marked_starts: List[int] = []
        marked_stops: List[int] = []
        aborted = False

        def _schedule_callback(callback: Callable[[int, float], None] | None, trial_idx: int) -> None:
            if callback is None:
                return

            def _run_callback() -> None:
                callback(int(trial_idx), core.getTime())

            self.win.callOnFlip(_run_callback)

        try:
            for frame_idx in range(plan.last_frame + 1):
                if self._escape_pressed() or self._abort_requested():
                    aborted = True
                    break

                sample_idx = float(frame_idx)

                for update in updaters:
                    update(sample_idx)
                self._draw_fix()

                for marker_role, trial_idx in plan.events_by_frame.get(frame_idx, ()):
                    if marker_role == "start":
                        _schedule_callback(on_trial_start, trial_idx)
                        marked_starts.append(trial_idx)
                    elif marker_role == "stop":
                        _schedule_callback(on_trial_stop, trial_idx)
                        marked_stops.append(trial_idx)

                flip_time = self.win.flip()
                if flip_time is None:
                    flip_time = core.getTime()
                if first_flip_time is None:
                    first_flip_time = flip_time
                if start_flip_time is None:
                    start_flip_time = flip_time
                last_flip_time = flip_time
        finally:
            self.win.recordFrameIntervals = False
            self._draw_fix()
            self.win.flip()
            self._draw_fix()
            self.win.flip()

        return {
            "aborted": aborted,
            "first_flip_time": first_flip_time,
            "last_flip_time": last_flip_time,
            "duration_s_actual": (last_flip_time - first_flip_time) if (first_flip_time and last_flip_time) else 0.0,
            "trial_duration_s": trial_duration,
            "trial_count": n_trials,
            "trial_frames": plan.trial_frames,
            "post_stop_hold_s": float(post_stop_hold_s),
            "post_stop_hold_frames": plan.post_stop_frames,
            "final_marker_frame": plan.final_marker_frame,
            "marked_trials": marked_starts,
            "marked_starts": marked_starts,
            "marked_stops": marked_stops,
            "frequencies_hz": freqs,
            "positions_deg": positions,
        }

    def run_multi_target_continuous_segments(
        self,
        base_cfg: Mapping[str, Any],
        frequencies_hz: Sequence[float],
        positions_deg: Sequence[Tuple[float, float]],
        segment_duration_s: float,
        segment_count: int,
        *,
        on_segment_flip: Callable[[int, float], None] | None = None,
        on_segment_start: Callable[[int, float], None] | None = None,
        on_segment_stop: Callable[[int, float], None] | None = None,
    ) -> Dict[str, Any]:
        """Compatibility wrapper. Prefer run_multi_target_continuous_trials."""

        return self.run_multi_target_continuous_trials(
            base_cfg=base_cfg,
            frequencies_hz=frequencies_hz,
            positions_deg=positions_deg,
            trial_duration_s=segment_duration_s,
            trial_count=segment_count,
            on_trial_start=on_segment_start or on_segment_flip,
            on_trial_stop=on_segment_stop,
        )

    def run_isi(self, duration_s: float = DEFAULT_ISI_S) -> None:
        if duration_s <= 0.0:
            return
        self._run_trial_loop(duration_s, lambda _f, _n: None)

    # ------------------------------------------------------------------
    # SSVEP
    # ------------------------------------------------------------------

    def _run_flicker(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        stim_a = self._circle(radius=self._size(cfg) / 2.0, fill_color=self._luminance_off_color(), pos=pos)
        stim_b = self._circle(radius=self._size(cfg) / 2.0, fill_color=self.stim_on_color, pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            alpha = self._alpha_lum(frame_idx, hz)
            stim_a.opacity = 1.0 - alpha
            stim_b.opacity = alpha
            stim_a.draw()
            stim_b.draw()

        self._run_trial_loop(dur, update)

    def _run_rectangle_flicker(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        stim_a = self._rect(size=self._size(cfg), fill_color=self._luminance_off_color(), pos=pos)
        stim_b = self._rect(size=self._size(cfg), fill_color=self.stim_on_color, pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            alpha = self._alpha_lum(frame_idx, hz)
            stim_a.opacity = 1.0 - alpha
            stim_b.opacity = alpha
            stim_a.draw()
            stim_b.draw()

        self._run_trial_loop(dur, update)

    def _run_checkerboard(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        checks = _cfg_int(cfg, "checks_per_row", default=8)
        img_a, img_b = _checker_imgs(
            checks,
            on_np=self._stim_on_np,
            off_np=self._luminance_off_np(),
            bg_np=self._background_np,
        )
        stim_a = self._img_stim(img_a, size=self._size(cfg), pos=pos)
        stim_b = self._img_stim(img_b, size=self._size(cfg), pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            alpha = self._alpha_lum(frame_idx, hz)
            stim_a.opacity = 1.0 - alpha
            stim_b.opacity = alpha
            stim_a.draw()
            stim_b.draw()

        self._run_trial_loop(dur, update)

    def _run_checkerboard_v2(self, cfg: Dict[str, Any]) -> None:
        """Spiral checkerboard with time-locked contrast modulation.

        Parameters matched to the earlier reference version:
        - n_radial = 16
        - n_angular = 24
        - spiral_twist = 1.2
        - min_radius_frac = 0.01
        - tex_size = 1024
        """
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        n_radial = _cfg_int(cfg, "n_radial", "n_rings", default=16)
        n_angular = _cfg_int(cfg, "n_angular", "n_spokes", default=24)
        twist = _cfg_float(cfg, "spiral_twist", default=1.2)
        spiral_depth = _cfg_float(cfg, "spiral_depth", default=1.0)
        min_radius_frac = _cfg_float(cfg, "min_radius_frac", default=0.01)
        tex_size = _cfg_int(cfg, "tex_size", default=V2_TEX_SIZE)
        edge_sigma = _cfg_float(cfg, "edge_sigma_px", default=0.0)
        img_a, img_b = _spiral_checker_imgs_v2(
            n_radial=n_radial,
            n_angular=n_angular,
            twist=twist,
            spiral_depth=spiral_depth,
            min_radius_frac=min_radius_frac,
            tex_size=tex_size,
            edge_sigma_px=edge_sigma,
            on_np=self._stim_on_np,
            off_np=self._luminance_off_np(),
            bg_np=self._background_np,
        )
        stim_a = self._img_stim(img_a, size=self._size(cfg), pos=pos, interpolate=True)
        stim_b = self._img_stim(img_b, size=self._size(cfg), pos=pos, interpolate=True)

        def update(frame_idx: float, _n_frames: int) -> None:
            alpha = self._alpha_lum(frame_idx, hz)
            self._draw_modulated_pair(stim_a, stim_b, alpha, binary_ok=True)

        self._run_trial_loop(dur, update)

    def _run_circle_rings(self, cfg: Dict[str, Any]) -> None:
        # Opacity crossfade respects luminance_depth and luminance_mod_kind.
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        n = _cfg_int(cfg, "n_rings", default=4)
        img_a, img_b = _ring_imgs(
            n,
            on_np=self._stim_on_np,
            off_np=self._luminance_off_np(),
            bg_np=self._background_np,
        )
        stim_a = self._img_stim(img_a, size=self._size(cfg), pos=pos)
        stim_b = self._img_stim(img_b, size=self._size(cfg), pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            alpha = self._alpha_lum(frame_idx, hz)
            stim_a.opacity = 1.0 - alpha
            stim_b.opacity = alpha
            stim_a.draw()
            stim_b.draw()

        self._run_trial_loop(dur, update)

    def _run_spokes(self, cfg: Dict[str, Any]) -> None:
        # Opacity crossfade respects luminance_depth and luminance_mod_kind.
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        n = _cfg_int(cfg, "n_spokes", default=8)
        img_a, img_b = _spoke_imgs(
            n,
            on_np=self._stim_on_np,
            off_np=self._luminance_off_np(),
            bg_np=self._background_np,
        )
        stim_a = self._img_stim(img_a, size=self._size(cfg), pos=pos)
        stim_b = self._img_stim(img_b, size=self._size(cfg), pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            alpha = self._alpha_lum(frame_idx, hz)
            stim_a.opacity = 1.0 - alpha
            stim_b.opacity = alpha
            stim_a.draw()
            stim_b.draw()

        self._run_trial_loop(dur, update)

    def _run_image_alternation(self, cfg: Dict[str, Any]) -> None:
        # Single-image luminance modulation respects luminance_depth and luminance_mod_kind.
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        stim = self._single_image_stim(cfg, interpolate=True)

        def update(frame_idx: float, _n_frames: int) -> None:
            stim.opacity = self._image_opacity_lum(frame_idx, hz)
            stim.draw()

        self._run_trial_loop(dur, update)

    # ------------------------------------------------------------------
    # SSMVEP
    # ------------------------------------------------------------------

    def _run_radial_zoom(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        r_min = _cfg_float(cfg, "size_min_deg", default=3.6) / 2.0
        r_max = _cfg_float(cfg, "size_max_deg", default=5.2) / 2.0
        pos = self._pos(cfg)
        circle = self._circle(radius=r_min, fill_color=self.stim_on_color, pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            circle.radius = _lerp(r_min, r_max, self._alpha_motion(frame_idx, hz))
            circle.draw()

        self._run_trial_loop(dur, update)

    def _run_rotation(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        n = _cfg_int(cfg, "n_spokes", default=8)
        angle_step = _cfg_float(cfg, "angle_step_deg", default=22.5)
        img, _ = _spoke_imgs(
            n,
            on_np=self._stim_on_np,
            off_np=self._stim_off_np,
            bg_np=self._background_np,
        )
        stim = self._img_stim(img, size=self._size(cfg), pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            stim.ori = _lerp(-angle_step, angle_step, self._alpha_motion(frame_idx, hz))
            stim.draw()

        self._run_trial_loop(dur, update)

    def _run_newtons_rings(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        n = _cfg_int(cfg, "n_rings", default=4)
        frames = [
            self._img_stim(img, size=self._size(cfg), pos=pos)
            for img in _newton_frames(
                n,
                n_frames=24,
                on_np=self._stim_on_np,
                off_np=self._stim_off_np,
                bg_np=self._background_np,
            )
        ]

        def update(frame_idx: float, _n_frames: int) -> None:
            frames[self._index_motion(frame_idx, hz, len(frames))].draw()

        self._run_trial_loop(dur, update)

    # ------------------------------------------------------------------
    # Hybrid
    # ------------------------------------------------------------------

    def _run_zoom_chromatic_flicker(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        r_min = _cfg_float(cfg, "size_min_deg", default=3.6) / 2.0
        r_max = _cfg_float(cfg, "size_max_deg", default=5.2) / 2.0
        pos = self._pos(cfg)
        stim_a = self._circle(radius=r_min, fill_color=self._luminance_off_color(), pos=pos)
        stim_b = self._circle(radius=r_min, fill_color=self.stim_on_color, pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            alpha_motion = self._alpha_motion(frame_idx, hz)
            alpha_lum = self._alpha_lum(frame_idx, hz)
            radius = _lerp(r_min, r_max, alpha_motion)
            stim_a.radius = radius
            stim_b.radius = radius
            stim_a.opacity = 1.0 - alpha_lum
            stim_b.opacity = alpha_lum
            stim_a.draw()
            stim_b.draw()

        self._run_trial_loop(dur, update)

    def _run_rotation_center_flicker(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        n = _cfg_int(cfg, "n_spokes", default=8)
        angle_step = _cfg_float(cfg, "angle_step_deg", default=22.5)
        center_r = _cfg_float(cfg, "center_radius_deg", default=1.0)
        center_mask_scale = _cfg_float(cfg, "center_mask_scale", default=1.05)

        img, _ = _spoke_imgs(
            n,
            on_np=self._stim_on_np,
            off_np=self._luminance_off_np(),
            bg_np=self._background_np,
        )
        stim = self._img_stim(img, size=self._size(cfg), pos=pos)

        core_bg = self._circle(radius=center_r * center_mask_scale, fill_color=self.background_color, pos=pos)
        stim_a = self._circle(radius=center_r, fill_color=self._luminance_off_color(), pos=pos)
        stim_b = self._circle(radius=center_r, fill_color=self.stim_on_color, pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            stim.ori = _lerp(-angle_step, angle_step, self._alpha_motion(frame_idx, hz))
            stim.draw()
            core_bg.draw()

            alpha_lum = self._alpha_lum(frame_idx, hz)
            stim_a.opacity = 1.0 - alpha_lum
            stim_b.opacity = alpha_lum
            stim_a.draw()
            stim_b.draw()

        self._run_trial_loop(dur, update)

    def _run_chromatic_morph_hybrid(self, cfg: Dict[str, Any]) -> None:
        # Motion drives shape morphing, luminance drives color modulation.
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        radius = self._size(cfg) / 2.0
        vertices_circle = _sample_circle_vertices(radius, n_vertices=128)
        vertices_square = _sample_square_vertices(radius, n_vertices=128)
        stim = self._shape(vertices_circle, self.stim_on_color, pos=pos)

        def update(frame_idx: float, n_frames: int) -> None:
            progress = self._progress(frame_idx, n_frames)
            alpha_motion = self._alpha_motion(frame_idx, hz)
            alpha_lum = self._alpha_lum(frame_idx, hz)
            stim.vertices = (1.0 - progress) * vertices_circle + progress * vertices_square
            stim.fillColor = _mix_rgb(self._luminance_off_color(), self.stim_on_color, alpha_lum)
            stim.draw()

        self._run_trial_loop(dur, update)

    def _run_oscillating_prs(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        base_x, base_y = self._pos(cfg)
        shift = _cfg_float(cfg, "position_shift_deg", default=0.25)
        checks = _cfg_int(cfg, "checks_per_row", default=8)
        img_a, img_b = _checker_imgs(
            checks,
            on_np=self._stim_on_np,
            off_np=self._luminance_off_np(),
            bg_np=self._background_np,
        )
        stim_a = self._img_stim(img_a, size=self._size(cfg), pos=(base_x, base_y))
        stim_b = self._img_stim(img_b, size=self._size(cfg), pos=(base_x, base_y))

        def update(frame_idx: float, _n_frames: int) -> None:
            alpha_motion = self._alpha_motion(frame_idx, hz)
            alpha_lum = self._alpha_lum(frame_idx, hz)
            x_pos = _lerp(-shift, shift, alpha_motion)
            stim_a.pos = (base_x + x_pos, base_y)
            stim_b.pos = (base_x + x_pos, base_y)
            stim_a.opacity = 1.0 - alpha_lum
            stim_b.opacity = alpha_lum
            stim_a.draw()
            stim_b.draw()

        self._run_trial_loop(dur, update)

    def _run_center_surround(self, cfg: Dict[str, Any]) -> None:
        dur = self._duration(cfg)
        hz = self._freq(cfg)
        pos = self._pos(cfg)
        center_r = _cfg_float(cfg, "center_radius_deg", default=1.0)
        surround_min = _cfg_float(cfg, "surround_min_deg", default=2.0)
        surround_max = _cfg_float(cfg, "surround_max_deg", default=2.5)
        hole_scale = _cfg_float(cfg, "center_mask_scale", default=1.01)

        outer = self._circle(radius=surround_max, fill_color=self.stim_on_color, pos=pos)
        hole = self._circle(radius=center_r * hole_scale, fill_color=self.background_color, pos=pos)
        stim_a = self._circle(radius=center_r, fill_color=self._luminance_off_color(), pos=pos)
        stim_b = self._circle(radius=center_r, fill_color=self.stim_on_color, pos=pos)

        def update(frame_idx: float, _n_frames: int) -> None:
            alpha_motion = self._alpha_motion(frame_idx, hz)
            alpha_lum = self._alpha_lum(frame_idx, hz)
            outer.radius = _lerp(surround_min, surround_max, alpha_motion)
            outer.draw()
            hole.draw()
            stim_a.opacity = 1.0 - alpha_lum
            stim_b.opacity = alpha_lum
            stim_a.draw()
            stim_b.draw()

        self._run_trial_loop(dur, update)


if __name__ == "__main__":
    print("Stimulus generator module demo")
    print("This demo does not open a PsychoPy window and does not render stimuli.")
    print(f"Default refresh rate: {FPS:g} Hz")
    print(f"Supported shapes: {', '.join(StimulusGenerator.supported_shapes())}")

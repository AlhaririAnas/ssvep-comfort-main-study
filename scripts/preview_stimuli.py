from __future__ import annotations

"""Preview configured visual stimuli without headset or Cortex connection."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.conditions import resolve_active_modulation  # noqa: E402
from core.config import (  # noqa: E402
    DEFAULT_FULLSCREEN,
    DEFAULT_MS_BACKGROUND_COLOR,
    DEFAULT_MS_SESSION_CONFIG,
    DEFAULT_MS_STIM_OFF_COLOR,
    DEFAULT_MS_STIM_ON_COLOR,
    DEFAULT_MS_STIMULI_JSON,
    DEFAULT_SCREEN_ID,
)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview all stimuli from a JSON config without headset.",
    )
    parser.add_argument("--stimuli-json", default=str(DEFAULT_MS_STIMULI_JSON), help="Stimulus JSON file.")
    parser.add_argument("--session-json", default=str(DEFAULT_MS_SESSION_CONFIG), help="Session JSON for colors.")
    parser.add_argument("--duration", type=float, default=1.5, help="Preview duration per stimulus in seconds.")
    parser.add_argument("--isi", type=float, default=0.4, help="Blank gap between stimuli in seconds.")
    parser.add_argument("--screen-id", type=int, default=DEFAULT_SCREEN_ID, help="PsychoPy screen index.")
    parser.add_argument("--windowed", action="store_true", help="Use a window instead of fullscreen.")
    parser.add_argument(
        "--luminance-depth",
        type=float,
        default=None,
        help="Optional SSVEP depth override. If omitted, each stimulus config is used.",
    )
    parser.add_argument(
        "--luminance-mod-kind",
        choices=("square", "sine", "triangle", "sawtooth"),
        default=None,
        help="Optional SSVEP modulation waveform override.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from stimuli.generator import StimulusGenerator, load_stimuli, make_window, measure_refresh_rate

    stimuli_path = Path(args.stimuli_json)
    if not stimuli_path.is_absolute():
        stimuli_path = PROJECT_ROOT / stimuli_path
    session_path = Path(args.session_json)
    if not session_path.is_absolute():
        session_path = PROJECT_ROOT / session_path

    session_cfg = _load_json_object(session_path)
    background_color = session_cfg.get("background_color") or list(DEFAULT_MS_BACKGROUND_COLOR)
    stim_on_color = session_cfg.get("stim_on_color") or list(DEFAULT_MS_STIM_ON_COLOR)
    stim_off_color = session_cfg.get("stim_off_color") or list(DEFAULT_MS_STIM_OFF_COLOR)

    stimuli = load_stimuli(stimuli_path, supported_shapes=StimulusGenerator.supported_shapes())
    if not stimuli:
        raise RuntimeError(f"No stimuli found in {stimuli_path}")

    win = make_window(
        fullscr=(not args.windowed and DEFAULT_FULLSCREEN),
        screen=args.screen_id,
        background_color=background_color,
    )
    try:
        refresh_hz = measure_refresh_rate(win)
        generator = StimulusGenerator(
            win,
            monitor_refresh_rate=refresh_hz,
            luminance_depth=1.0,
            luminance_mod_kind="sine",
            background_color=background_color,
            stim_on_color=stim_on_color,
            stim_off_color=stim_off_color,
        )

        for idx, stim in enumerate(stimuli, start=1):
            stim_type = str(stim.get("type", "ssvep")).lower()
            configured_depth = (
                float(args.luminance_depth)
                if args.luminance_depth is not None and stim_type == "ssvep"
                else float(stim.get("luminance_depth", 1.0))
            )
            lum_kind, lum_depth = resolve_active_modulation(
                stim_type,
                str(args.luminance_mod_kind or stim.get("luminance_mod_kind", "sine")),
                configured_depth,
            )
            generator.luminance_mod_kind = lum_kind
            generator.luminance_depth = lum_depth
            generator.set_colors(
                stim_on_color=stim.get("stim_on_color") or stim_on_color,
                stim_off_color=stim.get("stim_off_color") or stim_off_color,
                background_color=stim.get("background_color") or background_color,
            )

            preview_cfg = dict(stim)
            preview_cfg["duration"] = float(args.duration)
            print(
                f"[{idx}/{len(stimuli)}] id={stim.get('id')} "
                f"shape={stim.get('shape')} type={stim_type} mod={lum_kind} depth={lum_depth:g}"
            )
            generator.run_stimulus(preview_cfg)
            generator.run_isi(float(args.isi))
    finally:
        win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

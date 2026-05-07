from __future__ import annotations

"""
Command line entry point for the frequency pretest.

This file only prepares runtime options and calls the experiment runner. It
does not contain marker, recording, stimulus, or questionnaire logic.
"""

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import (  # noqa: E402
    DEFAULT_ACQUISITION_MODE,
    DEFAULT_BASELINE_OPEN_S,
    DEFAULT_FIX_RANGE_S,
    DEFAULT_FREQUENCIES_HZ,
    DEFAULT_HEADSET_ID,
    DEFAULT_ITI_RANGE_S,
    DEFAULT_LIVE_FLUSH_EVERY,
    DEFAULT_LIVE_STREAMS,
    DEFAULT_OUT_DIR,
    DEFAULT_PROBAND_ID,
    DEFAULT_SCREEN_ID,
    DEFAULT_STIM_DURATION_S,
    DEFAULT_TRIALS_PER_FREQ,
)


def _csv_floats(value: str) -> list[float]:
    """Parse comma-separated floats."""

    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _csv_strings(value: str) -> list[str]:
    """Parse comma-separated strings."""

    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(
        description="Run the SSVEP frequency pretest.",
    )
    parser.add_argument("--proband", default=DEFAULT_PROBAND_ID, help="Participant ID, for example P01.")
    parser.add_argument(
        "--mode",
        choices=("record", "stream"),
        default=DEFAULT_ACQUISITION_MODE,
        help="Acquisition mode.",
    )
    parser.add_argument("--headset-id", default=DEFAULT_HEADSET_ID, help="Cortex headset ID.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output data directory.")
    parser.add_argument("--folder-name", default=None, help="Optional proband phase folder name.")
    parser.add_argument("--screen-id", type=int, default=DEFAULT_SCREEN_ID, help="PsychoPy screen index.")
    parser.add_argument("--windowed", action="store_true", help="Use a window instead of fullscreen.")
    parser.add_argument(
        "--frequencies",
        type=_csv_floats,
        default=list(DEFAULT_FREQUENCIES_HZ),
        help="Comma-separated frequencies in Hz, for example 8,10,12,15.",
    )
    parser.add_argument(
        "--trials-per-freq",
        type=int,
        default=None,
        help="Trials per frequency. If omitted, .env is used; if .env is missing, 5 is used.",
    )
    parser.add_argument("--stim-duration-s", type=float, default=DEFAULT_STIM_DURATION_S, help="Stimulus duration in seconds.")
    parser.add_argument("--baseline-open-s", type=float, default=DEFAULT_BASELINE_OPEN_S, help="Eyes-open baseline duration.")
    parser.add_argument("--fix-min-s", type=float, default=DEFAULT_FIX_RANGE_S[0], help="Minimum fixation duration.")
    parser.add_argument("--fix-max-s", type=float, default=DEFAULT_FIX_RANGE_S[1], help="Maximum fixation duration.")
    parser.add_argument("--iti-min-s", type=float, default=DEFAULT_ITI_RANGE_S[0], help="Minimum inter-trial interval.")
    parser.add_argument("--iti-max-s", type=float, default=DEFAULT_ITI_RANGE_S[1], help="Maximum inter-trial interval.")
    parser.add_argument(
        "--luminance-mod-kind",
        choices=("square", "sine", "triangle", "sawtooth"),
        default="square",
        help="Luminance modulation waveform.",
    )
    parser.add_argument("--luminance-depth", type=float, default=1.0, help="Luminance modulation depth from 0.0 to 1.0.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Default uses current time.")
    parser.add_argument(
        "--live-streams",
        type=_csv_strings,
        default=list(DEFAULT_LIVE_STREAMS),
        help="Comma-separated Cortex streams for stream mode.",
    )
    parser.add_argument("--live-flush-every", type=int, default=DEFAULT_LIVE_FLUSH_EVERY, help="Rows per live-stream CSV flush.")
    return parser


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed arguments into a runner config dict."""

    cfg = {
        "proband_id": args.proband,
        "acquisition_mode": args.mode,
        "headset_id": args.headset_id,
        "out_dir": args.out_dir,
        "folder_name": args.folder_name,
        "screen_id": args.screen_id,
        "fullscreen": not args.windowed,
        "frequencies": args.frequencies,
        "stim_duration_s": args.stim_duration_s,
        "baseline_open_s": args.baseline_open_s,
        "fix_range_s": (args.fix_min_s, args.fix_max_s),
        "iti_range_s": (args.iti_min_s, args.iti_max_s),
        "luminance_mod_kind": args.luminance_mod_kind,
        "luminance_depth": args.luminance_depth,
        "seed": args.seed,
        "live_streams": args.live_streams,
        "live_flush_every": args.live_flush_every,
    }
    if args.trials_per_freq is not None:
        cfg["trials_per_freq"] = args.trials_per_freq
    return cfg


def main(argv: list[str] | None = None) -> int:
    """Run the frequency pretest command."""

    parser = build_parser()
    args = parser.parse_args(argv)

    from experiments.frequency_pretest import run_frequency_pretest

    result = run_frequency_pretest(build_config(args))
    if isinstance(result, dict):
        print(f"Status: {result.get('status')}")
        print(f"Participant: {result.get('proband_id')}")
        print(f"Session: {result.get('session_id')}")
        print(f"Output: {result.get('output_folder')}")
    else:
        print("Frequency pretest finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

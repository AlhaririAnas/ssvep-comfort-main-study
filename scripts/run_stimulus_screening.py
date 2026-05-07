from __future__ import annotations

"""
Command line entry point for stimulus screening.

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
    DEFAULT_BASELINE_CLOSED_S,
    DEFAULT_BASELINE_OPEN_S,
    DEFAULT_BLOCK_BREAK_S,
    DEFAULT_FIX_RANGE_S,
    DEFAULT_HEADSET_ID,
    DEFAULT_INCLUDE_EYES_CLOSED_BASELINE,
    DEFAULT_ITI_RANGE_S,
    DEFAULT_LIVE_FLUSH_EVERY,
    DEFAULT_LIVE_STREAMS,
    DEFAULT_LONG_BREAK_AFTER_BLOCKS,
    DEFAULT_LONG_BREAK_S,
    DEFAULT_LUM_DEPTH,
    DEFAULT_LUM_MOD_KIND,
    DEFAULT_MS_IMPEDANCE_QUALITY_THRESHOLD,
    DEFAULT_MS_IMPEDANCE_STABLE_S_INITIAL,
    DEFAULT_MS_IMPEDANCE_TIMEOUT_S,
    DEFAULT_OUT_DIR,
    DEFAULT_PROBAND_ID,
    DEFAULT_SCREEN_ID,
    DEFAULT_STIMULI_JSON,
    DEFAULT_TRIALS_PER_STIM,
)


def _csv_ints(value: str) -> list[int]:
    """Parse comma-separated integers."""

    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _csv_strings(value: str) -> list[str]:
    """Parse comma-separated strings."""

    return [part.strip() for part in value.split(",") if part.strip()]


def _bool_flag(value: str) -> bool:
    """Parse common boolean text values."""

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(
        description="Run the SSVEP stimulus screening phase.",
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
    parser.add_argument("--stimuli-json", default=str(DEFAULT_STIMULI_JSON), help="Stimulus JSON path.")
    parser.add_argument("--pilot-ids", type=_csv_ints, default=None, help="Optional comma-separated stimulus IDs.")
    parser.add_argument(
        "--trials-per-stim",
        type=int,
        default=None,
        help="Trials per stimulus. If omitted, .env is used; if .env is missing, 5 is used.",
    )
    parser.add_argument("--baseline-open-s", type=float, default=DEFAULT_BASELINE_OPEN_S, help="Eyes-open baseline duration.")
    parser.add_argument("--baseline-closed-s", type=float, default=DEFAULT_BASELINE_CLOSED_S, help="Eyes-closed baseline duration.")
    parser.add_argument(
        "--include-eyes-closed-baseline",
        type=_bool_flag,
        default=DEFAULT_INCLUDE_EYES_CLOSED_BASELINE,
        help="Whether to include eyes-closed baseline.",
    )
    parser.add_argument("--fix-min-s", type=float, default=DEFAULT_FIX_RANGE_S[0], help="Minimum fixation duration.")
    parser.add_argument("--fix-max-s", type=float, default=DEFAULT_FIX_RANGE_S[1], help="Maximum fixation duration.")
    parser.add_argument("--iti-min-s", type=float, default=DEFAULT_ITI_RANGE_S[0], help="Minimum inter-trial interval.")
    parser.add_argument("--iti-max-s", type=float, default=DEFAULT_ITI_RANGE_S[1], help="Maximum inter-trial interval.")
    parser.add_argument(
        "--luminance-mod-kind",
        choices=("square", "sine", "triangle", "sawtooth"),
        default=DEFAULT_LUM_MOD_KIND,
        help="Default SSVEP luminance modulation waveform.",
    )
    parser.add_argument("--luminance-depth", type=float, default=DEFAULT_LUM_DEPTH, help="Default SSVEP luminance depth.")
    parser.add_argument("--block-break-s", type=float, default=DEFAULT_BLOCK_BREAK_S, help="Short break duration.")
    parser.add_argument(
        "--long-break-after-blocks",
        type=_csv_ints,
        default=list(DEFAULT_LONG_BREAK_AFTER_BLOCKS),
        help="Comma-separated block numbers after which a long break is used.",
    )
    parser.add_argument("--long-break-s", type=float, default=DEFAULT_LONG_BREAK_S, help="Long break duration.")
    parser.add_argument("--impedance-stable-s", type=float, default=DEFAULT_MS_IMPEDANCE_STABLE_S_INITIAL, help="Initial impedance stable duration.")
    parser.add_argument("--impedance-quality-threshold", type=int, default=DEFAULT_MS_IMPEDANCE_QUALITY_THRESHOLD, help="Minimum contact quality.")
    parser.add_argument("--impedance-timeout-s", type=float, default=DEFAULT_MS_IMPEDANCE_TIMEOUT_S, help="Impedance gate timeout.")
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
        "stimuli_json": args.stimuli_json,
        "pilot_ids": args.pilot_ids,
        "baseline_open_s": args.baseline_open_s,
        "baseline_closed_s": args.baseline_closed_s,
        "include_eyes_closed_baseline": args.include_eyes_closed_baseline,
        "fix_range_s": (args.fix_min_s, args.fix_max_s),
        "iti_range_s": (args.iti_min_s, args.iti_max_s),
        "luminance_mod_kind": args.luminance_mod_kind,
        "luminance_depth": args.luminance_depth,
        "block_break_s": args.block_break_s,
        "long_break_after_blocks": args.long_break_after_blocks,
        "long_break_s": args.long_break_s,
        "impedance_stable_s": args.impedance_stable_s,
        "impedance_quality_threshold": args.impedance_quality_threshold,
        "impedance_timeout_s": args.impedance_timeout_s,
        "seed": args.seed,
        "live_streams": args.live_streams,
        "live_flush_every": args.live_flush_every,
    }
    if args.trials_per_stim is not None:
        cfg["trials_per_stim"] = args.trials_per_stim
    return cfg


def main(argv: list[str] | None = None) -> int:
    """Run the stimulus screening command."""

    parser = build_parser()
    args = parser.parse_args(argv)

    from experiments.stimulus_screening import run_stimulus_screening

    result = run_stimulus_screening(build_config(args))
    if isinstance(result, dict):
        print(f"Status: {result.get('status')}")
        print(f"Participant: {result.get('proband_id')}")
        print(f"Session: {result.get('session_id')}")
        print(f"Output: {result.get('output_folder')}")
    else:
        print("Stimulus screening finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""
Command line entry point for the main study.

This file only prepares runtime options and calls the experiment runner. It
does not contain marker, recording, stimulus, or questionnaire logic.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import (  # noqa: E402
    DEFAULT_ACQUISITION_MODE,
    DEFAULT_HEADSET_ID,
    DEFAULT_LIVE_FLUSH_EVERY,
    DEFAULT_LIVE_STREAMS,
    DEFAULT_MS_BASELINE_CLOSED_S,
    DEFAULT_MS_BASELINE_OPEN_S,
    DEFAULT_MS_BLOCK_BREAK_S,
    DEFAULT_MS_CUE_DURATION_S,
    DEFAULT_MS_FIX_RANGE_S,
    DEFAULT_MS_FREQUENCIES_HZ,
    DEFAULT_MS_IDLE_ANGLES_DEG,
    DEFAULT_MS_IDLE_TRIALS_PER_ANGLE,
    DEFAULT_MS_IDLE_TRIALS_PER_SOURCE_BLOCK,
    DEFAULT_MS_IMPEDANCE_QUALITY_THRESHOLD,
    DEFAULT_MS_IMPEDANCE_STABLE_S_BETWEEN,
    DEFAULT_MS_IMPEDANCE_STABLE_S_INITIAL,
    DEFAULT_MS_IMPEDANCE_TIMEOUT_S,
    DEFAULT_MS_INCLUDE_EYES_CLOSED_BASELINE,
    DEFAULT_MS_ITI_RANGE_S,
    DEFAULT_MS_LONG_BREAK_AFTER_BLOCKS,
    DEFAULT_MS_LONG_BREAK_S,
    DEFAULT_MS_PROBAND_SCHEMA,
    DEFAULT_MS_SEED_BASE,
    DEFAULT_MS_SESSION_CONFIG,
    DEFAULT_MS_STIMULI_JSON,
    DEFAULT_MS_TARGET_SPACING_DEG,
    DEFAULT_MS_TARGET_SIZE_DEG,
    DEFAULT_MS_TRIAL_DURATION_S,
    DEFAULT_OUT_DIR,
    DEFAULT_PROBAND_ID,
    DEFAULT_SCREEN_ID,
)
from experiments.main_study import run_main_study_session  # noqa: E402


def _csv_floats(value: str) -> list[float]:
    """Parse comma-separated floats."""

    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _csv_ints(value: str) -> list[int]:
    """Parse comma-separated integers."""

    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _csv_strings(value: str) -> list[str]:
    """Parse comma-separated strings."""

    return [part.strip() for part in value.split(",") if part.strip()]


def _bool_flag(value: str) -> bool:
    """Parse common boolean text values."""

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    with target.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Session config must be a JSON object: {target}")
    return data


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(
        description="Run the main 4-target SSVEP study.",
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
    parser.add_argument("--session-config", default=str(DEFAULT_MS_SESSION_CONFIG), help="Main study session JSON path.")
    parser.add_argument("--stimuli-json", default=None, help="Override the stimuli JSON path from the session file.")
    parser.add_argument("--proband-schema", default=None, help="Override the proband schema path from the session file.")
    parser.add_argument(
        "--interface-language",
        choices=("en", "de", "English", "Deutsch"),
        default=None,
        help="Interface language for dialogs and on-screen instructions. If omitted, the session JSON is used.",
    )
    parser.add_argument(
        "--frequencies",
        type=_csv_floats,
        default=None,
        help="Override simultaneous target frequencies from the session file.",
    )
    parser.add_argument(
        "--trials-per-target",
        type=int,
        default=None,
        help="Trials per target in each block. If omitted, session JSON is used; if JSON is missing it, 5 is used.",
    )
    parser.add_argument("--trial-duration-s", type=float, default=None, help="Override trial stimulation duration.")
    parser.add_argument("--cue-duration-s", type=float, default=None, help="Override cue duration before each trial.")
    parser.add_argument("--target-spacing-deg", type=float, default=None, help="Override target distance from center.")
    parser.add_argument("--target-size-deg", type=float, default=None, help="Override target size in visual degrees.")
    parser.add_argument("--baseline-open-s", type=float, default=None, help="Override eyes-open baseline duration.")
    parser.add_argument("--baseline-closed-s", type=float, default=None, help="Override eyes-closed baseline duration.")
    parser.add_argument(
        "--include-eyes-closed-baseline",
        type=_bool_flag,
        default=None,
        help="Whether to include eyes-closed baseline.",
    )
    parser.add_argument("--fix-min-s", type=float, default=None, help="Override minimum fixation duration.")
    parser.add_argument("--fix-max-s", type=float, default=None, help="Override maximum fixation duration.")
    parser.add_argument("--iti-min-s", type=float, default=None, help="Override minimum inter-trial interval.")
    parser.add_argument("--iti-max-s", type=float, default=None, help="Override maximum inter-trial interval.")
    parser.add_argument("--block-break-s", type=float, default=None, help="Override short break duration.")
    parser.add_argument("--long-break-after-blocks", type=_csv_ints, default=None, help="Override long-break block numbers.")
    parser.add_argument("--long-break-s", type=float, default=None, help="Override long break duration.")
    parser.add_argument("--impedance-stable-s-initial", type=float, default=None, help="Override initial impedance stable duration.")
    parser.add_argument("--impedance-stable-s-between", type=float, default=None, help="Override between-block impedance stable duration.")
    parser.add_argument("--impedance-quality-threshold", type=int, default=None, help="Override minimum contact quality.")
    parser.add_argument("--impedance-timeout-s", type=float, default=None, help="Override impedance gate timeout.")
    parser.add_argument("--idle-angles-deg", type=_csv_floats, default=None, help="Override idle gaze angles.")
    parser.add_argument("--idle-trials-per-angle", type=int, default=None, help="Override idle trials per gaze angle.")
    parser.add_argument("--idle-trials-per-source-block", type=int, default=None, help="Override idle trials paired with each active block.")
    parser.add_argument("--seed-base", type=int, default=None, help="Override base seed for deterministic randomization.")
    parser.add_argument("--live-streams", type=_csv_strings, default=list(DEFAULT_LIVE_STREAMS), help="Comma-separated Cortex streams for stream mode.")
    parser.add_argument("--live-flush-every", type=int, default=DEFAULT_LIVE_FLUSH_EVERY, help="Rows per live-stream CSV flush.")
    parser.add_argument("--dry-run", action="store_true", help="Build protocol files without PsychoPy or Cortex.")
    parser.add_argument("--skip-dialogs", action="store_true", help="Use config defaults without GUI dialogs.")
    parser.add_argument("--skip-compliance", action="store_true", help="Skip the pre-flight checklist.")
    parser.add_argument("--headless", action="store_true", help="Use headless questionnaire defaults.")
    return parser


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed arguments into a runner config dict."""

    session_config = _load_json_object(args.session_config)
    optional_updates = {
        "stimuli_json": args.stimuli_json,
        "proband_schema": args.proband_schema,
        "frequencies_hz": args.frequencies,
        "trial_duration_s": args.trial_duration_s,
        "cue_duration_s": args.cue_duration_s,
        "target_spacing_deg": args.target_spacing_deg,
        "target_size_deg": args.target_size_deg,
        "baseline_open_s": args.baseline_open_s,
        "baseline_closed_s": args.baseline_closed_s,
        "include_eyes_closed_baseline": args.include_eyes_closed_baseline,
        "block_break_s": args.block_break_s,
        "long_break_after_blocks": args.long_break_after_blocks,
        "long_break_s": args.long_break_s,
        "impedance_stable_s_initial": args.impedance_stable_s_initial,
        "impedance_stable_s_between": args.impedance_stable_s_between,
        "impedance_quality_threshold": args.impedance_quality_threshold,
        "impedance_timeout_s": args.impedance_timeout_s,
        "idle_angles_deg": args.idle_angles_deg,
        "idle_trials_per_angle": args.idle_trials_per_angle,
        "idle_trials_per_source_block": args.idle_trials_per_source_block,
        "seed_base": args.seed_base,
    }
    for key, value in optional_updates.items():
        if value is not None:
            session_config[key] = value
    if args.fix_min_s is not None or args.fix_max_s is not None:
        current = list(session_config.get("fix_range_s", DEFAULT_MS_FIX_RANGE_S))
        session_config["fix_range_s"] = [
            args.fix_min_s if args.fix_min_s is not None else current[0],
            args.fix_max_s if args.fix_max_s is not None else current[1],
        ]
    if args.iti_min_s is not None or args.iti_max_s is not None:
        current = list(session_config.get("iti_range_s", DEFAULT_MS_ITI_RANGE_S))
        session_config["iti_range_s"] = [
            args.iti_min_s if args.iti_min_s is not None else current[0],
            args.iti_max_s if args.iti_max_s is not None else current[1],
        ]
    if args.trials_per_target is not None:
        session_config["trials_per_target"] = args.trials_per_target
    if args.interface_language is not None:
        session_config["interface_language"] = args.interface_language

    return {
        "proband_id": args.proband,
        "acquisition_mode": args.mode,
        "headset_id": args.headset_id,
        "out_dir": args.out_dir,
        "folder_name": args.folder_name,
        "screen_id": args.screen_id,
        "fullscreen": not args.windowed,
        "session_config_data": session_config,
        "session_config_label": args.session_config,
        "stimuli_json": args.stimuli_json,
        "proband_schema": args.proband_schema,
        "live_streams": args.live_streams,
        "live_flush_every": args.live_flush_every,
        "dry_run": args.dry_run,
        "skip_dialogs": args.skip_dialogs,
        "skip_compliance": args.skip_compliance,
        "headless": args.headless,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the main study command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    protocol = run_main_study_session(build_config(args))
    print(f"Status: {protocol.get('status')}")
    print(f"Participant: {protocol.get('proband_id')}")
    print(f"Session: {protocol.get('session_id')}")
    print(f"Output: {protocol.get('output_folder')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

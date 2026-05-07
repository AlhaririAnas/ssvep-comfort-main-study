from __future__ import annotations

"""Local visual test for the main-study idle block."""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.conditions import (  # noqa: E402
    idle_trial_marker_label,
    idle_trial_start_marker,
    idle_trial_stop_marker,
    resolve_active_modulation,
)
from core.config import (  # noqa: E402
    DEFAULT_ACQUISITION_MODE,
    DEFAULT_LIVE_FLUSH_EVERY,
    DEFAULT_LIVE_STREAMS,
    DEFAULT_MS_SESSION_CONFIG,
    DEFAULT_MS_STIMULI_JSON,
    DEFAULT_MS_TARGET_SIZE_DEG,
    DEFAULT_MS_TRIAL_DURATION_S,
    DEFAULT_PROBAND_ID,
    DEFAULT_SCREEN_ID,
)
from experiments.main_study import _contrast_percent, _load_idle_target_positions  # noqa: E402
from stimuli.display import make_info_stim, measure_refresh_rate, show_message  # noqa: E402
from stimuli.generator import StimulusGenerator, make_window, normalize_stimulus_cfg  # noqa: E402
from acquisition.live_stream import CortexLiveStream  # noqa: E402
from acquisition.marker import Marker  # noqa: E402
from acquisition.recording import CortexRecording  # noqa: E402


def _load_json(path: str | Path) -> Any:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    with target.open("r", encoding="utf-8") as file:
        return json.load(file)


def _csv_values(text: str) -> set[str]:
    return {part.strip().upper() for part in text.split(",") if part.strip()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run only the B8 idle visual and marker schedule.")
    parser.add_argument("--proband", default=DEFAULT_PROBAND_ID, help="Participant ID.")
    parser.add_argument(
        "--mode",
        choices=("local", "stream", "record"),
        default="local",
        help="local = no Cortex, stream = local EEG CSV, record = Cortex record/export.",
    )
    parser.add_argument("--headset-id", default="", help="Optional Cortex headset ID.")
    parser.add_argument("--session-config", default=str(DEFAULT_MS_SESSION_CONFIG), help="Main-study session JSON.")
    parser.add_argument("--stimuli-json", default=str(DEFAULT_MS_STIMULI_JSON), help="Main-study stimuli JSON.")
    parser.add_argument("--source-blocks", default="all", help="Comma list like B1,B2 or 'all'.")
    parser.add_argument("--idle-trials-per-source-block", type=int, default=None, help="Idle trials per source block.")
    parser.add_argument("--trial-duration-s", type=float, default=None, help="Single idle-trial duration.")
    parser.add_argument("--post-stop-hold-s", type=float, default=None, help="Stimulus hold after the final stop marker.")
    parser.add_argument("--screen-id", type=int, default=DEFAULT_SCREEN_ID, help="PsychoPy screen index.")
    parser.add_argument("--windowed", action="store_true", help="Use a window instead of fullscreen.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "idle_test"), help="Output folder.")
    parser.add_argument(
        "--live-streams",
        default=",".join(DEFAULT_LIVE_STREAMS),
        help="Comma-separated streams for --mode stream. Use eeg,dev for a light EEG test.",
    )
    parser.add_argument("--live-flush-every", type=int, default=DEFAULT_LIVE_FLUSH_EVERY, help="Rows per CSV flush.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_cfg = _load_json(args.session_config)
    stimuli_list = _load_json(args.stimuli_json)
    stimuli_by_id = {int(stim["id"]): stim for stim in stimuli_list if "id" in stim}

    active_blocks = [
        (idx, block)
        for idx, block in enumerate(session_cfg["block_order"], start=1)
        if block.get("stimulus_id") is not None
    ]
    if args.source_blocks.strip().lower() != "all":
        wanted = _csv_values(args.source_blocks)
        active_blocks = [
            (idx, block)
            for idx, block in active_blocks
            if str(block.get("block_id", f"B{idx}")).upper() in wanted
            or str(block.get("dataset_id", f"DS{idx}")).upper() in wanted
        ]
    if not active_blocks:
        raise RuntimeError("No source block selected for idle test.")

    frequencies_hz = [float(freq) for freq in session_cfg["frequencies_hz"]]
    trial_duration_s = float(args.trial_duration_s or session_cfg.get("trial_duration_s", DEFAULT_MS_TRIAL_DURATION_S))
    idle_trials_per_source = int(
        args.idle_trials_per_source_block
        or session_cfg.get("idle_trials_per_source_block", 6)
    )
    post_stop_hold_s = float(
        args.post_stop_hold_s
        if args.post_stop_hold_s is not None
        else session_cfg.get("idle_post_stop_hold_s", 0.1)
    )
    target_size_deg = float(session_cfg.get("target_size_deg", DEFAULT_MS_TARGET_SIZE_DEG))
    positions_deg = _load_idle_target_positions(
        session_cfg,
        spacing_deg=float(session_cfg.get("target_spacing_deg", 9.0)),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    run_dir = out_dir / f"{args.proband}_idle_test_{timestamp}"
    marker_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    win = None
    marker: Marker | None = None
    live: CortexLiveStream | None = None
    recording: CortexRecording | None = None
    export_ok = False
    try:
        if args.mode != "local":
            run_dir.mkdir(parents=True, exist_ok=True)
            marker = Marker(debug_mode=False)
            marker.connect(headset_id=args.headset_id)
            if not marker.wait_for_session(timeout=90.0):
                raise RuntimeError("Cortex session could not be established.")
            if args.mode == "record":
                recording = CortexRecording(marker)
                record_title = f"{args.proband}_idle_test_{timestamp}"
                if not recording.start_record(record_title, description="BCI idle block test"):
                    raise RuntimeError("Cortex recording could not be started.")
                print(f"Cortex record started: {record_title}")
            else:
                streams = [part.strip().lower() for part in args.live_streams.split(",") if part.strip()]
                live = CortexLiveStream(
                    marker=marker,
                    output_dir=run_dir / "live_stream",
                    streams=streams,
                    flush_every=args.live_flush_every,
                )
                live.start(wait_labels_s=10.0)
                print(f"Live EEG stream started: {run_dir / 'live_stream'}")

        win = make_window(
            fullscr=not args.windowed,
            screen=args.screen_id,
            background_color=session_cfg.get("background_color", [-0.8, -0.8, -0.8]),
        )
        info_stim = make_info_stim(win)
        refresh_hz = measure_refresh_rate(win)
        generator = StimulusGenerator(
            win,
            monitor_refresh_rate=refresh_hz,
            luminance_depth=1.0,
            luminance_mod_kind="sine",
            background_color=session_cfg.get("background_color", [-0.8, -0.8, -0.8]),
            stim_on_color=session_cfg.get("stim_on_color", [0.5, 1.0, 0.5]),
            stim_off_color=session_cfg.get("stim_off_color", [-0.8, -0.8, -0.8]),
            abort_check=lambda: marker is not None and not marker.is_headset_connected(),
        )

        phase_duration_s = idle_trials_per_source * trial_duration_s + post_stop_hold_s
        key = show_message(
            win,
            info_stim,
            "Idle test only\n\n"
            f"Source blocks: {', '.join(str(block.get('block_id', f'B{idx}')) for idx, block in active_blocks)}\n"
            f"Each source phase: {idle_trials_per_source} x {trial_duration_s:.1f} s + {post_stop_hold_s:.1f} s hold "
            f"= {phase_duration_s:.1f} s\n\n"
            "Press SPACE to start.",
        )
        if key == "escape":
            return 1

        for source_block_idx, source_block in active_blocks:
            source_block_id = str(source_block.get("block_id", f"B{source_block_idx}"))
            dataset_id = str(source_block.get("dataset_id", f"DS{source_block_idx}"))
            source_stim_id = int(source_block["stimulus_id"])
            source_stim = normalize_stimulus_cfg(stimuli_by_id[source_stim_id])
            source_type = str(source_stim.get("type", "ssvep")).lower()
            source_lum_kind, source_lum_depth = resolve_active_modulation(
                source_type,
                str(source_stim.get("luminance_mod_kind", "sine")),
                float(source_stim.get("luminance_depth", 1.0)),
            )
            generator.luminance_mod_kind = source_lum_kind
            generator.luminance_depth = source_lum_depth
            generator.set_colors(
                stim_on_color=source_stim.get("stim_on_color") or session_cfg.get("stim_on_color", [0.5, 1.0, 0.5]),
                stim_off_color=source_stim.get("stim_off_color") or session_cfg.get("stim_off_color", [-0.8, -0.8, -0.8]),
                background_color=source_stim.get("background_color") or session_cfg.get("background_color", [-0.8, -0.8, -0.8]),
            )

            onsets: dict[int, tuple[float, float, int, str]] = {}
            offsets: dict[int, tuple[float, float, int, str]] = {}
            phase_start_mono: list[float | None] = [None]

            def _record_marker(trial_idx: int, *, role: str) -> None:
                mono = time.monotonic()
                if phase_start_mono[0] is None:
                    phase_start_mono[0] = mono
                if role == "START":
                    marker_value = idle_trial_start_marker(
                        source_block_idx,
                        trial_idx,
                        trials_per_source=idle_trials_per_source,
                    )
                else:
                    marker_value = idle_trial_stop_marker(
                        source_block_idx,
                        trial_idx,
                        trials_per_source=idle_trials_per_source,
                    )
                label = idle_trial_marker_label(source_block_id, trial_idx, role=role)
                marker_time_ms = marker.capture_flip_time_ms(mono) if marker is not None else mono * 1000.0
                marker_role = f"idle_trial_{role.lower()}"
                if marker is not None and args.mode == "record":
                    marker.inject_marker(
                        value=str(marker_value),
                        label=label,
                        timestamp_ms=marker_time_ms,
                        capture_method="flip_callback",
                    )
                if live is not None:
                    live.register_marker(
                        marker_value=int(marker_value),
                        marker_label=label,
                        marker_time_ms=float(marker_time_ms),
                        marker_role=marker_role,
                        metadata={
                            "proband_id": args.proband,
                            "source_block_idx": source_block_idx,
                            "source_block_id": source_block_id,
                            "dataset_id": dataset_id,
                            "idle_trial_idx": int(trial_idx),
                        },
                    )
                row = {
                    "source_block_idx": source_block_idx,
                    "source_block_id": source_block_id,
                    "dataset_id": dataset_id,
                    "trial_idx": trial_idx,
                    "marker_role": marker_role,
                    "marker_value": marker_value,
                    "marker_label": label,
                    "t_monotonic": round(mono, 6),
                    "t_relative_s": round(mono - float(phase_start_mono[0]), 6),
                    "marker_time_ms": round(marker_time_ms, 3),
                }
                marker_rows.append(row)
                if role == "START":
                    onsets[trial_idx] = (mono, marker_time_ms, marker_value, label)
                else:
                    offsets[trial_idx] = (mono, marker_time_ms, marker_value, label)

            def _trial_start(trial_idx: int, _flip_time: float | None = None) -> None:
                _record_marker(trial_idx, role="START")

            def _trial_stop(trial_idx: int, _flip_time: float | None = None) -> None:
                _record_marker(trial_idx, role="STOP")

            result = generator.run_multi_target_continuous_trials(
                base_cfg={**source_stim, "size_deg": source_stim.get("size_deg", target_size_deg)},
                frequencies_hz=frequencies_hz,
                positions_deg=positions_deg,
                trial_duration_s=trial_duration_s,
                trial_count=idle_trials_per_source,
                on_trial_start=_trial_start,
                on_trial_stop=_trial_stop,
                post_stop_hold_s=post_stop_hold_s,
            )
            if result.get("aborted"):
                break

            for trial_idx in range(1, idle_trials_per_source + 1):
                if trial_idx not in onsets or trial_idx not in offsets:
                    continue
                onset_mono, onset_ms, start_marker, start_label = onsets[trial_idx]
                offset_mono, offset_ms, stop_marker, stop_label = offsets[trial_idx]
                trial_rows.append({
                    "source_block_idx": source_block_idx,
                    "source_block_id": source_block_id,
                    "dataset_id": dataset_id,
                    "trial_idx": trial_idx,
                    "stimulus_id": source_stim_id,
                    "stimulus_name": source_stim.get("name", ""),
                    "stimulus_shape": source_stim.get("shape", ""),
                    "luminance_mod_kind": source_lum_kind,
                    "luminance_depth": source_lum_depth,
                    "contrast_percent": _contrast_percent(source_stim, session_cfg),
                    "nominal_duration_s": round(trial_duration_s, 6),
                    "actual_duration_s": round(offset_mono - onset_mono, 6),
                    "start_marker": start_marker,
                    "start_marker_label": start_label,
                    "stop_marker": stop_marker,
                    "stop_marker_label": stop_label,
                    "t_onset_marker_ms": round(onset_ms, 3),
                    "t_offset_marker_ms": round(offset_ms, 3),
                })

        _write_csv(run_dir / "idle_test_markers.csv", marker_rows)
        _write_csv(run_dir / "idle_test_trials.csv", trial_rows)
        print(f"Refresh rate used: {refresh_hz:.3f} Hz")
        print(f"Output: {run_dir}")
        print(f"Marker rows: {len(marker_rows)}")
        print(f"Trial rows: {len(trial_rows)}")
    finally:
        if win is not None:
            win.close()
        if args.mode == "record" and recording is not None:
            export_ok = recording.stop_and_export(
                export_folder=run_dir,
                data_types=["EEG", "MOTION", "PM", "BP"],
                export_format="CSV",
                version="V2",
                timeout=300.0,
            )
            print(f"Cortex export ok: {export_ok}")
        if live is not None:
            live.stop()
            print(f"Live stream summary: {live.summary()}")
        if marker is not None:
            marker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

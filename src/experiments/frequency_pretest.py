from __future__ import annotations

"""
frequency_pretest.py
====================
Phase 1 of the BCI pipeline: frequency pretest for SSVEP paradigms.

Supports two acquisition modes
------------------------------
- "record": Cortex record + export at the end.
- "stream": live Cortex stream writing via live_stream.py.

In both modes, protocol / trials / events are saved continuously.
In stream mode, marker information is embedded directly into eeg.csv.
In record mode, marker information is embedded by Cortex in the exported CSV.
"""

import time
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __name__ == "__main__":  # pragma: no cover - supports direct module demos
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import (
    DEFAULT_PROBAND_ID,
    DEFAULT_OUT_DIR,
    DEFAULT_SCREEN_ID,
    DEFAULT_FULLSCREEN,
    DEFAULT_HEADSET_ID,
    DEFAULT_FREQUENCIES_HZ,
    DEFAULT_TRIALS_PER_FREQ,
    DEFAULT_STIM_DURATION_S,
    DEFAULT_BASELINE_OPEN_S,
    DEFAULT_FIX_RANGE_S,
    DEFAULT_ITI_RANGE_S,
    DEFAULT_LIVE_STREAMS,
    DEFAULT_LIVE_FLUSH_EVERY,
    DEFAULT_ACQUISITION_MODE,
    MONITOR_REFRESH_RATE,
)
from core.session import (
    build_run_timestamp,
    determine_next_session_id,
    now_iso,
    build_output_paths,
    build_cortex_record_title,
)
from core.logging_utils import setup_logging
from data_io.writers import (
    safe_write_json,
    append_csv_rows,
)
from acquisition.live_stream import CortexLiveStream
from acquisition.impedance import ImpedanceMonitor, run_impedance_gate
from acquisition.marker import Marker
from acquisition.recording import CortexRecording
from acquisition.recovery import TrialConnectionLost, raise_if_disconnected, recover_after_disconnect
from core.conditions import (
    CONDITION_SCHEMA_VERSION,
    assign_condition_ids,
    attach_condition_fields,
    build_frequency_pretest_condition_spec,
    condition_marker_label,
)
__all__ = ["run_frequency_pretest"]

# ---------------------------------------------------------------------------
# Marker codes
# ---------------------------------------------------------------------------
# Baseline events
MARKER_BASELINE_OPEN_START = 10   # eyes-open baseline onset
MARKER_BASELINE_OPEN_STOP  = 11   # eyes-open baseline offset

# ---------------------------------------------------------------------------
# Phase identifier
# ---------------------------------------------------------------------------
PHASE_SHORT = "freqtest"
PHASE_LABEL = "Frequency pretest"

# ---------------------------------------------------------------------------
# Output fields
# ---------------------------------------------------------------------------
TRIAL_FIELDS = [
    "session_id", "proband_id", "phase",
    "trial_idx_global", "block_idx", "trial_idx_in_block",
    "condition_id", "condition_numeric_id", "condition_label", "condition_key",
    "frequency_hz",
    "stimulus_type", "stimulus_name", "stimulus_shape",
    "luminance_mod_kind", "luminance_depth",
    "stim_type", "stim_shape", "lum_mod_kind", "lum_depth",
    "nominal_duration_s", "fix_duration_s", "iti_duration_s",
    "start_marker", "stop_marker",
    "t_onset_marker_ms", "t_offset_marker_ms",
    "t_onset_monotonic", "t_offset_monotonic",
    "actual_duration_s",
]

EVENT_FIELDS = [
    "event_type", "session_id", "proband_id", "phase",
    "baseline_type", "block_idx", "trial_idx_global",
    "condition_id", "condition_numeric_id", "condition_label", "condition_key",
    "frequency_hz",
    "stimulus_type", "stimulus_name", "stimulus_shape",
    "luminance_mod_kind", "luminance_depth",
    "marker_value", "marker_label",
    "t_marker_ms", "t_monotonic",
    "t_block_monotonic", "actual_duration_s",
    "timestamp_iso",
]


def _estimate_session_duration_min(
    *,
    baseline_open_s: float,
    total_trials: int,
    stim_duration_s: float,
    fix_range_s: tuple[float, float],
    iti_range_s: tuple[float, float],
    block_count: int,
    block_intro_s: float = 5.0,
    safety_buffer_min: float = 5.0,
) -> float:
    avg_fix_s = sum(float(v) for v in fix_range_s) / 2.0
    avg_iti_s = sum(float(v) for v in iti_range_s) / 2.0
    total_seconds = (
        float(baseline_open_s)
        + total_trials * (float(stim_duration_s) + avg_fix_s + avg_iti_s)
        + max(0, int(block_count)) * float(block_intro_s)
        + float(safety_buffer_min) * 60.0
    )
    return total_seconds / 60.0


def _collect_trial_count(default_value: int) -> int:
    """Show an editable runtime field for the trial count."""

    from psychopy import gui  # type: ignore

    dlg = gui.Dlg(title=f"{PHASE_LABEL} - runtime settings")
    dlg.addText("Review and edit the session settings.")
    dlg.addField("Trials per frequency", initial=int(default_value))
    data = dlg.show()
    if not dlg.OK:
        raise RuntimeError("Runtime settings dialog cancelled by operator.")
    raw_value = data[0] if isinstance(data, list) else list(data.values())[0]
    value = int(raw_value)
    if value < 1:
        raise ValueError("Trials per frequency must be >= 1.")
    return value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_frequencies(values: Iterable[Any]) -> List[float]:
    freqs = [float(v) for v in values]
    if not freqs:
        raise ValueError("At least one frequency must be provided.")
    if any(f <= 0 for f in freqs):
        raise ValueError("All frequencies must be > 0 Hz.")
    return freqs


def _build_condition_map(
    frequencies_hz: List[float],
    *,
    luminance_mod_kind: str,
    luminance_depth: float,
) -> Dict[float, Dict[str, Any]]:
    specs = [
        build_frequency_pretest_condition_spec(
            frequency_hz=freq,
            luminance_mod_kind=luminance_mod_kind,
            luminance_depth=luminance_depth,
        )
        for freq in frequencies_hz
    ]
    catalog = assign_condition_ids(specs)
    return {
        float(cond["frequency_hz"]): cond
        for cond in catalog["conditions"]
    }


def _flush_pending_events(
    pending_rows: List[Dict[str, Any]],
    event_rows: List[Dict[str, Any]],
    events_path: str,
    logger,
) -> None:
    if not pending_rows:
        return
    rows = list(pending_rows)
    pending_rows.clear()
    event_rows.extend(rows)
    append_csv_rows(events_path, EVENT_FIELDS, rows, logger=logger)
    logger.info("Event rows appended | count=%d", len(rows))


def _schedule_marker(
    win: visual.Window,
    marker: Marker,
    *,
    acquisition_mode: str,
    live: Optional[CortexLiveStream],
    value: int,
    label: str,
    session_id: str,
    proband_id: str,
    block_idx: Optional[int],
    trial_idx_global: Optional[int],
    condition: Optional[Dict[str, Any]],
    pending_event_rows: List[Dict[str, Any]],
    flip_time_out: Optional[List[float]] = None,
    marker_time_ms_out: Optional[List[float]] = None,
    logger=None,
) -> None:
    def _callback() -> None:
        mono = time.monotonic()
        ts_ms = marker.capture_flip_time_ms(mono)

        if flip_time_out is not None:
            flip_time_out[0] = mono
        if marker_time_ms_out is not None:
            marker_time_ms_out[0] = ts_ms

        if acquisition_mode == "record":
            marker.inject_marker(
                value=str(value),
                label=label,
                timestamp_ms=ts_ms,
                capture_method="flip_callback",
            )

        if live is not None:
            metadata = {
                "session_id": session_id,
                "proband_id": proband_id,
                "block_idx": block_idx,
                "trial_idx_global": trial_idx_global,
            }
            if condition is not None:
                metadata.update({
                    "condition_id": condition["condition_id"],
                    "condition_numeric_id": condition["condition_id"],
                    "condition_label": condition["condition_label"],
                    "condition_key": condition["condition_key"],
                    "frequency_hz": condition["frequency_hz"],
                    "stimulus_type": condition["stimulus_type"],
                    "stimulus_name": condition["stimulus_name"],
                    "stimulus_shape": condition["stimulus_shape"],
                    "luminance_mod_kind": condition["luminance_mod_kind"],
                    "luminance_depth": condition["luminance_depth"],
                })
            live.register_marker(
                marker_value=value,
                marker_label=label,
                marker_time_ms=ts_ms,
                marker_role="trial_marker",
                metadata=metadata,
            )

        event_row = {
            "event_type": "local_marker",
            "session_id": session_id,
            "proband_id": proband_id,
            "phase": PHASE_SHORT,
            "baseline_type": None,
            "block_idx": block_idx,
            "trial_idx_global": trial_idx_global,
            "marker_value": value,
            "marker_label": label,
            "t_marker_ms": round(ts_ms, 3),
            "t_monotonic": round(mono, 6),
            "t_block_monotonic": None,
            "actual_duration_s": None,
            "timestamp_iso": now_iso(),
        }
        if condition is not None:
            attach_condition_fields(event_row, condition)
        pending_event_rows.append(event_row)

        if logger is not None:
            logger.info("Marker captured | label=%s | value=%s | t_ms=%.3f", label, value, ts_ms)

    win.callOnFlip(_callback)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_frequency_pretest(cfg: Dict[str, Any]) -> None:
    from psychopy import core  # type: ignore
    from stimuli.display import (
        seconds_to_frames,
        estimate_realized_frequency,
        check_escape,
        draw_all,
        show_fixation,
        show_blank,
        show_message,
        make_info_stim,
    )
    from stimuli.generator import (
        StimulusGenerator,
        build_fixation_stimuli,
        make_window,
        measure_refresh_rate,
    )

    marker: Optional[Marker] = None
    recording: Optional[CortexRecording] = None
    live: Optional[CortexLiveStream] = None
    impedance_monitor: Optional[ImpedanceMonitor] = None
    win: Optional[visual.Window] = None
    aborted = False

    # --- Parameters ---
    proband_id = str(cfg.get("proband_id", DEFAULT_PROBAND_ID))
    folder_name = cfg.get("folder_name")
    frequencies_hz = _validate_frequencies(cfg.get("frequencies", DEFAULT_FREQUENCIES_HZ))
    trials_per_freq = int(cfg.get("trials_per_freq", DEFAULT_TRIALS_PER_FREQ))
    trials_per_freq = _collect_trial_count(trials_per_freq)
    stim_duration_s = float(cfg.get("stim_duration_s", DEFAULT_STIM_DURATION_S))
    baseline_open_s = float(cfg.get("baseline_open_s", DEFAULT_BASELINE_OPEN_S))
    fix_range_s = tuple(cfg.get("fix_range_s", DEFAULT_FIX_RANGE_S))
    iti_range_s = tuple(cfg.get("iti_range_s", DEFAULT_ITI_RANGE_S))
    screen_id = int(cfg.get("screen_id", DEFAULT_SCREEN_ID))
    fullscreen = bool(cfg.get("fullscreen", DEFAULT_FULLSCREEN))
    out_dir = str(cfg.get("out_dir", DEFAULT_OUT_DIR))
    headset_id = str(cfg.get("headset_id", DEFAULT_HEADSET_ID))
    seed = int(cfg.get("seed") or int(time.time() * 1000) % (2**32))
    luminance_depth = float(cfg.get("luminance_depth", 1.0))
    luminance_mod_kind = str(cfg.get("luminance_mod_kind", "square"))
    acquisition_mode = str(cfg.get("acquisition_mode", cfg.get("storage_mode", DEFAULT_ACQUISITION_MODE))).lower()
    live_streams = [str(s).lower() for s in cfg.get("live_streams", list(DEFAULT_LIVE_STREAMS))]
    live_flush_every = int(cfg.get("live_flush_every", DEFAULT_LIVE_FLUSH_EVERY))

    if acquisition_mode not in {"record", "stream"}:
        raise ValueError("acquisition_mode must be 'record' or 'stream'.")
    if trials_per_freq < 1:
        raise ValueError("trials_per_freq must be >= 1.")
    if stim_duration_s <= 0 or baseline_open_s < 0:
        raise ValueError("Durations must be positive.")

    # --- Session IDs and paths ---
    run_timestamp = build_run_timestamp()
    session_id, _ = determine_next_session_id(
        proband_id,
        PHASE_SHORT,
        out_dir,
        folder_name=folder_name,
    )
    paths = build_output_paths(
        proband_id=proband_id,
        session_id=session_id,
        phase_short=PHASE_SHORT,
        run_timestamp=run_timestamp,
        out_dir=out_dir,
        folder_name=folder_name,
    ).as_dict()
    record_title = build_cortex_record_title(
        proband_id,
        PHASE_SHORT,
        session_id,
        run_timestamp,
    )

    logger = setup_logging(paths["log"], f"bci.{PHASE_SHORT}")
    rng = random.Random(seed)

    condition_map = _build_condition_map(
        frequencies_hz,
        luminance_mod_kind=luminance_mod_kind,
        luminance_depth=luminance_depth,
    )
    frequency_order = list(frequencies_hz)
    rng.shuffle(frequency_order)
    total_trials = len(frequency_order) * trials_per_freq
    ordered_conditions = [
        condition_map[freq]
        for freq in sorted(condition_map, key=lambda value: condition_map[value]["condition_id"])
    ]

    protocol: Dict[str, Any] = {
        "study": "BCI_BA",
        "phase": PHASE_SHORT,
        "proband_id": proband_id,
        "session_id": session_id,
        "datetime_start": now_iso(),
        "random_seed": seed,
        "status": "running",
        "aborted": False,
        "completed_trials": 0,
        "total_trials": total_trials,
        "config": {
            "frequencies_hz": frequencies_hz,
            "trials_per_freq": trials_per_freq,
            "stim_duration_s": stim_duration_s,
            "stimulus_type": "ssvep",
            "stimulus_name": "SSVEP Frequency Pretest Flicker",
            "stimulus_shape": "flicker",
            "lum_mod_kind": luminance_mod_kind,
            "lum_depth": luminance_depth,
            "baseline_open_s": baseline_open_s,
            "fix_range_s": list(fix_range_s),
            "iti_range_s": list(iti_range_s),
            "screen_id": screen_id,
            "fullscreen": fullscreen,
            "headset_id": headset_id,
            "acquisition_mode": acquisition_mode,
            "live_streams": live_streams if acquisition_mode == "stream" else [],
            "live_flush_every": live_flush_every if acquisition_mode == "stream" else None,
            "expected_refresh_hz": round(MONITOR_REFRESH_RATE, 3),
            "monitor_refresh_hz": None,
            "freq_realization": None,
        },
        "condition_schema_version": CONDITION_SCHEMA_VERSION,
        "condition_sort_rule": [
            "stimulus_type",
            "frequency_hz",
            "stimulus_name",
            "stimulus_shape",
            "luminance_mod_kind",
            "luminance_depth",
        ],
        "condition_map": {str(cond["condition_id"]): cond for cond in ordered_conditions},
        "condition_list": ordered_conditions,
        "block_order": [condition_map[f]["condition_id"] for f in frequency_order],
        "baseline_log": [],
        "block_log": [],
        "trial_log": [],
        "event_log": [],
        "marker_timing_log": [],
        "export_ok": False,
        "live_summary": {},
        "output_folder": paths["folder"],
        "datetime_end": None,
    }

    trial_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    pending_event_rows: List[Dict[str, Any]] = []

    def snapshot_protocol() -> None:
        protocol["event_log"] = event_rows
        if marker is not None:
            protocol["marker_timing_log"] = marker.stats.log
        safe_write_json(protocol, paths["protocol"], logger=logger)
        logger.info("Protocol snapshot saved")

    def append_trial(row: Dict[str, Any]) -> None:
        trial_rows.append(row)
        protocol["trial_log"].append(row)
        protocol["completed_trials"] = len(trial_rows)
        append_csv_rows(paths["trials"], TRIAL_FIELDS, [row], logger=logger)
        logger.info("Trial row saved | trial=%d", row["trial_idx_global"])

    try:
        logger.info("Experiment setup start | acquisition_mode=%s", acquisition_mode)
        logger.info("Frequencies: %s", ", ".join(f"{f:.2f} Hz" for f in frequencies_hz))
        logger.info("Output folder: %s", paths["folder"])

        # --- Cortex / live setup ---
        marker = Marker(debug_mode=False)
        marker.connect(headset_id=headset_id)
        logger.info("Marker connect requested")

        if not marker.wait_for_session(timeout=90.0):
            raise RuntimeError("Cortex session is not available.")
        logger.info("Marker session ready")
        impedance_monitor = ImpedanceMonitor(marker.c, log=logger)
        impedance_monitor.start()

        if acquisition_mode == "record":
            recording = CortexRecording(marker, log=logger)
            if not recording.start_record(record_title, description=f"BCI BA {PHASE_LABEL}"):
                raise RuntimeError("EEG recording could not be started.")
            logger.info("Cortex record started | title=%s", record_title)
        else:
            live = CortexLiveStream(
                marker=marker,
                output_dir=f"{paths['folder']}/live_stream",
                streams=live_streams,
                flush_every=live_flush_every,
                log=logger,
            )
            live.start(wait_labels_s=10.0)
            logger.info("Live stream started | streams=%s", ", ".join(live_streams))

        # --- PsychoPy ---
        win = make_window(fullscr=fullscreen, screen=screen_id)
        info_stim = make_info_stim(win)
        info_stim.text = "Connecting to Emotiv Cortex ...\n\nPlease wait."
        info_stim.draw()
        win.flip()

        measured_refresh: list[float | None] = []
        refresh_hz = measure_refresh_rate(win, measured_out=measured_refresh)
        measured_hz = measured_refresh[0] if measured_refresh else None
        protocol["config"]["monitor_refresh_hz"] = round(refresh_hz, 3)
        protocol["config"]["measured_refresh_hz"] = round(measured_hz, 3) if measured_hz else None
        protocol["config"]["freq_realization"] = [
            estimate_realized_frequency(f, refresh_hz) for f in frequencies_hz
        ]
        logger.info(
            "Refresh rate expected=%.3f Hz | measured=%s Hz",
            MONITOR_REFRESH_RATE,
            f"{measured_hz:.3f}" if measured_hz else "unavailable",
        )

        generator = StimulusGenerator(
            win,
            monitor_refresh_rate=refresh_hz,
            luminance_depth=luminance_depth,
            luminance_mod_kind=luminance_mod_kind,
            abort_check=lambda: marker is not None and not marker.is_headset_connected(),
        )
        fixation = build_fixation_stimuli(win)
        snapshot_protocol()

        # --- Intro ---
        est_min = _estimate_session_duration_min(
            baseline_open_s=baseline_open_s,
            total_trials=total_trials,
            stim_duration_s=stim_duration_s,
            fix_range_s=fix_range_s,
            iti_range_s=iti_range_s,
            block_count=len(frequency_order),
        )
        key = show_message(
            win,
            info_stim,
            f"{PHASE_LABEL}\n\n"
            f"Participant: {proband_id}\n"
            f"Session: {session_id}\n"
            f"Frequencies: {', '.join(f'{f:.2f}' for f in sorted(frequencies_hz))} Hz\n"
            f"Trials per frequency: {trials_per_freq}\n"
            f"Stimulus duration: {stim_duration_s:.1f} s\n"
            f"Acquisition mode: {acquisition_mode}\n"
            f"Estimated duration: ~{est_min:.0f} min\n\n"
            f"Please fixate the center cross during all trials.\n\n"
            f"Press SPACE to start.",
        )
        if key == "escape":
            aborted = True
            return

        # --- Baseline open ---
        key = show_message(
            win,
            info_stim,
            f"Baseline - eyes open\n\n"
            f"Please look calmly at the fixation cross.\n"
            f"Duration: {baseline_open_s:.0f} s\n\n"
            f"Press SPACE to start.",
        )
        if key == "escape":
            aborted = True
            return

        bl_start_flip = [0.0]
        bl_start_ms = [0.0]
        draw_all(fixation)
        _schedule_marker(
            win, marker,
            acquisition_mode=acquisition_mode, live=live,
            value=MARKER_BASELINE_OPEN_START, label="BASELINE_OPEN_START",
            session_id=session_id, proband_id=proband_id,
            block_idx=None, trial_idx_global=None,
            condition=None,
            pending_event_rows=pending_event_rows,
            flip_time_out=bl_start_flip, marker_time_ms_out=bl_start_ms, logger=logger,
        )
        win.flip()
        _flush_pending_events(pending_event_rows, event_rows, paths["events"], logger)

        bl_frames = seconds_to_frames(baseline_open_s, refresh_hz)
        if show_fixation(win, fixation, bl_frames - 1):
            aborted = True
            return

        bl_stop_flip = [0.0]
        bl_stop_ms = [0.0]
        _schedule_marker(
            win, marker,
            acquisition_mode=acquisition_mode, live=live,
            value=MARKER_BASELINE_OPEN_STOP, label="BASELINE_OPEN_STOP",
            session_id=session_id, proband_id=proband_id,
            block_idx=None, trial_idx_global=None,
            condition=None,
            pending_event_rows=pending_event_rows,
            flip_time_out=bl_stop_flip, marker_time_ms_out=bl_stop_ms, logger=logger,
        )
        win.flip()
        _flush_pending_events(pending_event_rows, event_rows, paths["events"], logger)

        baseline_rec = {
            "session_id": session_id,
            "proband_id": proband_id,
            "phase": PHASE_SHORT,
            "baseline_type": "eyes_open",
            "start_marker": MARKER_BASELINE_OPEN_START,
            "stop_marker": MARKER_BASELINE_OPEN_STOP,
            "t_start_marker_ms": round(bl_start_ms[0], 3),
            "t_stop_marker_ms": round(bl_stop_ms[0], 3),
            "t_start_monotonic": round(bl_start_flip[0], 6),
            "t_stop_monotonic": round(bl_stop_flip[0], 6),
            "actual_duration_s": round(bl_stop_flip[0] - bl_start_flip[0], 6),
        }
        protocol["baseline_log"].append(baseline_rec)
        baseline_event = {
            "event_type": "baseline_open",
            "session_id": session_id,
            "proband_id": proband_id,
            "phase": PHASE_SHORT,
            "baseline_type": "eyes_open",
            "block_idx": None,
            "trial_idx_global": None,
            "condition_id": None,
            "frequency_hz": None,
            "marker_value": None,
            "marker_label": None,
            "t_marker_ms": None,
            "t_monotonic": None,
            "t_block_monotonic": None,
            "actual_duration_s": baseline_rec["actual_duration_s"],
            "timestamp_iso": now_iso(),
        }
        event_rows.append(baseline_event)
        append_csv_rows(paths["events"], EVENT_FIELDS, [baseline_event], logger=logger)
        snapshot_protocol()

        # --- Frequency blocks ---
        global_trial = 0
        for block_idx, freq in enumerate(frequency_order, start=1):
            if check_escape():
                aborted = True
                break

            cond = condition_map[freq]
            key = show_message(
                win,
                info_stim,
                f"Block {block_idx}/{len(frequency_order)}\n\n"
                f"Frequency: {freq:.2f} Hz\n"
                f"Condition: C{cond['condition_id']:03d}\n"
                f"Trials: {trials_per_freq}\n\n"
                f"Press SPACE when ready.",
            )
            if key == "escape":
                aborted = True
                break

            block_rec = {
                "session_id": session_id,
                "proband_id": proband_id,
                "phase": PHASE_SHORT,
                "block_idx": block_idx,
                "t_block_monotonic": round(time.monotonic(), 6),
            }
            attach_condition_fields(block_rec, cond)
            protocol["block_log"].append(block_rec)

            block_event = {
                "event_type": "block_start",
                "session_id": session_id,
                "proband_id": proband_id,
                "phase": PHASE_SHORT,
                "baseline_type": None,
                "block_idx": block_idx,
                "trial_idx_global": None,
                "marker_value": None,
                "marker_label": None,
                "t_marker_ms": None,
                "t_monotonic": None,
                "t_block_monotonic": block_rec["t_block_monotonic"],
                "actual_duration_s": None,
                "timestamp_iso": now_iso(),
            }
            attach_condition_fields(block_event, cond)
            event_rows.append(block_event)
            append_csv_rows(paths["events"], EVENT_FIELDS, [block_event], logger=logger)
            snapshot_protocol()
            logger.info("Block start | block=%d | freq=%.2f Hz", block_idx, freq)

            trial_in_block = 1
            while trial_in_block <= trials_per_freq:
                if check_escape():
                    aborted = True
                    break

                planned_global_trial = global_trial + 1
                fix_s = rng.uniform(*fix_range_s)
                iti_s = rng.uniform(*iti_range_s)
                logger.info(
                    "Trial prep | trial=%d | block=%d | fix=%.3f | iti=%.3f",
                    planned_global_trial, block_idx, fix_s, iti_s
                )

                try:
                    if show_fixation(win, fixation, seconds_to_frames(fix_s, refresh_hz)):
                        aborted = True
                        break
                    raise_if_disconnected(marker)

                    global_trial = planned_global_trial
                    onset_flip = [0.0]
                    onset_ms = [0.0]
                    _schedule_marker(
                        win, marker,
                        acquisition_mode=acquisition_mode, live=live,
                        value=cond["start_marker"], label=condition_marker_label("TRIAL_ON", cond["condition_id"]),
                        session_id=session_id, proband_id=proband_id,
                        block_idx=block_idx, trial_idx_global=global_trial,
                        condition=cond,
                        pending_event_rows=pending_event_rows,
                        flip_time_out=onset_flip, marker_time_ms_out=onset_ms, logger=logger,
                    )
                    generator.run_stimulus({
                        "shape": "flicker",
                        "freq": freq,
                        "duration": stim_duration_s,
                        "size_deg": 5.0,
                    })
                    raise_if_disconnected(marker)
                    _flush_pending_events(pending_event_rows, event_rows, paths["events"], logger)

                    offset_flip = [0.0]
                    offset_ms = [0.0]
                    _schedule_marker(
                        win, marker,
                        acquisition_mode=acquisition_mode, live=live,
                        value=cond["stop_marker"], label=condition_marker_label("TRIAL_OFF", cond["condition_id"]),
                        session_id=session_id, proband_id=proband_id,
                        block_idx=block_idx, trial_idx_global=global_trial,
                        condition=cond,
                        pending_event_rows=pending_event_rows,
                        flip_time_out=offset_flip, marker_time_ms_out=offset_ms, logger=logger,
                    )
                    win.flip()
                    _flush_pending_events(pending_event_rows, event_rows, paths["events"], logger)
                except TrialConnectionLost:
                    pending_event_rows.clear()
                    global_trial = planned_global_trial - 1
                    recovery = recover_after_disconnect(
                        win=win,
                        info_stim=info_stim,
                        marker=marker,
                        marker_cortex=marker.c,
                        run_impedance_gate=run_impedance_gate,
                        stable_s=3.0,
                        quality_threshold=3,
                        timeout_s=120.0,
                        mode_label=f"recovery_block_{block_idx}_trial_{trial_in_block}",
                        monitor=impedance_monitor,
                        log=logger,
                    )
                    protocol.setdefault("recovery_events", []).append(recovery)
                    snapshot_protocol()
                    continue

                iti_frames = seconds_to_frames(iti_s, refresh_hz)
                if iti_frames > 1 and show_blank(win, iti_frames - 1):
                    aborted = True
                    break

                trial_rec = {
                    "session_id": session_id,
                    "proband_id": proband_id,
                    "phase": PHASE_SHORT,
                    "trial_idx_global": global_trial,
                    "block_idx": block_idx,
                    "trial_idx_in_block": trial_in_block,
                    "nominal_duration_s": round(stim_duration_s, 6),
                    "fix_duration_s": round(fix_s, 6),
                    "iti_duration_s": round(iti_s, 6),
                    "t_onset_marker_ms": round(onset_ms[0], 3),
                    "t_offset_marker_ms": round(offset_ms[0], 3),
                    "t_onset_monotonic": round(onset_flip[0], 6),
                    "t_offset_monotonic": round(offset_flip[0], 6),
                    "actual_duration_s": round(offset_flip[0] - onset_flip[0], 6),
                }
                attach_condition_fields(trial_rec, cond)
                append_trial(trial_rec)
                snapshot_protocol()
                logger.info(
                    "Trial done | trial=%d | actual_duration=%.3f s",
                    global_trial, trial_rec["actual_duration_s"]
                )
                trial_in_block += 1

            if aborted:
                break

        # --- Finalize ---
        protocol["aborted"] = aborted
        protocol["status"] = "aborted" if aborted else "completed"
        protocol["datetime_end"] = now_iso()

        if acquisition_mode == "record" and recording is not None:
            export_ok = recording.stop_and_export(
                export_folder=paths["folder"],
                data_types=["EEG", "MOTION", "PM", "BP"],
                export_format="CSV",
                version="V2",
                timeout=300.0,
            )
            protocol["export_ok"] = bool(export_ok)
            logger.info("Record export finished | ok=%s", export_ok)
        elif acquisition_mode == "stream" and live is not None:
            live.stop()
            protocol["live_summary"] = live.summary()
            protocol["export_ok"] = True
            logger.info("Live stream final summary | %s", protocol["live_summary"])

        snapshot_protocol()

        if win is not None:
            if aborted:
                show_message(
                    win,
                    info_stim,
                    f"{PHASE_LABEL} aborted.\n\n"
                    f"{protocol['completed_trials']}/{total_trials} trials completed.\n\n"
                    f"Press SPACE.",
                )
            else:
                show_message(
                    win,
                    info_stim,
                    f"{PHASE_LABEL} completed.\n\n"
                    f"{protocol['completed_trials']} trials saved.\n\n"
                    f"Press SPACE.",
                )

    except Exception:
        logger.exception("Critical error in %s", PHASE_LABEL)
        protocol["aborted"] = True
        protocol["status"] = "error"
        protocol["datetime_end"] = now_iso()

        try:
            if acquisition_mode == "stream" and live is not None:
                live.stop()
                protocol["live_summary"] = live.summary()
        except Exception:
            logger.exception("Failed to stop live stream after error")

        try:
            snapshot_protocol()
        except Exception:
            logger.exception("Failed to snapshot protocol after error")
        raise

    finally:
        if impedance_monitor is not None:
            try:
                impedance_monitor.stop()
            except Exception:
                pass
        if win is not None:
            try:
                win.close()
            except Exception:
                pass
        if marker is not None:
            try:
                marker.close()
            except Exception:
                pass
        core.quit()


if __name__ == "__main__":
    print("Frequency pretest module demo")
    print("This demo does not open PsychoPy, connect to Cortex, or start a recording.")
    print(f"Phase: {PHASE_SHORT} ({PHASE_LABEL})")
    print(f"Baseline markers: {MARKER_BASELINE_OPEN_START}, {MARKER_BASELINE_OPEN_STOP}")
    print(f"Default frequencies: {DEFAULT_FREQUENCIES_HZ}")


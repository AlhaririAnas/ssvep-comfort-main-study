"""
stimulus_screening.py
=====================
Phase 2 of the BCI pipeline: stimulus screening with block-level ratings.

Supports two acquisition modes
------------------------------
- "record": Cortex record + export at the end.
- "stream": live Cortex stream writing via live_stream.py.

In both modes, protocol / trials / ratings / events are saved continuously.
In stream mode, marker information is embedded directly into eeg.csv.
In record mode, marker information is embedded by Cortex in the exported CSV.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __name__ == "__main__":  # pragma: no cover - supports direct module demos
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psychopy import core, event, gui, visual

from core.conditions import (
    CONDITION_SCHEMA_VERSION,
    assign_condition_ids,
    attach_condition_fields,
    build_stimulus_condition_spec,
    condition_marker_label,
    resolve_active_modulation,
)
from acquisition.live_stream import CortexLiveStream
from acquisition.impedance import ImpedanceMonitor, run_impedance_gate
from acquisition.marker import Marker
from acquisition.recording import CortexRecording
from acquisition.recovery import TrialConnectionLost, raise_if_disconnected, recover_after_disconnect
from stimuli.generator import StimulusGenerator, build_fixation_stimuli, make_window, normalize_stimulus_cfg
from core.config import (
    DEFAULT_PROBAND_ID,
    DEFAULT_OUT_DIR,
    DEFAULT_SCREEN_ID,
    DEFAULT_FULLSCREEN,
    DEFAULT_HEADSET_ID,
    DEFAULT_STIMULI_JSON,
    DEFAULT_TRIALS_PER_STIM,
    DEFAULT_BASELINE_OPEN_S,
    DEFAULT_BASELINE_CLOSED_S,
    DEFAULT_FIX_RANGE_S,
    DEFAULT_ITI_RANGE_S,
    DEFAULT_LUM_MOD_KIND,
    DEFAULT_LUM_DEPTH,
    DEFAULT_LIVE_STREAMS,
    DEFAULT_LIVE_FLUSH_EVERY,
    DEFAULT_ACQUISITION_MODE,
    DEFAULT_BLOCK_BREAK_S,
    DEFAULT_LONG_BREAK_AFTER_BLOCKS,
    DEFAULT_LONG_BREAK_S,
    DEFAULT_INCLUDE_EYES_CLOSED_BASELINE,
    DEFAULT_MS_IMPEDANCE_QUALITY_THRESHOLD,
    DEFAULT_MS_IMPEDANCE_STABLE_S_INITIAL,
    DEFAULT_MS_IMPEDANCE_TIMEOUT_S,
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
    save_session_outputs,
)
from stimuli.display import (
    seconds_to_frames,
    measure_refresh_rate,
    check_escape,
    draw_all,
    show_fixation,
    show_blank,
    show_message,
    make_info_stim,
)

# ---------------------------------------------------------------------------
# Marker codes
# ---------------------------------------------------------------------------
MARKER_BASELINE_OPEN_START = 10
MARKER_BASELINE_OPEN_STOP = 11
MARKER_BASELINE_CLOSED_START = 20
MARKER_BASELINE_CLOSED_STOP = 21
MARKER_TRIAL_START_BASE = 100
MARKER_TRIAL_STOP_BASE = 150

PHASE_SHORT = "stimscr"
PHASE_LABEL = "Stimulus Screening"

STIMULUS_FAMILY_ORDER = ("ssmvep", "hybrid", "ssvep")
_DIGIT_KEYS = {str(i) for i in range(7)}

# Keep the runtime prompts simple and readable.
COMFORT_ITEMS: List[Dict[str, str]] = [
    {
        "item_id": "flicker_discomfort",
        "label": "Flicker discomfort",
        "construct": "sensory - stimulus related",
        "question": "How uncomfortable was the flicker or motion in this block?",
        "anchors": "0 = not uncomfortable at all          6 = extremely uncomfortable",
    },
    {
        "item_id": "eye_fatigue",
        "label": "Eye fatigue",
        "construct": "physiological - ocular",
        "question": "How tired or strained did your eyes feel after this block?",
        "anchors": "0 = not tired at all          6 = extremely tired",
    },
    {
        "item_id": "physical_symptoms",
        "label": "Physical symptoms",
        "construct": "physiological - systemic",
        "question": "Did you notice headache, nausea, dizziness, or other symptoms?",
        "anchors": "0 = no symptoms          6 = strong symptoms",
    },
    {
        "item_id": "focus_maintenance",
        "label": "Focus",
        "construct": "cognitive - task related",
        "question": "How well could you keep your attention on the target?",
        "anchors": "0 = not well at all          6 = very well",
    },
    {
        "item_id": "overall_discomfort",
        "label": "Overall strain",
        "construct": "global - overall judgement",
        "question": "How demanding was this block overall?",
        "anchors": "0 = not demanding at all          6 = extremely demanding",
    },
]

TRIAL_FIELDS = [
    "session_id", "proband_id", "phase",
    "trial_idx_global", "block_idx", "trial_idx_in_block",
    "condition_id", "condition_numeric_id", "condition_label", "condition_key",
    "stimulus_id", "stimulus_name", "stimulus_type", "stimulus_shape",
    "frequency_hz", "luminance_mod_kind", "luminance_depth",
    "lum_mod_kind", "lum_depth",
    "nominal_duration_s", "fix_duration_s", "iti_duration_s",
    "start_marker", "stop_marker",
    "t_onset_marker_ms", "t_offset_marker_ms",
    "t_onset_monotonic", "t_offset_monotonic",
    "actual_duration_s",
]

RATING_FIELDS = [
    "session_id", "proband_id", "phase",
    "block_idx", "condition_id", "stimulus_id", "stimulus_name",
    "item_idx", "item_id", "item_label", "item_construct",
    "rating", "scale_min", "scale_max", "timestamp_iso",
]

EVENT_FIELDS = [
    "event_type", "session_id", "proband_id", "phase",
    "baseline_type", "block_idx", "trial_idx_global", "after_block_idx",
    "condition_id", "condition_numeric_id", "condition_label", "condition_key",
    "stimulus_id", "stimulus_name", "stimulus_type", "stimulus_shape",
    "frequency_hz", "luminance_mod_kind", "luminance_depth",
    "break_type", "planned_duration_s",
    "next_stimulus_id", "next_stimulus_name",
    "marker_value", "marker_label", "t_marker_ms", "t_monotonic",
    "t_block_monotonic", "actual_duration_s", "timestamp_iso",
]

COMFORT_ITEM_TEXT_DE: Dict[str, Dict[str, str]] = {
    "flicker_discomfort": {
        "question": "Wie unangenehm war das Flackern oder die Bewegung in diesem Block?",
        "anchors": "0 = gar nicht unangenehm          6 = extrem unangenehm",
    },
    "eye_fatigue": {
        "question": "Wie müde oder belastet haben sich deine Augen nach diesem Block angefühlt?",
        "anchors": "0 = gar nicht müde          6 = extrem müde",
    },
    "physical_symptoms": {
        "question": "Hast du Kopfschmerzen, Übelkeit, Schwindel oder andere Symptome bemerkt?",
        "anchors": "0 = keine Symptome          6 = starke Symptome",
    },
    "focus_maintenance": {
        "question": "Wie gut konntest du deine Aufmerksamkeit auf dem Ziel halten?",
        "anchors": "0 = gar nicht gut          6 = sehr gut",
    },
    "overall_discomfort": {
        "question": "Wie anstrengend war dieser Block insgesamt?",
        "anchors": "0 = gar nicht anstrengend          6 = extrem anstrengend",
    },
}


def _estimate_session_duration_min(
    *,
    baseline_open_s: float,
    baseline_closed_s: float,
    include_eyes_closed_baseline: bool,
    total_trials: int,
    total_trial_stimulus_s: float,
    fix_range_s: tuple[float, float],
    iti_range_s: tuple[float, float],
    block_break_s: float,
    long_break_s: float,
    long_break_after_blocks: Sequence[int],
    block_count: int,
    rating_s_per_block: float = 60.0,
    safety_buffer_min: float = 5.0,
) -> float:
    avg_fix_s = sum(float(v) for v in fix_range_s) / 2.0
    avg_iti_s = sum(float(v) for v in iti_range_s) / 2.0
    long_break_set = {int(v) for v in long_break_after_blocks}
    break_total_s = 0.0
    for block_idx in range(1, max(0, int(block_count))):
        break_total_s += float(long_break_s if block_idx in long_break_set else block_break_s)

    total_seconds = (
        float(baseline_open_s)
        + (float(baseline_closed_s) if include_eyes_closed_baseline else 0.0)
        + float(total_trial_stimulus_s)
        + total_trials * (avg_fix_s + avg_iti_s)
        + break_total_s
        + max(0, int(block_count)) * float(rating_s_per_block)
        + float(safety_buffer_min) * 60.0
    )
    return total_seconds / 60.0


def _collect_trial_count(default_value: int) -> int:
    """Show an editable runtime field for the trial count."""

    dlg = gui.Dlg(title=f"{PHASE_LABEL} - runtime settings")
    dlg.addText("Review and edit the session settings.")
    dlg.addField("Trials per stimulus", initial=int(default_value))
    data = dlg.show()
    if not dlg.OK:
        raise RuntimeError("Runtime settings dialog cancelled by operator.")
    raw_value = data[0] if isinstance(data, list) else list(data.values())[0]
    value = int(raw_value)
    if value < 1:
        raise ValueError("Trials per stimulus must be >= 1.")
    return value


def _load_stimuli(path: str, pilot_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stimuli-Konfiguration nicht gefunden: {path}")
    with open(path, "r", encoding="utf-8-sig") as fh:
        raw = json.load(fh)

    stimuli = [normalize_stimulus_cfg(cfg) for cfg in raw]
    required = {"id", "name", "type", "freq", "duration", "shape"}
    for stim in stimuli:
        missing = sorted(required - set(stim.keys()))
        if missing:
            raise KeyError(f"Stimulus {stim.get('id', '?')} is missing fields: {missing}")
        stim["id"] = int(stim["id"])
        stim["freq"] = float(stim["freq"])
        stim["duration"] = float(stim["duration"])
        stim["type"] = str(stim["type"]).lower()
        if stim["freq"] <= 0 or stim["duration"] <= 0:
            raise ValueError(f"Stimulus {stim['id']}: frequency and duration must be > 0.")

    if len({stim["id"] for stim in stimuli}) != len(stimuli):
        raise ValueError("Stimulus IDs must be unique.")

    unknown_types = sorted({stim["type"] for stim in stimuli} - set(STIMULUS_FAMILY_ORDER))
    if unknown_types:
        raise ValueError(f"Unknown stimulus types: {unknown_types}")

    stimuli.sort(key=lambda stim: stim["id"])
    if pilot_ids:
        allowed = {int(v) for v in pilot_ids}
        stimuli = [stim for stim in stimuli if stim["id"] in allowed]
        if not stimuli:
            raise ValueError(f"No stimuli found for pilot IDs: {sorted(allowed)}")
    return stimuli


def _build_condition_map(
    stimuli: List[Dict[str, Any]],
    *,
    luminance_mod_kind: str,
    luminance_depth: float,
) -> Dict[int, Dict[str, Any]]:
    specs = [
        build_stimulus_condition_spec(
            stim,
            default_luminance_mod_kind=luminance_mod_kind,
            default_luminance_depth=luminance_depth,
        )
        for stim in stimuli
    ]
    catalog = assign_condition_ids(specs)
    by_stimulus_id: Dict[int, Dict[str, Any]] = {}
    for cond in catalog["conditions"]:
        stim_id = cond.get("source_stimulus_id")
        if stim_id is None:
            continue
        by_stimulus_id[int(stim_id)] = cond
    return by_stimulus_id


def _build_block_order(stimuli: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    if any("presentation_order" in stim for stim in stimuli):
        return [
            copy.deepcopy(stim)
            for stim in sorted(
                stimuli,
                key=lambda stim: (int(stim.get("presentation_order", 9999)), int(stim["id"])),
            )
        ]

    if any("presentation_group" in stim for stim in stimuli):
        block_order: List[Dict[str, Any]] = []
        group_ids = sorted({int(stim.get("presentation_group", 999)) for stim in stimuli})
        for group_id in group_ids:
            group_items = [
                copy.deepcopy(stim)
                for stim in stimuli
                if int(stim.get("presentation_group", 999)) == group_id
            ]
            rng.shuffle(group_items)
            block_order.extend(group_items)
        return block_order

    block_order: List[Dict[str, Any]] = []
    for family in STIMULUS_FAMILY_ORDER:
        family_items = [copy.deepcopy(stim) for stim in stimuli if str(stim["type"]).lower() == family]
        rng.shuffle(family_items)
        block_order.extend(family_items)
    return block_order


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
    stimulus_id: Optional[int],
    stimulus_name: Optional[str],
    stimulus_type: Optional[str],
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
                "stimulus_id": stimulus_id,
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
            "after_block_idx": None,
            "stimulus_id": stimulus_id,
            "stimulus_name": stimulus_name,
            "stimulus_type": stimulus_type,
            "break_type": None,
            "planned_duration_s": None,
            "next_stimulus_id": None,
            "next_stimulus_name": None,
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


def _run_countdown_screen(
    win: visual.Window,
    info_stim: visual.TextStim,
    *,
    title: str,
    body_lines: Sequence[str],
    duration_s: float,
    countdown_label: str = "Continue in",
    ready_text: str = "Press SPACE to continue.",
) -> bool:
    event.clearEvents(eventType="keyboard")
    deadline = time.monotonic() + max(0.0, float(duration_s))
    while True:
        remaining_s = max(0.0, deadline - time.monotonic())
        remaining_display = int(math.ceil(remaining_s))
        footer = f"{countdown_label}: {remaining_display} s" if remaining_s > 0.0 else ready_text
        info_stim.text = "\n\n".join([title, *body_lines, footer])
        info_stim.draw()
        win.flip()
        keys = event.getKeys(keyList=["escape", "space"])
        if "escape" in keys:
            return True
        if remaining_s <= 0.0:
            if "space" in keys:
                return False
        core.wait(0.05)


def _stimulus_modulation_lines(
    stimulus_type: str,
    ssvep_kind: str,
    ssvep_depth: float,
) -> List[str]:
    """Return clear modulation lines for one stimulus type."""

    stim_type = str(stimulus_type).lower()
    if stim_type == "hybrid":
        return [
            "SSMVEP Modulation: sine, 100%",
            f"SSVEP Modulation: {ssvep_kind}, {ssvep_depth:.0%}",
        ]
    if stim_type == "ssmvep":
        return ["SSMVEP Modulation: sine, 100%"]
    return [f"SSVEP Modulation: {ssvep_kind}, {ssvep_depth:.0%}"]


class RatingDisplay:
    def __init__(self, win: visual.Window, language: str = "en") -> None:
        self.language = str(language).lower()
        self.header = visual.TextStim(win, units="height", color="white", height=0.038, pos=(0.0, 0.40), wrapWidth=1.55, alignText="center")
        self.question = visual.TextStim(win, units="height", color="white", height=0.048, pos=(0.0, 0.12), wrapWidth=1.35, alignText="center")
        self.anchors = visual.TextStim(win, units="height", color="white", height=0.036, pos=(0.0, -0.10), wrapWidth=1.40, alignText="center")
        self.scale = visual.TextStim(win, units="height", color="white", height=0.040, pos=(0.0, -0.22), wrapWidth=1.50, alignText="center")
        self.footer = visual.TextStim(win, units="height", color="white", height=0.032, pos=(0.0, -0.36), wrapWidth=1.55, alignText="center")

    def draw_question(self, *, block_idx: int, stimulus_name: str, item_idx: int, item_count: int, item: Dict[str, str], selection: Optional[int]) -> None:
        scale = "      ".join(f"[{i}]" if selection == i else f" {i} " for i in range(7))
        current = "-" if selection is None else str(selection)
        if self.language == "de":
            de_text = COMFORT_ITEM_TEXT_DE.get(item["item_id"], {})
            self.header.text = f"Subjektive Bewertung - Block {block_idx} | {stimulus_name}\n\nFrage {item_idx}/{item_count}"
            self.question.text = de_text.get("question", item["question"])
            self.anchors.text = de_text.get("anchors", item["anchors"])
        else:
            self.header.text = f"Subjective Rating - Block {block_idx} | {stimulus_name}\n\nQuestion {item_idx}/{item_count}"
            self.question.text = item["question"]
            self.anchors.text = item["anchors"]
        self.scale.text = scale
        self.footer.text = (
            f"Aktuelle Antwort: {current}\n\nWähle 0-6, drücke ENTER zum Bestätigen, ESC zum Abbrechen"
            if self.language == "de" else
            f"Current answer: {current}\n\nChoose 0-6, press ENTER to confirm, ESC to stop"
        )
        self.header.draw()
        self.question.draw()
        self.anchors.draw()
        self.scale.draw()
        self.footer.draw()


def _collect_ratings(
    win: visual.Window,
    rating_ui: RatingDisplay,
    *,
    session_id: str,
    proband_id: str,
    block_idx: int,
    condition_id: str,
    stimulus_id: int,
    stimulus_name: str,
    language: str = "en",
) -> Optional[List[Dict[str, Any]]]:
    rating_ui.language = str(language).lower()
    rows: List[Dict[str, Any]] = []
    for item_idx, item in enumerate(COMFORT_ITEMS, start=1):
        current: Optional[int] = None
        confirmed = False
        while not confirmed:
            rating_ui.draw_question(
                block_idx=block_idx,
                stimulus_name=stimulus_name,
                item_idx=item_idx,
                item_count=len(COMFORT_ITEMS),
                item=item,
                selection=current,
            )
            win.flip()
            event.clearEvents(eventType="keyboard")
            keys = event.waitKeys(keyList=sorted(_DIGIT_KEYS | {"return", "num_enter", "escape"}))
            if not keys:
                continue
            key = keys[0]
            if key == "escape":
                return None
            if key in _DIGIT_KEYS:
                current = int(key)
            elif key in {"return", "num_enter"} and current is not None:
                confirmed = True
        rows.append({
            "session_id": session_id,
            "proband_id": proband_id,
            "phase": PHASE_SHORT,
            "block_idx": block_idx,
            "condition_id": condition_id,
            "stimulus_id": stimulus_id,
            "stimulus_name": stimulus_name,
            "item_idx": item_idx,
            "item_id": item["item_id"],
            "item_label": item["label"],
            "item_construct": item.get("construct", ""),
            "rating": int(current),
            "scale_min": 0,
            "scale_max": 6,
            "timestamp_iso": now_iso(),
        })
    event.clearEvents(eventType="keyboard")
    return rows


# ---------------------------------------------------------------------------
# Environment metadata dialog
# ---------------------------------------------------------------------------
ENVIRONMENT_METADATA_FIELDS = {
    "monitor_brightness_pct": "80",
    "monitor_contrast_pct": "50",
    "room_lighting": "dim",
    "room_temperature_c": "22",
    "caffeine_last_2h": "no",
    "visual_aid": "no",
    "notes": "",
}

ENVIRONMENT_FIELD_LABELS = {
    "monitor_brightness_pct": "Monitor brightness (%)",
    "monitor_contrast_pct": "Monitor contrast (%)",
    "room_lighting": "Room lighting (dark / dim / bright)",
    "room_temperature_c": "Room temperature (degree C)",
    "caffeine_last_2h": "Caffeine in the last 2 h (yes / no)",
    "visual_aid": "Visual aid (no / glasses / contacts)",
    "notes": "Notes (optional)",
}

_ROOM_LIGHTING_OPTIONS = ("dark", "dim", "bright")


def _collect_environment_metadata() -> Dict[str, Any]:
    """Show a pre-experiment dialog for environment metadata.

    Returns a dict with the entered values.  If the user presses Cancel
    the defaults are returned (acts as 'skip').
    """
    fields = dict(ENVIRONMENT_METADATA_FIELDS)

    dlg = gui.Dlg(
        title="Environment Metadata (Cancel = keep defaults)",
        alwaysOnTop=True,
    )
    dlg.addText("Please fill this in before the experiment. Cancel keeps the defaults.")
    dlg.addField(ENVIRONMENT_FIELD_LABELS["monitor_brightness_pct"], fields["monitor_brightness_pct"])
    dlg.addField(ENVIRONMENT_FIELD_LABELS["monitor_contrast_pct"], fields["monitor_contrast_pct"])
    dlg.addField(ENVIRONMENT_FIELD_LABELS["room_lighting"], choices=list(_ROOM_LIGHTING_OPTIONS), initial=fields["room_lighting"])
    dlg.addField(ENVIRONMENT_FIELD_LABELS["room_temperature_c"], fields["room_temperature_c"])
    dlg.addField(ENVIRONMENT_FIELD_LABELS["caffeine_last_2h"], fields["caffeine_last_2h"])
    dlg.addField(ENVIRONMENT_FIELD_LABELS["visual_aid"], fields["visual_aid"])
    dlg.addField(ENVIRONMENT_FIELD_LABELS["notes"], fields["notes"])

    data = dlg.show()

    if dlg.OK and data is not None:
        keys = list(ENVIRONMENT_METADATA_FIELDS.keys())
        for i, key in enumerate(keys):
            if i < len(data):
                fields[key] = str(data[i]).strip()

    # Normalize numeric fields
    for num_key in ("monitor_brightness_pct", "monitor_contrast_pct", "room_temperature_c"):
        try:
            fields[num_key] = float(fields[num_key])
        except (ValueError, TypeError):
            pass

    fields["skipped"] = not dlg.OK
    fields["timestamp_iso"] = now_iso()
    return fields


def run_stimulus_screening(cfg: Dict[str, Any]) -> None:
    marker: Optional[Marker] = None
    recording: Optional[CortexRecording] = None
    live: Optional[CortexLiveStream] = None
    impedance_monitor: Optional[ImpedanceMonitor] = None
    win: Optional[visual.Window] = None
    aborted = False

    proband_id = str(cfg.get("proband_id", DEFAULT_PROBAND_ID))
    folder_name = cfg.get("folder_name")
    stimuli_json = str(cfg.get("stimuli_json", DEFAULT_STIMULI_JSON))
    pilot_ids = cfg.get("pilot_ids")
    trials_per_stim = int(cfg.get("trials_per_stim", DEFAULT_TRIALS_PER_STIM))
    trials_per_stim = _collect_trial_count(trials_per_stim)
    baseline_open_s = float(cfg.get("baseline_open_s", DEFAULT_BASELINE_OPEN_S))
    baseline_closed_s = float(cfg.get("baseline_closed_s", DEFAULT_BASELINE_CLOSED_S))
    include_eyes_closed_baseline = bool(cfg.get("include_eyes_closed_baseline", DEFAULT_INCLUDE_EYES_CLOSED_BASELINE))
    fix_range_s = tuple(cfg.get("fix_range_s", DEFAULT_FIX_RANGE_S))
    iti_range_s = tuple(cfg.get("iti_range_s", DEFAULT_ITI_RANGE_S))
    luminance_mod_kind = str(cfg.get("luminance_mod_kind", DEFAULT_LUM_MOD_KIND))
    luminance_depth = float(cfg.get("luminance_depth", DEFAULT_LUM_DEPTH))
    block_break_s = float(cfg.get("block_break_s", DEFAULT_BLOCK_BREAK_S))
    long_break_after_blocks = tuple(int(v) for v in cfg.get("long_break_after_blocks", DEFAULT_LONG_BREAK_AFTER_BLOCKS))
    long_break_s = float(cfg.get("long_break_s", DEFAULT_LONG_BREAK_S))
    impedance_stable_s = float(cfg.get("impedance_stable_s", DEFAULT_MS_IMPEDANCE_STABLE_S_INITIAL))
    impedance_quality_threshold = int(cfg.get("impedance_quality_threshold", DEFAULT_MS_IMPEDANCE_QUALITY_THRESHOLD))
    impedance_timeout_s = float(cfg.get("impedance_timeout_s", DEFAULT_MS_IMPEDANCE_TIMEOUT_S))
    screen_id = int(cfg.get("screen_id", DEFAULT_SCREEN_ID))
    fullscreen = bool(cfg.get("fullscreen", DEFAULT_FULLSCREEN))
    out_dir = str(cfg.get("out_dir", DEFAULT_OUT_DIR))
    headset_id = str(cfg.get("headset_id", DEFAULT_HEADSET_ID))
    seed = int(cfg.get("seed") or int(time.time() * 1000) % (2**32))
    acquisition_mode = str(cfg.get("acquisition_mode", cfg.get("storage_mode", DEFAULT_ACQUISITION_MODE))).lower()
    live_streams = [str(s).lower() for s in cfg.get("live_streams", list(DEFAULT_LIVE_STREAMS))]
    live_flush_every = int(cfg.get("live_flush_every", DEFAULT_LIVE_FLUSH_EVERY))
    proband_metadata = copy.deepcopy(cfg.get("proband_metadata", {}))
    background_color = cfg.get("background_color")
    stim_on_color = cfg.get("stim_on_color")
    stim_off_color = cfg.get("stim_off_color")

    if acquisition_mode not in {"record", "stream"}:
        raise ValueError("acquisition_mode must be 'record' or 'stream'.")
    if trials_per_stim < 1:
        raise ValueError("trials_per_stim must be >= 1.")
    if not 0.0 <= luminance_depth <= 1.0:
        raise ValueError("luminance_depth must be between 0.0 and 1.0.")
    if block_break_s < 0.0:
        raise ValueError("block_break_s must be >= 0.")
    if long_break_s < 180.0:
        raise ValueError("long_break_s must be at least 180 seconds.")

    stimuli = _load_stimuli(stimuli_json, pilot_ids=pilot_ids)
    condition_map = _build_condition_map(
        stimuli,
        luminance_mod_kind=luminance_mod_kind,
        luminance_depth=luminance_depth,
    )

    run_timestamp = build_run_timestamp()
    session_id, _ = determine_next_session_id(proband_id, PHASE_SHORT, out_dir, folder_name=folder_name)
    paths = build_output_paths(
        proband_id=proband_id,
        session_id=session_id,
        phase_short=PHASE_SHORT,
        run_timestamp=run_timestamp,
        out_dir=out_dir,
        folder_name=folder_name,
    ).as_dict()
    record_title = build_cortex_record_title(proband_id, PHASE_SHORT, session_id, run_timestamp)
    logger = setup_logging(paths["log"], f"bci.{PHASE_SHORT}")
    rng = random.Random(seed)

    # Collect environment metadata before fullscreen window opens.
    # Cancel in the dialog skips with defaults (no data lost).
    environment_metadata = _collect_environment_metadata()
    logger.info(
        "Environment metadata collected | skipped=%s | brightness=%s | lighting=%s",
        environment_metadata.get("skipped"),
        environment_metadata.get("monitor_brightness_pct"),
        environment_metadata.get("room_lighting"),
    )

    block_order = _build_block_order(stimuli, rng)
    total_trials = len(block_order) * trials_per_stim
    ordered_conditions = [
        condition_map[stim_id]
        for stim_id in sorted(condition_map, key=lambda value: condition_map[value]["condition_id"])
    ]

    protocol: Dict[str, Any] = {
        "study": "BCI_BA",
        "phase": PHASE_SHORT,
        "proband_id": proband_id,
        "session_id": session_id,
        "datetime_start": now_iso(),
        "random_seed": seed,
        "environment_metadata": environment_metadata,
        "proband_metadata": proband_metadata,
        "status": "running",
        "aborted": False,
        "completed_trials": 0,
        "total_trials": total_trials,
        "config": {
            "stimuli_json": stimuli_json,
            "trials_per_stim": trials_per_stim,
            "baseline_open_s": baseline_open_s,
            "baseline_closed_s": baseline_closed_s,
            "include_eyes_closed_baseline": include_eyes_closed_baseline,
            "fix_range_s": list(fix_range_s),
            "iti_range_s": list(iti_range_s),
            "lum_mod_kind": luminance_mod_kind,
            "lum_depth": luminance_depth,
            "block_break_s": block_break_s,
            "long_break_after_blocks": list(long_break_after_blocks),
            "long_break_s": long_break_s,
            "impedance_stable_s": impedance_stable_s,
            "impedance_quality_threshold": impedance_quality_threshold,
            "impedance_timeout_s": impedance_timeout_s,
            "screen_id": screen_id,
            "fullscreen": fullscreen,
            "headset_id": headset_id,
            "acquisition_mode": acquisition_mode,
            "live_streams": live_streams if acquisition_mode == "stream" else [],
            "live_flush_every": live_flush_every if acquisition_mode == "stream" else None,
            "expected_refresh_hz": round(MONITOR_REFRESH_RATE, 3),
            "monitor_refresh_hz": None,
            "background_color": background_color,
            "stim_on_color": stim_on_color,
            "stim_off_color": stim_off_color,
        },
        "stimulus_definition": copy.deepcopy(stimuli),
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
        "family_order": list(STIMULUS_FAMILY_ORDER),
        "block_order": [condition_map[int(s["id"])]["condition_id"] for s in block_order],
        "baseline_log": [],
        "block_log": [],
        "trial_log": [],
        "comfort_log": [],
        "impedance_events": [],
        "break_log": [],
        "event_log": [],
        "marker_timing_log": [],
        "export_ok": False,
        "live_summary": {},
        "output_folder": paths["folder"],
        "datetime_end": None,
    }

    trial_rows: List[Dict[str, Any]] = []
    rating_rows: List[Dict[str, Any]] = []
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

    def append_ratings(rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        rating_rows.extend(rows)
        protocol["comfort_log"].extend(rows)
        append_csv_rows(paths["ratings"], RATING_FIELDS, rows, logger=logger)
        logger.info("Rating rows saved | count=%d", len(rows))

    try:
        logger.info("Experiment setup start | acquisition_mode=%s", acquisition_mode)
        marker = Marker(debug_mode=False)
        marker.connect(headset_id=headset_id)
        logger.info("Marker connect requested")
        if not marker.wait_for_session(timeout=90.0):
            raise RuntimeError("No Cortex session available.")
        logger.info("Marker session ready")

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

        win = make_window(
            fullscr=fullscreen,
            screen=screen_id,
            background_color=background_color or (-0.8, -0.8, -0.8),
        )
        info_stim = make_info_stim(win)
        measured_refresh: List[float | None] = []
        refresh_hz = measure_refresh_rate(win, measured_out=measured_refresh)
        measured_hz = measured_refresh[0] if measured_refresh else None
        protocol["config"]["monitor_refresh_hz"] = round(refresh_hz, 3)
        protocol["config"]["measured_refresh_hz"] = round(measured_hz, 3) if measured_hz else None
        logger.info(
            "Refresh rate expected=%.3f Hz | measured=%s Hz",
            MONITOR_REFRESH_RATE,
            f"{measured_hz:.3f}" if measured_hz else "unavailable",
        )

        generator_lum = StimulusGenerator(
            win,
            monitor_refresh_rate=refresh_hz,
            luminance_depth=luminance_depth,
            luminance_mod_kind=luminance_mod_kind,
            background_color=background_color or (-0.8, -0.8, -0.8),
            stim_on_color=stim_on_color or (0.5, 1.0, 0.5),
            stim_off_color=stim_off_color,
            abort_check=lambda: marker is not None and not marker.is_headset_connected(),
        )
        generator_mot = StimulusGenerator(
            win,
            monitor_refresh_rate=refresh_hz,
            luminance_depth=1.0,
            luminance_mod_kind="sine",
            background_color=background_color or (-0.8, -0.8, -0.8),
            stim_on_color=stim_on_color or (0.5, 1.0, 0.5),
            stim_off_color=stim_off_color,
            abort_check=lambda: marker is not None and not marker.is_headset_connected(),
        )
        fixation = build_fixation_stimuli(win)
        rating_ui = RatingDisplay(win)

        impedance_monitor = ImpedanceMonitor(marker.c, log=logger)
        impedance_monitor.start()
        impedance_result = run_impedance_gate(
            win,
            marker.c,
            stable_s=impedance_stable_s,
            quality_threshold=impedance_quality_threshold,
            timeout_s=impedance_timeout_s,
            mode_label="initial",
            monitor=impedance_monitor,
        )
        protocol["impedance_events"].append(impedance_result)
        snapshot_protocol()
        if impedance_result.get("aborted"):
            aborted = True
            raise RuntimeError("Session aborted at initial impedance gate.")
        if impedance_result.get("timeout"):
            logger.warning("Initial impedance gate timed out; operator should check headset contact.")

        snapshot_protocol()

        est_min = _estimate_session_duration_min(
            baseline_open_s=baseline_open_s,
            baseline_closed_s=baseline_closed_s,
            include_eyes_closed_baseline=include_eyes_closed_baseline,
            total_trials=total_trials,
            total_trial_stimulus_s=sum(float(stim["duration"]) * trials_per_stim for stim in block_order),
            fix_range_s=fix_range_s,
            iti_range_s=iti_range_s,
            block_break_s=block_break_s,
            long_break_s=long_break_s,
            long_break_after_blocks=long_break_after_blocks,
            block_count=len(block_order),
        )
        key = show_message(
            win,
            info_stim,
            f"{PHASE_LABEL}\n\n"
            f"Participant: {proband_id}\n"
            f"Session: {session_id}\n"
            f"Stimuli: {len(block_order)}\n"
            f"Trials per stimulus: {trials_per_stim}\n"
            f"Family order: SSMVEP -> Hybrid -> SSVEP\n"
            f"Storage mode: {acquisition_mode}\n"
            f"Estimated duration: ~{est_min:.0f} min\n\n"
            f"After each block there is a short rating.\n"
            f"Please keep your eyes on the central fixation cross during all trials.\n\n"
            f"Press SPACE to start.",
        )
        if key == "escape":
            aborted = True
            return

        key = show_message(
            win,
            info_stim,
            f"Baseline - Eyes Open\n\n"
            f"Please look at the fixation cross and stay relaxed.\n"
            f"Duration: {baseline_open_s:.0f} s\n\n"
            f"Press SPACE to start.",
        )
        if key == "escape":
            aborted = True
            return

        blo_start_flip = [0.0]
        blo_start_ms = [0.0]
        draw_all(fixation)
        _schedule_marker(
            win, marker,
            acquisition_mode=acquisition_mode, live=live,
            value=MARKER_BASELINE_OPEN_START, label="BASELINE_OPEN_START",
            session_id=session_id, proband_id=proband_id,
            block_idx=None, trial_idx_global=None,
            condition=None, stimulus_id=None, stimulus_name=None, stimulus_type=None,
            pending_event_rows=pending_event_rows,
            flip_time_out=blo_start_flip, marker_time_ms_out=blo_start_ms, logger=logger,
        )
        win.flip()
        _flush_pending_events(pending_event_rows, event_rows, paths["events"], logger)
        if show_fixation(win, fixation, seconds_to_frames(baseline_open_s, refresh_hz) - 1):
            aborted = True
            return
        blo_stop_flip = [0.0]
        blo_stop_ms = [0.0]
        _schedule_marker(
            win, marker,
            acquisition_mode=acquisition_mode, live=live,
            value=MARKER_BASELINE_OPEN_STOP, label="BASELINE_OPEN_STOP",
            session_id=session_id, proband_id=proband_id,
            block_idx=None, trial_idx_global=None,
            condition=None, stimulus_id=None, stimulus_name=None, stimulus_type=None,
            pending_event_rows=pending_event_rows,
            flip_time_out=blo_stop_flip, marker_time_ms_out=blo_stop_ms, logger=logger,
        )
        win.flip()
        _flush_pending_events(pending_event_rows, event_rows, paths["events"], logger)

        blo_rec = {
            "session_id": session_id,
            "proband_id": proband_id,
            "phase": PHASE_SHORT,
            "baseline_type": "eyes_open",
            "start_marker": MARKER_BASELINE_OPEN_START,
            "stop_marker": MARKER_BASELINE_OPEN_STOP,
            "t_start_marker_ms": round(blo_start_ms[0], 3),
            "t_stop_marker_ms": round(blo_stop_ms[0], 3),
            "t_start_monotonic": round(blo_start_flip[0], 6),
            "t_stop_monotonic": round(blo_stop_flip[0], 6),
            "actual_duration_s": round(blo_stop_flip[0] - blo_start_flip[0], 6),
        }
        protocol["baseline_log"].append(blo_rec)
        baseline_event = {
            "event_type": "baseline_open",
            "session_id": session_id,
            "proband_id": proband_id,
            "phase": PHASE_SHORT,
            "baseline_type": "eyes_open",
            "block_idx": None,
            "trial_idx_global": None,
            "after_block_idx": None,
            "condition_id": None,
            "stimulus_id": None,
            "stimulus_name": None,
            "stimulus_type": None,
            "frequency_hz": None,
            "break_type": None,
            "planned_duration_s": None,
            "next_stimulus_id": None,
            "next_stimulus_name": None,
            "marker_value": None,
            "marker_label": None,
            "t_marker_ms": None,
            "t_monotonic": None,
            "t_block_monotonic": None,
            "actual_duration_s": blo_rec["actual_duration_s"],
            "timestamp_iso": now_iso(),
        }
        event_rows.append(baseline_event)
        append_csv_rows(paths["events"], EVENT_FIELDS, [baseline_event], logger=logger)
        snapshot_protocol()
        logger.info("Open baseline done | duration=%.3f s", blo_rec["actual_duration_s"])

        if include_eyes_closed_baseline:
            key = show_message(
                win,
                info_stim,
                f"Baseline - Eyes Closed\n\n"
                f"After the start, close your eyes and relax.\n"
                f"Duration: {baseline_closed_s:.0f} s\n\n"
                f"Press SPACE, then close your eyes.",
            )
            if key == "escape":
                aborted = True
                return
            blc_start_flip = [0.0]
            blc_start_ms = [0.0]
            _schedule_marker(
                win, marker,
                acquisition_mode=acquisition_mode, live=live,
                value=MARKER_BASELINE_CLOSED_START, label="BASELINE_CLOSED_START",
                session_id=session_id, proband_id=proband_id,
                block_idx=None, trial_idx_global=None,
                condition=None, stimulus_id=None, stimulus_name=None, stimulus_type=None,
                pending_event_rows=pending_event_rows,
                flip_time_out=blc_start_flip, marker_time_ms_out=blc_start_ms, logger=logger,
            )
            win.flip()
            _flush_pending_events(pending_event_rows, event_rows, paths["events"], logger)
            if show_blank(win, seconds_to_frames(baseline_closed_s, refresh_hz) - 1):
                aborted = True
                return
            blc_stop_flip = [0.0]
            blc_stop_ms = [0.0]
            _schedule_marker(
                win, marker,
                acquisition_mode=acquisition_mode, live=live,
                value=MARKER_BASELINE_CLOSED_STOP, label="BASELINE_CLOSED_STOP",
                session_id=session_id, proband_id=proband_id,
                block_idx=None, trial_idx_global=None,
                condition=None, stimulus_id=None, stimulus_name=None, stimulus_type=None,
                pending_event_rows=pending_event_rows,
                flip_time_out=blc_stop_flip, marker_time_ms_out=blc_stop_ms, logger=logger,
            )
            win.flip()
            _flush_pending_events(pending_event_rows, event_rows, paths["events"], logger)
            protocol["baseline_log"].append({
                "session_id": session_id,
                "proband_id": proband_id,
                "phase": PHASE_SHORT,
                "baseline_type": "eyes_closed",
                "start_marker": MARKER_BASELINE_CLOSED_START,
                "stop_marker": MARKER_BASELINE_CLOSED_STOP,
                "t_start_marker_ms": round(blc_start_ms[0], 3),
                "t_stop_marker_ms": round(blc_stop_ms[0], 3),
                "t_start_monotonic": round(blc_start_flip[0], 6),
                "t_stop_monotonic": round(blc_stop_flip[0], 6),
                "actual_duration_s": round(blc_stop_flip[0] - blc_start_flip[0], 6),
            })
            snapshot_protocol()

        global_trial = 0
        for block_idx, stim_cfg in enumerate(block_order, start=1):
            if check_escape():
                aborted = True
                break

            stim_id = int(stim_cfg["id"])
            stim_name = str(stim_cfg["name"])
            stim_type = str(stim_cfg["type"]).lower()
            stim_freq = float(stim_cfg["freq"])
            stim_dur = float(stim_cfg["duration"])
            cond = condition_map[stim_id]
            stim_shape = str(stim_cfg["shape"]).lower()

            if stim_type == "ssmvep":
                gen = generator_mot
            else:
                gen = generator_lum
            active_kind, active_depth = resolve_active_modulation(
                stim_type,
                luminance_mod_kind,
                luminance_depth,
            )
            modulation_lines = _stimulus_modulation_lines(
                stim_type,
                active_kind,
                active_depth,
            )

            if block_idx == 1:
                if _run_countdown_screen(
                    win,
                    info_stim,
                    title=f"Block {block_idx}/{len(block_order)}",
                    body_lines=[
                        f"Stimulus: {stim_name}  (C{cond['condition_id']:03d})",
                        f"Type: {stim_type.upper()}",
                        f"Frequency: {stim_freq:.2f} Hz",
                        *modulation_lines,
                    ],
                    duration_s=0.0,
                    countdown_label="Block starts in",
                    ready_text="Press SPACE to start this block.",
                ):
                    aborted = True
                    break

            block_rec = {
                "session_id": session_id,
                "proband_id": proband_id,
                "phase": PHASE_SHORT,
                "block_idx": block_idx,
                "stimulus_id": stim_id,
                "stimulus_name": stim_name,
                "stimulus_type": stim_type,
                "t_block_monotonic": round(time.monotonic(), 6),
            }
            attach_condition_fields(block_rec, cond)
            protocol["block_log"].append(block_rec)
            snapshot_protocol()
            logger.info("Block start | block=%d | stimulus=%s", block_idx, stim_name)

            trial_in_block = 1
            while trial_in_block <= trials_per_stim:
                if check_escape():
                    aborted = True
                    break

                planned_global_trial = global_trial + 1
                fix_s = rng.uniform(*fix_range_s)
                iti_s = rng.uniform(*iti_range_s)
                logger.info("Trial prep | trial=%d | block=%d | fix=%.3f | iti=%.3f", planned_global_trial, block_idx, fix_s, iti_s)
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
                        condition=cond, stimulus_id=stim_id, stimulus_name=stim_name,
                        stimulus_type=stim_type,
                        pending_event_rows=pending_event_rows,
                        flip_time_out=onset_flip, marker_time_ms_out=onset_ms, logger=logger,
                    )
                    logger.info("Trial start scheduled | trial=%d", global_trial)

                    gen.run_stimulus(stim_cfg)
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
                        condition=cond, stimulus_id=stim_id, stimulus_name=stim_name,
                        stimulus_type=stim_type,
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
                        stable_s=impedance_stable_s,
                        quality_threshold=impedance_quality_threshold,
                        timeout_s=impedance_timeout_s,
                        mode_label=f"recovery_block_{block_idx}_trial_{trial_in_block}",
                        monitor=impedance_monitor,
                        on_event=_impedance_event,
                        log=logger,
                    )
                    protocol.setdefault("recovery_events", []).append(recovery)
                    snapshot_protocol()
                    continue
                logger.info("Trial stop captured | trial=%d", global_trial)

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
                    "stimulus_id": stim_id,
                    "stimulus_name": stim_name,
                    "stimulus_type": stim_type,
                    "stimulus_shape": stim_shape,
                    "nominal_duration_s": round(stim_dur, 6),
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
                logger.info("Trial done | trial=%d | duration=%.3f s", global_trial, trial_rec["actual_duration_s"])
                trial_in_block += 1

            if aborted:
                break

            ratings = _collect_ratings(
                win, rating_ui,
                session_id=session_id, proband_id=proband_id,
                block_idx=block_idx, condition_id=cond["condition_id"],
                stimulus_id=stim_id, stimulus_name=stim_name,
            )
            if ratings is None:
                aborted = True
                break
            append_ratings(ratings)
            snapshot_protocol()
            logger.info("Ratings done | block=%d | count=%d", block_idx, len(ratings))

            if block_idx < len(block_order):
                next_stim = block_order[block_idx]
                is_long_break = block_idx in long_break_after_blocks
                pause_type = "long_break" if is_long_break else "short_break"
                pause_duration_s = long_break_s if is_long_break else block_break_s
                pause_event = {
                    "event_type": pause_type,
                    "session_id": session_id,
                    "proband_id": proband_id,
                    "phase": PHASE_SHORT,
                    "baseline_type": None,
                    "block_idx": None,
                    "trial_idx_global": None,
                    "after_block_idx": block_idx,
                    "condition_id": None,
                    "stimulus_id": None,
                    "stimulus_name": None,
                    "stimulus_type": None,
                    "frequency_hz": None,
                    "break_type": pause_type,
                    "planned_duration_s": round(pause_duration_s, 3),
                    "next_stimulus_id": int(next_stim["id"]),
                    "next_stimulus_name": str(next_stim["name"]),
                    "marker_value": None,
                    "marker_label": None,
                    "t_marker_ms": None,
                    "t_monotonic": None,
                    "t_block_monotonic": None,
                    "actual_duration_s": None,
                    "timestamp_iso": now_iso(),
                }
                event_rows.append(pause_event)
                append_csv_rows(paths["events"], EVENT_FIELDS, [pause_event], logger=logger)
                snapshot_protocol()
                logger.info("Pause start | type=%s | after_block=%d | duration=%.1f", pause_type, block_idx, pause_duration_s)
                next_cond = condition_map[int(next_stim["id"])]
                next_type = str(next_stim["type"]).lower()
                next_kind, next_depth = resolve_active_modulation(next_type, luminance_mod_kind, luminance_depth)
                body_lines = [
                    f"Block {block_idx}/{len(block_order)} finished.",
                    f"Next block: {next_stim['name']}  (C{next_cond['condition_id']:03d})",
                    f"Type: {next_type.upper()}",
                    f"Frequency: {float(next_stim['freq']):.2f} Hz",
                    *_stimulus_modulation_lines(next_type, next_kind, next_depth),
                ]
                if is_long_break:
                    body_lines.append("Longer break. Please relax your eyes and stand up for a moment.")
                if _run_countdown_screen(
                    win, info_stim,
                    title="Long Break" if is_long_break else "Short Break",
                    body_lines=body_lines,
                    duration_s=pause_duration_s,
                    countdown_label="Next block starts in",
                    ready_text="Press SPACE to continue.",
                ):
                    aborted = True
                    break

        protocol["aborted"] = aborted
        protocol["status"] = "aborted" if aborted else "completed"
        protocol["datetime_end"] = now_iso()

        if acquisition_mode == "record":
            info_stim.text = "Exporting EEG data ...\n\nPlease wait."
            info_stim.draw()
            win.flip()
            if recording is None:
                raise RuntimeError("Recording controller is not available.")
            export_ok = recording.stop_and_export(
                export_folder=paths["folder"],
                data_types=["EEG", "MOTION", "PM", "BP"],
                export_format="CSV",
                version="V2",
                timeout=300.0,
            )
            protocol["export_ok"] = bool(export_ok)
            logger.info("EEG export | ok=%s", export_ok)
        else:
            if live is not None:
                live.stop()
                protocol["live_summary"] = live.summary()
                logger.info("Live stream stopped | summary=%s", protocol["live_summary"])

        snapshot_protocol()

        if aborted:
            show_message(win, info_stim, f"{PHASE_LABEL} stopped.\n\n{global_trial}/{total_trials} trials finished.\n\nPress SPACE.")
        else:
            show_message(win, info_stim, f"{PHASE_LABEL} finished.\n\n{total_trials} trials recorded.\n\nPress SPACE.")

        if acquisition_mode == "record":
            save_session_outputs(
                protocol=protocol,
                trial_rows=trial_rows,
                event_rows=event_rows,
                paths=paths,
                marker=marker,
                rating_rows=rating_rows,
                trial_fields=TRIAL_FIELDS,
                rating_fields=RATING_FIELDS,
            )

    except Exception:
        logger.exception("Critical error in %s", PHASE_LABEL)
        protocol["aborted"] = True
        protocol["status"] = "error"
        protocol["datetime_end"] = now_iso()
        try:
            if live is not None:
                live.stop()
                protocol["live_summary"] = live.summary()
        except Exception:
            logger.exception("Live stream stop failed during exception handling")
        try:
            snapshot_protocol()
        except Exception:
            pass
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
    print("Stimulus screening module demo")
    print("This demo does not open a PsychoPy window and does not start Cortex recording.")
    print(f"Phase: {PHASE_LABEL} ({PHASE_SHORT})")
    print(f"Rating items: {len(COMFORT_ITEMS)}")
    print(f"Stimulus family order: {', '.join(STIMULUS_FAMILY_ORDER)}")



from __future__ import annotations

"""
main_study.py
=====================
End-to-end runner for the main SSVEP study (post-pilot data collection).

The session presents four stimuli simultaneously at 8.57 / 12 / 15 / 20 Hz and
cues one of them per trial; a classifier (evaluated offline) must decide which
of the four frequencies the proband is looking at. A final center-fixation
idle block provides a null-class reference.

Control flow (matches the approved plan):
    1. Runtime parameter GUI (prefilled from .env + session JSON, editable).
    2. Pre-flight compliance checklist.
    3. Metadata dialogs: Q1 trait (skipped if already collected), Q2 state, Q3 environment.
    4. Cortex session start (record or live-stream mode).
    5. Impedance gate (5 s stable, all channels >= threshold).
    6. Baseline eyes-open (+ optional eyes-closed).
    7. Block loop (B1..B7): multi-target block -> comfort ratings ->
       countdown rest (60 s or 180 s) -> 3 s impedance gate.
    8. Idle block (B8) with 5 s center-fixation idle trials and no visual gaps.
    9. End-of-session comfort ranking.
    10. Write protocol.json + CSVs (trials, events, ratings).

Reuses:
    - StimulusGenerator.run_multi_target_trial          (stimulus_generator.py)
    - run_impedance_gate                                (acquisition/impedance.py)
    - questionnaires.main_study.*                       (Q1/Q2/Q3/post)
    - core.conditions.multi_target_cue_marker / idle_trial_start_marker
    - experiments.stimulus_screening.RatingDisplay
    - core.session and data_io.writers
"""

import copy
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __name__ == "__main__":  # pragma: no cover - supports direct module demos
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.conditions import (
    MULTI_TARGET_CUE_MARKER_BASE,
    MULTI_TARGET_MAX_TARGETS,
    CONDITION_MARKER_START_BASE,
    CONDITION_MARKER_STOP_BASE,
    idle_trial_marker_label,
    idle_trial_start_marker,
    idle_trial_stop_marker,
    multi_target_cue_label,
    multi_target_cue_marker,
    resolve_active_modulation,
)
from core.config import (
    DEFAULT_ACQUISITION_MODE,
    DEFAULT_LIVE_FLUSH_EVERY,
    DEFAULT_LIVE_STREAMS,
    DEFAULT_MS_BACKGROUND_COLOR,
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
    DEFAULT_MS_INTERFACE_LANGUAGE,
    DEFAULT_MS_ITI_RANGE_S,
    DEFAULT_MS_LONG_BREAK_AFTER_BLOCKS,
    DEFAULT_MS_LONG_BREAK_S,
    DEFAULT_MS_PROBAND_SCHEMA,
    DEFAULT_MS_SEED_BASE,
    DEFAULT_MS_SESSION_CONFIG,
    DEFAULT_MS_STIM_OFF_COLOR,
    DEFAULT_MS_STIM_ON_COLOR,
    DEFAULT_MS_STIMULI_JSON,
    DEFAULT_MS_TARGET_SPACING_DEG,
    DEFAULT_MS_TARGET_SIZE_DEG,
    DEFAULT_MS_TRIAL_DURATION_S,
    DEFAULT_MS_TRIALS_PER_TARGET,
    DEFAULT_OUT_DIR,
    DEFAULT_PROBAND_ID,
    DEFAULT_SCREEN_ID,
    DEFAULT_FULLSCREEN,
    DEFAULT_HEADSET_ID,
    MONITOR_REFRESH_RATE,
    PROJECT_ROOT,
    target_positions_from_spacing,
)
from core.session import (
    build_cortex_record_title,
    build_output_paths,
    build_run_timestamp,
    determine_next_session_id,
    now_iso,
)
from core.logging_utils import setup_logging
from data_io.writers import (
    append_csv_rows,
    safe_write_json,
    write_csv,
)
from questionnaires.main_study import (
    check_exclusion_criteria,
    load_proband_schema,
    normalize_interface_language,
    run_compliance_checklist,
    run_questionnaire_section,
    run_runtime_parameter_dialog,
)

logger = logging.getLogger(__name__)


PHASE_SHORT = "mainstudy"
PHASE_LABEL = "Main Study (4-target SSVEP)"
MAIN_STUDY_BLOCK_START_BASE = 500
MAIN_STUDY_BLOCK_STOP_BASE = 550


# =============================================================================
# CSV field definitions
# =============================================================================

TRIAL_FIELDS = [
    "session_id", "proband_id", "phase",
    "trial_idx_global", "block_idx", "trial_idx_in_block",
    "block_id", "dataset_id", "stimulus_id", "stimulus_name", "stimulus_type", "stimulus_shape",
    "luminance_mod_kind", "luminance_depth", "contrast_percent",
    "cue_target_idx", "cue_position_label", "cue_position_deg_x", "cue_position_deg_y",
    "cue_frequency_hz",
    "frequencies_hz", "positions_deg",
    "fix_duration_s", "cue_duration_s", "iti_duration_s",
    "nominal_duration_s", "actual_duration_s",
    "cue_marker_value", "cue_marker_label",
    "start_marker", "stop_marker",
    "t_cue_monotonic", "t_onset_monotonic", "t_offset_monotonic",
    "t_cue_marker_ms", "t_onset_marker_ms", "t_offset_marker_ms",
]

IDLE_TRIAL_FIELDS = [
    "session_id", "proband_id", "phase",
    "trial_idx_in_idle", "trial_idx_in_idle_source",
    "idle_block_idx", "idle_block_id", "dataset_id",
    "source_block_idx", "source_block_id",
    "angle_idx", "angle_deg", "idle_gaze_id", "idle_gaze_label",
    "stimulus_id", "stimulus_name", "stimulus_shape",
    "luminance_mod_kind", "luminance_depth", "contrast_percent",
    "frequencies_hz", "positions_deg",
    "fix_duration_s", "iti_duration_s",
    "nominal_duration_s", "actual_duration_s",
    "start_marker", "start_marker_label", "stop_marker", "stop_marker_label",
    "t_onset_monotonic", "t_offset_monotonic",
    "t_onset_marker_ms", "t_offset_marker_ms",
]

RATING_FIELDS = [
    "session_id", "proband_id", "phase",
    "block_idx", "block_id", "stimulus_id", "stimulus_name",
    "contrast_percent",
    "item_idx", "item_id", "item_label", "item_construct",
    "rating", "scale_min", "scale_max", "timestamp_iso",
]

RANKING_FIELDS = [
    "session_id", "proband_id", "phase",
    "rank", "letter", "block_idx", "block_id", "dataset_id",
    "stimulus_id", "stimulus_name", "stimulus_shape",
    "luminance_mod_kind", "luminance_depth", "contrast_percent",
    "timestamp_iso",
]

EVENT_FIELDS = [
    "event_type", "session_id", "proband_id", "phase",
    "block_idx", "block_id", "trial_idx_global", "after_block_idx",
    "stimulus_id", "stimulus_name", "stimulus_shape", "contrast_percent",
    "angle_idx", "angle_deg",
    "break_type", "planned_duration_s", "actual_duration_s",
    "break_anchor",
    "impedance_mode", "impedance_min_quality", "impedance_final_quality",
    "impedance_passed", "impedance_forced", "impedance_timeout", "impedance_aborted",
    "marker_value", "marker_label", "t_marker_ms", "t_monotonic",
    "timestamp_iso",
]


# =============================================================================
# Utility helpers
# =============================================================================

def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return Path(PROJECT_ROOT) / p


def _load_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def _xy_label(pos: Tuple[float, float]) -> str:
    x, y = pos
    if abs(x) > abs(y):
        return "left" if x < 0 else "right"
    return "bottom" if y < 0 else "top"


def _is_standard_cross_layout(
    positions: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Return True when positions are just the standard left/top/right/bottom cross."""

    if len(positions) != MULTI_TARGET_MAX_TARGETS:
        return False
    try:
        pts = [(float(pos[0]), float(pos[1])) for pos in positions]
    except (TypeError, ValueError, IndexError):
        return False
    axis_distances = [abs(v) for point in pts for v in point if abs(v) > tolerance]
    if not axis_distances:
        return False
    spacing = axis_distances[0]
    expected = set(target_positions_from_spacing(spacing))
    rounded_pts = {(round(x, 6), round(y, 6)) for x, y in pts}
    rounded_expected = {(round(x, 6), round(y, 6)) for x, y in expected}
    return rounded_pts == rounded_expected


def _validate_range_pair(name: str, values: Sequence[Any]) -> Tuple[float, float]:
    """Validate a [min, max] duration range."""

    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low = float(values[0])
    high = float(values[1])
    if low < 0.0 or high < low:
        raise ValueError(f"{name} must be non-negative and ordered.")
    return low, high


def _build_runtime_defaults(
    session_cfg: Dict[str, Any],
    cli_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge .env defaults + session JSON into the dialog prefill dict."""
    n_freqs = len(session_cfg.get("frequencies_hz", DEFAULT_MS_FREQUENCIES_HZ))
    if "trials_per_target" in session_cfg:
        trials_per_target_default = int(session_cfg["trials_per_target"])
    elif "trials_per_block" in session_cfg:
        trials_per_target_default = max(1, int(session_cfg["trials_per_block"]) // max(1, n_freqs))
    else:
        trials_per_target_default = 5
    defaults = {
        "proband_id": str(cli_cfg.get("proband_id", DEFAULT_PROBAND_ID)),
        "interface_language": str(
            cli_cfg.get(
                "interface_language",
                session_cfg.get("interface_language", DEFAULT_MS_INTERFACE_LANGUAGE),
            )
        ),
        "acquisition_mode": str(cli_cfg.get("acquisition_mode", DEFAULT_ACQUISITION_MODE)),
        "trials_per_target": trials_per_target_default,
        "trial_duration_s": float(session_cfg.get("trial_duration_s", DEFAULT_MS_TRIAL_DURATION_S)),
        "cue_duration_s": float(session_cfg.get("cue_duration_s", DEFAULT_MS_CUE_DURATION_S)),
        "target_spacing_deg": float(
            cli_cfg.get(
                "target_spacing_deg",
                session_cfg.get("target_spacing_deg", DEFAULT_MS_TARGET_SPACING_DEG),
            )
        ),
        "target_size_deg": float(session_cfg.get("target_size_deg", DEFAULT_MS_TARGET_SIZE_DEG)),
        "baseline_open_s": float(session_cfg.get("baseline_open_s", DEFAULT_MS_BASELINE_OPEN_S)),
        "include_eyes_closed_baseline": bool(
            session_cfg.get("include_eyes_closed_baseline", DEFAULT_MS_INCLUDE_EYES_CLOSED_BASELINE)
        ),
        "baseline_closed_s": float(session_cfg.get("baseline_closed_s", DEFAULT_MS_BASELINE_CLOSED_S)),
        "block_break_s": float(session_cfg.get("block_break_s", DEFAULT_MS_BLOCK_BREAK_S)),
        "long_break_s": float(session_cfg.get("long_break_s", DEFAULT_MS_LONG_BREAK_S)),
        "impedance_stable_s_initial": float(
            session_cfg.get("impedance_stable_s_initial", DEFAULT_MS_IMPEDANCE_STABLE_S_INITIAL)
        ),
        "impedance_stable_s_between": float(
            session_cfg.get("impedance_stable_s_between", DEFAULT_MS_IMPEDANCE_STABLE_S_BETWEEN)
        ),
        "impedance_quality_threshold": int(
            session_cfg.get("impedance_quality_threshold", DEFAULT_MS_IMPEDANCE_QUALITY_THRESHOLD)
        ),
        "seed_base": int(session_cfg.get("seed_base", DEFAULT_MS_SEED_BASE)),
    }
    if any(block.get("stimulus_id") is None for block in session_cfg.get("block_order") or []):
        defaults["idle_trials_per_source_block"] = int(
            session_cfg.get("idle_trials_per_source_block", DEFAULT_MS_IDLE_TRIALS_PER_SOURCE_BLOCK)
        )
    return defaults


def _assign_frequency_mapping(
    frequencies_hz: Sequence[float],
    positions_deg: Sequence[Tuple[float, float]],
    proband_id: str,
) -> Dict[int, int]:
    """Return `{target_idx -> position_idx}` deterministic per proband.

    Target indices are identified by their frequency order in
    `frequencies_hz`; each target is assigned exactly one position slot so
    that between-subject position/frequency counterbalancing is preserved.
    """
    rng = random.Random(f"freq_pos::{proband_id}")
    n = len(frequencies_hz)
    if n != len(positions_deg):
        raise ValueError(
            f"frequencies_hz ({n}) and positions_deg ({len(positions_deg)}) must be equal length."
        )
    slot_order = list(range(n))
    rng.shuffle(slot_order)
    return {t_idx: slot_order[t_idx] for t_idx in range(n)}


def _build_cue_sequence(
    *,
    trials_per_target: int,
    n_targets: int,
    seed: int,
    fixed_target_idx: Optional[int] = None,
) -> List[int]:
    """Balanced constrained shuffle.

    Each target is cued exactly `trials_per_target` times. The complete set of
    cue labels is shuffled as one balanced pool, so the cue order is not a
    fixed round-robin sequence. Direct repetitions are avoided when possible.

    If `fixed_target_idx` is given, all trials are cued to that single target
    (used for single-frequency validation sessions).
    """
    if trials_per_target <= 0:
        raise ValueError("trials_per_target must be >= 1")
    if n_targets <= 0:
        raise ValueError("n_targets must be >= 1")
    if fixed_target_idx is not None:
        return [fixed_target_idx] * (trials_per_target * n_targets)
    rng = random.Random(seed)
    pool = [target_idx for target_idx in range(n_targets) for _ in range(trials_per_target)]
    best_sequence = list(pool)
    best_repeats = len(pool)
    for _ in range(1000):
        candidate = list(pool)
        rng.shuffle(candidate)
        repeats = sum(1 for a, b in zip(candidate, candidate[1:]) if a == b)
        if repeats == 0:
            return candidate
        if repeats < best_repeats:
            best_sequence = candidate
            best_repeats = repeats
    return best_sequence


def _participant_order_index(proband_id: str, n_sequences: int) -> int:
    """Return a stable zero-based sequence index for a participant ID.

    Numeric IDs use their last number, so P01 maps to sequence 0, P02 to 1,
    etc. Non-numeric IDs fall back to a deterministic character checksum.
    """

    if n_sequences <= 0:
        raise ValueError("n_sequences must be >= 1")
    text = str(proband_id or "").strip()
    matches = re.findall(r"\d+", text)
    if matches:
        value = int(matches[-1])
        return max(0, value - 1) % n_sequences
    checksum = sum((idx + 1) * ord(char) for idx, char in enumerate(text))
    return checksum % n_sequences


def _resolve_active_block_order(
    stim_blocks: Sequence[Dict[str, Any]],
    *,
    proband_id: str,
    strategy: str,
    seed_base: int,
    lock_tail_count: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return the active block order used for this participant.

    ``balanced_rotation_by_proband`` is the default for the main study. It is
    not a full carry-over-balanced Latin square, but it removes the strongest
    fatigue confound by placing each stimulus in each serial position once
    across seven consecutive numeric participant IDs.
    """

    canonical: List[Dict[str, Any]] = []
    for idx, block in enumerate(stim_blocks, start=1):
        block_copy = copy.deepcopy(block)
        block_copy.setdefault("canonical_order_idx", idx)
        canonical.append(block_copy)
    if not canonical:
        return [], {
            "strategy": "empty",
            "sequence_index": 0,
            "canonical_block_ids": [],
            "presented_block_ids": [],
        }

    strategy_norm = str(strategy or "fixed").strip().lower()
    locked_tail_n = max(0, min(int(lock_tail_count), len(canonical)))
    unlocked = canonical[: len(canonical) - locked_tail_n] if locked_tail_n else list(canonical)
    locked_tail = canonical[len(canonical) - locked_tail_n:] if locked_tail_n else []

    if strategy_norm in {"fixed", "as_configured", "none"}:
        ordered = canonical
        sequence_index = 0
    elif strategy_norm in {"balanced_rotation_by_proband", "rotation_by_proband"}:
        sequence_index = _participant_order_index(proband_id, len(unlocked))
        ordered = unlocked[sequence_index:] + unlocked[:sequence_index] + locked_tail
    elif strategy_norm in {"shuffle_by_proband", "random_by_proband"}:
        sequence_index = _participant_order_index(proband_id, len(canonical))
        ordered = list(canonical)
        random.Random(f"block_order::{proband_id}::{seed_base}").shuffle(ordered)
    elif strategy_norm in {"shuffle_first_n_keep_tail", "shuffle_unlocked_keep_tail"}:
        sequence_index = _participant_order_index(proband_id, max(1, len(unlocked)))
        ordered = list(unlocked)
        random.Random(f"block_order_tail_locked::{proband_id}::{seed_base}").shuffle(ordered)
        ordered.extend(locked_tail)
    else:
        raise ValueError(
            "Unsupported block_order_strategy. Use fixed, "
            "balanced_rotation_by_proband, shuffle_by_proband, "
            "or shuffle_first_n_keep_tail."
        )

    ordered_with_position: List[Dict[str, Any]] = []
    for presentation_idx, block in enumerate(ordered, start=1):
        block_copy = copy.deepcopy(block)
        block_copy["presentation_order_idx"] = presentation_idx
        ordered_with_position.append(block_copy)

    meta = {
        "strategy": strategy_norm,
        "sequence_index": int(sequence_index),
        "canonical_block_ids": [
            str(block.get("block_id", f"B{idx}"))
            for idx, block in enumerate(canonical, start=1)
        ],
        "presented_block_ids": [
            str(block.get("block_id", f"B{idx}"))
            for idx, block in enumerate(ordered_with_position, start=1)
        ],
        "presented_stimulus_ids": [
            int(block["stimulus_id"])
            for block in ordered_with_position
            if block.get("stimulus_id") is not None
        ],
        "locked_tail_count": locked_tail_n,
        "note": (
            "This controls serial-position fatigue/order effects. The original "
            "block_id remains the condition ID; block_idx is the actual "
            "presentation position."
        ),
    }
    return ordered_with_position, meta


def _build_refresh_frequency_audit(
    frequencies_hz: Sequence[float],
    *,
    refresh_hz: float,
    trial_duration_s: float,
) -> Dict[str, Any]:
    """Document monitor/frequency constraints for the fixed 60 Hz setup."""

    rows: List[Dict[str, Any]] = []
    for freq in frequencies_hz:
        f = float(freq)
        period_frames = float(refresh_hz) / f
        nearest_period_frames = max(1, int(round(period_frames)))
        realized_hz_if_integer_period = float(refresh_hz) / nearest_period_frames
        cycles_in_trial = f * float(trial_duration_s)
        square_cycle_balance = (
            "balanced_even_frame_cycle"
            if nearest_period_frames % 2 == 0
            else "uneven_square_duty_possible"
        )
        rows.append({
            "frequency_hz": round(f, 6),
            "period_frames_at_refresh": round(period_frames, 6),
            "nearest_integer_period_frames": nearest_period_frames,
            "integer_period_error_frames": round(period_frames - nearest_period_frames, 6),
            "realized_hz_if_integer_period": round(realized_hz_if_integer_period, 6),
            "cycles_in_trial": round(cycles_in_trial, 6),
            "integer_cycles_in_trial": abs(cycles_in_trial - round(cycles_in_trial)) < 0.01,
            "square_cycle_balance": square_cycle_balance,
        })
    return {
        "refresh_hz": round(float(refresh_hz), 6),
        "trial_duration_s": round(float(trial_duration_s), 6),
        "frequencies_hz": [round(float(f), 6) for f in frequencies_hz],
        "fixed_design_note": (
            "Frequencies are intentionally fixed for this BA. On a 60 Hz monitor "
            "the waveform is sampled once per video frame. This is acceptable for "
            "the planned empirical comparison, but the thesis should describe it "
            "as a hardware constraint, especially for square-wave duty balance."
        ),
        "per_frequency": rows,
    }


def _build_idle_sequence(
    *,
    angles_deg: Sequence[float],
    trials_per_angle: int,
    seed: int,
) -> List[int]:
    rng = random.Random(seed)
    sequence: List[int] = []
    for angle_idx in range(len(angles_deg)):
        sequence.extend([angle_idx] * int(trials_per_angle))
    rng.shuffle(sequence)
    return sequence


def _ui(language: str, english: str, german: str) -> str:
    return german if normalize_interface_language(language) == "de" else english


def _load_idle_gaze_targets(session_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_targets = session_cfg.get("idle_gaze_targets")
    targets: List[Dict[str, Any]] = []
    if isinstance(raw_targets, list) and raw_targets:
        for idx, raw in enumerate(raw_targets):
            if not isinstance(raw, dict):
                continue
            cue_pos = raw.get("cue_pos_deg", [0.0, 0.0])
            if not isinstance(cue_pos, (list, tuple)) or len(cue_pos) != 2:
                cue_pos = [0.0, 0.0]
            targets.append({
                "idx": idx,
                "id": str(raw.get("id", f"gaze_{idx + 1}")),
                "label": str(raw.get("label", f"gaze target {idx + 1}")),
                "label_de": str(raw.get("label_de", raw.get("label", f"Blickziel {idx + 1}"))),
                "cue_pos_deg": [float(cue_pos[0]), float(cue_pos[1])],
                "angle_deg": float(raw.get("angle_deg", 0.0)),
            })
    if targets:
        return targets

    angles = [float(a) for a in session_cfg.get("idle_angles_deg", DEFAULT_MS_IDLE_ANGLES_DEG)]
    for idx, angle_deg in enumerate(angles):
        targets.append({
            "idx": idx,
            "id": f"angle_{idx + 1}",
            "label": f"outside gaze angle {angle_deg:+.1f} deg",
            "label_de": f"Blickwinkel ausserhalb {angle_deg:+.1f} Grad",
            "cue_pos_deg": [angle_deg * 0.3, 0.0],
            "angle_deg": angle_deg,
        })
    return targets


def _build_idle_gaze_sequence(
    *,
    n_targets: int,
    trials_per_source_block: int,
    seed: int,
) -> List[int]:
    if n_targets <= 0:
        raise ValueError("Idle block requires at least one gaze target.")
    if trials_per_source_block <= 0:
        raise ValueError("idle_trials_per_source_block must be >= 1.")
    rng = random.Random(seed)
    sequence: List[int] = []
    while len(sequence) < trials_per_source_block:
        round_targets = list(range(n_targets))
        rng.shuffle(round_targets)
        sequence.extend(round_targets)
    return sequence[:trials_per_source_block]


def _load_idle_target_positions(
    session_cfg: Dict[str, Any],
    *,
    spacing_deg: float,
) -> List[Tuple[float, float]]:
    raw_positions = session_cfg.get("idle_target_positions_deg")
    if isinstance(raw_positions, list) and len(raw_positions) == MULTI_TARGET_MAX_TARGETS:
        return [(float(pos[0]), float(pos[1])) for pos in raw_positions]
    spacing = float(spacing_deg)
    return [
        (-spacing, spacing),
        (spacing, spacing),
        (-spacing, -spacing),
        (spacing, -spacing),
    ]


def _load_target_positions(
    session_cfg: Dict[str, Any],
    *,
    spacing_deg: float,
) -> List[Tuple[float, float]]:
    raw_positions = session_cfg.get("target_positions_deg")
    if isinstance(raw_positions, list) and len(raw_positions) == MULTI_TARGET_MAX_TARGETS:
        if _is_standard_cross_layout(raw_positions):
            return [(float(pos[0]), float(pos[1])) for pos in target_positions_from_spacing(spacing_deg)]
        return [(float(pos[0]), float(pos[1])) for pos in raw_positions]
    return [(float(pos[0]), float(pos[1])) for pos in target_positions_from_spacing(spacing_deg)]


def _main_study_block_marker(block_idx: int, *, role: str) -> int:
    """Return a dedicated marker value for block boundaries.

    Block markers must stay disjoint from trial markers, cue markers,
    and idle-trial markers so that offline parsing can recover the event
    hierarchy from the numeric marker stream alone.
    """

    if block_idx < 1:
        raise ValueError("block_idx must be >= 1")
    role_normalized = str(role).strip().lower()
    if role_normalized == "start":
        return MAIN_STUDY_BLOCK_START_BASE + block_idx
    if role_normalized == "stop":
        return MAIN_STUDY_BLOCK_STOP_BASE + block_idx
    raise ValueError(f"Unsupported block marker role: {role}")


def _compute_remaining_break_s(
    *,
    break_total_s: float,
    block_end_monotonic: float,
    now_monotonic: Optional[float] = None,
) -> Tuple[float, float]:
    """Return elapsed break-budget time and the remaining visible countdown.

    The protocol defines the break budget relative to the *block end*,
    not relative to the moment when the post-block rating finishes.
    """

    now = time.monotonic() if now_monotonic is None else float(now_monotonic)
    elapsed_since_block_end = max(0.0, now - float(block_end_monotonic))
    remaining_s = max(0.0, float(break_total_s) - elapsed_since_block_end)
    return elapsed_since_block_end, remaining_s


def _estimate_session_duration_min(
    *,
    baseline_open_s: float,
    baseline_closed_s: float,
    include_eyes_closed_baseline: bool,
    active_block_count: int,
    trials_per_block: int,
    cue_duration_s: float,
    trial_duration_s: float,
    fix_range_s: Sequence[float],
    iti_range_s: Sequence[float],
    block_break_s: float,
    long_break_s: float,
    long_break_after_blocks: Sequence[int],
    idle_block_count: int,
    idle_trials_total: int,
    idle_post_stop_hold_s: float,
    impedance_stable_s_initial: float,
    impedance_stable_s_between: float,
    rating_s_per_block: float = 60.0,
    ranking_s: float = 180.0,
    safety_buffer_min: float = 5.0,
) -> float:
    avg_fix_s = sum(float(v) for v in fix_range_s) / max(1, len(fix_range_s))
    avg_iti_s = sum(float(v) for v in iti_range_s) / max(1, len(iti_range_s))
    long_break_set = {int(v) for v in long_break_after_blocks}
    break_total_s = 0.0
    between_block_impedance_s = 0.0
    for block_idx in range(1, max(0, int(active_block_count))):
        break_total_s += float(long_break_s if block_idx in long_break_set else block_break_s)
        between_block_impedance_s += float(impedance_stable_s_between)

    total_seconds = (
        float(baseline_open_s)
        + (float(baseline_closed_s) if include_eyes_closed_baseline else 0.0)
        + float(impedance_stable_s_initial)
        + int(active_block_count) * int(trials_per_block) * (float(cue_duration_s) + float(trial_duration_s) + avg_fix_s + avg_iti_s)
        + int(idle_trials_total) * float(trial_duration_s)
        + max(0, int(idle_block_count)) * float(idle_post_stop_hold_s)
        + break_total_s
        + between_block_impedance_s
        + int(active_block_count) * float(rating_s_per_block)
        + float(ranking_s)
        + float(safety_buffer_min) * 60.0
    )
    return total_seconds / 60.0


def _countdown_remaining(
    *,
    win: Any,
    info_stim: Any,
    title: str,
    body_lines: Sequence[str],
    remaining_s: float,
    countdown_label: str = "Break ends in",
    ready_text: str = "Press SPACE to continue.",
) -> bool:
    """Display one info screen, then wait for SPACE after the countdown."""
    from psychopy import core, event  # type: ignore

    remaining_s = max(0.0, float(remaining_s))
    deadline = time.monotonic() + remaining_s
    event.clearEvents(eventType="keyboard")
    while True:
        now = time.monotonic()
        left = max(0.0, deadline - now)
        disp = int(math.ceil(left))
        footer = f"{countdown_label}: {disp} s" if left > 0.0 else ready_text
        info_stim.text = "\n\n".join([title, *body_lines, footer])
        info_stim.draw()
        win.flip()
        keys = event.getKeys(keyList=["escape", "space"])
        if "escape" in keys:
            return True
        if left <= 0.0:
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


def _short_stimulus_name(name: str) -> str:
    """Return a compact display name without parenthetical parameter text."""

    return str(name).split("(", 1)[0].strip()


def _stimulus_display_name(name: str, modulation: str, contrast_percent: float, language: str) -> str:
    """Return a compact stimulus label with the relevant modulation settings."""

    return f"{_short_stimulus_name(name)} ({modulation}, {contrast_percent:.0f}%)"


def _contrast_percent(stimulus_cfg: Dict[str, Any], session_cfg: Dict[str, Any]) -> float:
    """Return spatial contrast in percent, defaulting to the planned 100%."""

    raw = stimulus_cfg.get("contrast_percent", session_cfg.get("contrast_percent", 100.0))
    if raw is None and "contrast" in stimulus_cfg:
        raw = stimulus_cfg["contrast"]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 100.0
    if 0.0 <= value <= 1.0:
        value *= 100.0
    return round(value, 3)


def _ranking_positions_deg() -> List[Tuple[float, float]]:
    """Seven compact preview positions for the end-of-session ranking."""

    return [
        (-9.0, 4.8),
        (-4.5, 4.8),
        (0.0, 4.8),
        (4.5, 4.8),
        (-6.75, -3.0),
        (-2.25, -3.0),
        (2.25, -3.0),
    ]


def _ranking_letter(index: int) -> str:
    return chr(ord("A") + int(index))


def _parse_ranking_text(text: str, valid_letters: Sequence[str]) -> Optional[List[str]]:
    cleaned = [char.upper() for char in str(text) if char.isalpha()]
    valid = set(valid_letters)
    if len(cleaned) != len(valid_letters):
        return None
    if set(cleaned) != valid:
        return None
    return cleaned


def _collect_end_ranking(
    *,
    win: Any,
    info_stim: Any,
    stim_blocks: Sequence[Dict[str, Any]],
    stimuli_by_id: Dict[int, Dict[str, Any]],
    session_cfg: Dict[str, Any],
    generator_cls: Any,
    base_generator: Any,
    frequencies_hz: Sequence[float],
    refresh_hz: float,
    target_size_deg: float,
    background_color: Sequence[float],
    stim_on_color: Sequence[float],
    stim_off_color: Sequence[float],
    interface_language: str,
    session_id: str,
    proband_id: str,
    phase_short: str,
) -> Optional[List[Dict[str, Any]]]:
    """Show active previews and collect a comfort ranking for blocks 1..7."""

    from psychopy import core, event, visual  # type: ignore
    from stimuli.generator import normalize_stimulus_cfg  # type: ignore

    letters = [_ranking_letter(i) for i in range(len(stim_blocks))]
    positions = _ranking_positions_deg()
    preview_freq_hz = 12.0 if 12.0 in [float(f) for f in frequencies_hz] else float(frequencies_hz[0])
    preview_size = min(float(target_size_deg), 3.2)

    preview_generators = []
    updaters = []
    label_stims = []
    letter_to_meta: Dict[str, Dict[str, Any]] = {}

    for idx, block_spec in enumerate(stim_blocks):
        letter = letters[idx]
        stim_id = int(block_spec["stimulus_id"])
        stim_cfg = normalize_stimulus_cfg(stimuli_by_id[stim_id])
        stim_type = str(stim_cfg.get("type", "ssvep")).lower()
        lum_kind, lum_depth = resolve_active_modulation(
            stim_type,
            str(stim_cfg.get("luminance_mod_kind", "sine")),
            float(stim_cfg.get("luminance_depth", 1.0)),
        )
        preview_gen = generator_cls(
            win,
            monitor_refresh_rate=refresh_hz,
            luminance_depth=lum_depth,
            luminance_mod_kind=lum_kind,
            background_color=stim_cfg.get("background_color") or background_color,
            stim_on_color=stim_cfg.get("stim_on_color") or stim_on_color,
            stim_off_color=stim_cfg.get("stim_off_color") or stim_off_color,
            abort_check=base_generator.abort_check,
        )
        preview_generators.append(preview_gen)
        cfg = dict(stim_cfg)
        cfg["size_deg"] = preview_size
        updaters.append(preview_gen._build_multi_target_updater(cfg, preview_freq_hz, positions[idx]))
        label_stims.append(
            visual.TextStim(
                win,
                text=f"{letter}\nB{idx + 1}",
                pos=(positions[idx][0], positions[idx][1] - 2.3),
                units="deg",
                height=0.42,
                color="white",
                alignText="center",
            )
        )
        letter_to_meta[letter] = {
            "block_idx": idx + 1,
            "block_id": str(block_spec.get("block_id", f"B{idx + 1}")),
            "dataset_id": str(block_spec.get("dataset_id", f"DS{idx + 1}")),
            "stimulus_id": stim_id,
            "stimulus_name": str(stim_cfg.get("name", "")),
            "stimulus_shape": str(stim_cfg.get("shape", "")),
            "luminance_mod_kind": lum_kind,
            "luminance_depth": lum_depth,
            "contrast_percent": _contrast_percent(stim_cfg, session_cfg),
        }

    prompt = visual.TextStim(
        win,
        units="height",
        color="white",
        height=0.035,
        pos=(0.0, -0.44),
        wrapWidth=1.65,
        alignText="center",
    )
    typed = ""
    event.clearEvents(eventType="keyboard")
    start = core.getTime()

    while True:
        elapsed = max(0.0, core.getTime() - start)
        frame_idx = elapsed * float(refresh_hz)
        for update in updaters:
            update(frame_idx)
        for label in label_stims:
            label.draw()

        prompt.text = _ui(
            interface_language,
            (
                "Comfort ranking: type the letters from most comfortable to least comfortable, then press ENTER.\n"
                f"Example: ACBDEFG   Current input: {typed or '-'}"
            ),
            (
                "Komfort-Ranking: Buchstaben von angenehm bis unangenehm eingeben, dann ENTER drücken.\n"
                f"Beispiel: ACBDEFG   Aktuelle Eingabe: {typed or '-'}"
            ),
        )
        prompt.draw()
        win.flip()

        keys = event.getKeys()
        for key in keys:
            if key == "escape":
                return None
            if key in {"backspace", "delete"}:
                typed = typed[:-1]
                continue
            if key in {"return", "num_enter"}:
                ranking = _parse_ranking_text(typed, letters)
                if ranking is None:
                    typed = ""
                    prompt.text = _ui(
                        interface_language,
                        "Invalid ranking. Use every letter A-G exactly once.",
                        "Ungültige Reihenfolge. Bitte jeden Buchstaben A-G genau einmal nutzen.",
                    )
                    prompt.draw()
                    win.flip()
                    core.wait(1.2)
                    break
                rows = []
                for rank, letter in enumerate(ranking, start=1):
                    row = dict(letter_to_meta[letter])
                    row.update({
                        "session_id": session_id,
                        "proband_id": proband_id,
                        "phase": phase_short,
                        "rank": rank,
                        "letter": letter,
                        "timestamp_iso": now_iso(),
                    })
                    rows.append(row)
                return rows
            if len(key) == 1 and key.upper() in letters and len(typed) < len(letters):
                typed += key.upper()


# =============================================================================
# Runner
# =============================================================================

def run_main_study_session(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full main-study session.

    Returns
    -------
    dict
        The final protocol dict (same object that was written to disk).
    """
    # ---- load session JSON + merge defaults -------------------------------
    session_cfg_data = cfg.get("session_config_data")
    if session_cfg_data is not None:
        if not isinstance(session_cfg_data, dict):
            raise TypeError("session_config_data must be a dict.")
        session_cfg_path = Path(str(cfg.get("session_config_label", "<in-memory session config>")))
        session_cfg = dict(session_cfg_data)
    else:
        session_cfg_path = _resolve_path(str(cfg.get("session_config", DEFAULT_MS_SESSION_CONFIG)))
        session_cfg = _load_json_object(session_cfg_path)
    phase_short = str(session_cfg.get("phase_short", cfg.get("phase_short", PHASE_SHORT)))
    phase_label = str(session_cfg.get("phase_label", cfg.get("phase_label", PHASE_LABEL)))

    stimuli_json_path = _resolve_path(
        str(cfg.get("stimuli_json") or session_cfg.get("stimuli_json", DEFAULT_MS_STIMULI_JSON))
    )
    proband_schema_path = _resolve_path(
        str(cfg.get("proband_schema") or session_cfg.get("proband_schema", DEFAULT_MS_PROBAND_SCHEMA))
    )
    schema = load_proband_schema(proband_schema_path)

    # ---- Step 1: runtime parameter GUI ------------------------------------
    dialog_defaults = _build_runtime_defaults(session_cfg, cfg)
    headless = bool(cfg.get("headless", False))
    skip_dialogs = bool(cfg.get("skip_dialogs", False))
    skip_compliance = bool(cfg.get("skip_compliance", False))
    if skip_dialogs:
        runtime_params = dict(dialog_defaults)
    else:
        runtime_params = run_runtime_parameter_dialog(
            dialog_defaults,
            title=f"{phase_label} - session parameters",
            headless=headless,
        )

    interface_language = normalize_interface_language(
        runtime_params.get("interface_language", session_cfg.get("interface_language", DEFAULT_MS_INTERFACE_LANGUAGE))
    )
    runtime_params["interface_language"] = interface_language

    proband_id = str(runtime_params.get("proband_id") or DEFAULT_PROBAND_ID)
    acquisition_mode = str(runtime_params.get("acquisition_mode") or DEFAULT_ACQUISITION_MODE).lower()
    if acquisition_mode not in {"record", "stream"}:
        raise ValueError("acquisition_mode must be 'record' or 'stream'")

    trials_per_target = int(runtime_params["trials_per_target"])
    trials_per_block  = trials_per_target * len(
        session_cfg.get("frequencies_hz", DEFAULT_MS_FREQUENCIES_HZ)
    )
    trial_duration_s = float(runtime_params["trial_duration_s"])
    cue_duration_s = float(runtime_params["cue_duration_s"])
    target_spacing_deg = float(runtime_params["target_spacing_deg"])
    target_size_deg = float(runtime_params.get("target_size_deg", session_cfg.get("target_size_deg", DEFAULT_MS_TARGET_SIZE_DEG)))
    baseline_open_s = float(runtime_params["baseline_open_s"])
    include_eyes_closed_baseline = bool(runtime_params["include_eyes_closed_baseline"])
    baseline_closed_s = float(runtime_params["baseline_closed_s"])
    block_break_s = float(runtime_params["block_break_s"])
    long_break_s = float(runtime_params["long_break_s"])
    impedance_stable_s_initial = float(runtime_params["impedance_stable_s_initial"])
    impedance_stable_s_between = float(runtime_params["impedance_stable_s_between"])
    impedance_quality_threshold = int(runtime_params["impedance_quality_threshold"])
    seed_base = int(runtime_params["seed_base"])
    if trials_per_target <= 0:
        raise ValueError("trials_per_target must be >= 1")
    if trial_duration_s <= 0.0:
        raise ValueError("trial_duration_s must be > 0")
    if cue_duration_s <= 0.0:
        raise ValueError("cue_duration_s must be > 0")
    if target_spacing_deg <= 0.0:
        raise ValueError("target_spacing_deg must be > 0")
    if target_size_deg <= 0.0:
        raise ValueError("target_size_deg must be > 0")
    if baseline_open_s < 0.0 or baseline_closed_s < 0.0:
        raise ValueError("baseline durations must be >= 0")
    if block_break_s < 0.0 or long_break_s < 0.0:
        raise ValueError("break durations must be >= 0")
    if not 0 <= impedance_quality_threshold <= 4:
        raise ValueError("impedance_quality_threshold must be in [0, 4]")

    fix_range_s = _validate_range_pair("fix_range_s", session_cfg.get("fix_range_s", DEFAULT_MS_FIX_RANGE_S))
    iti_range_s = _validate_range_pair("iti_range_s", session_cfg.get("iti_range_s", DEFAULT_MS_ITI_RANGE_S))
    impedance_timeout_s = float(session_cfg.get("impedance_timeout_s", DEFAULT_MS_IMPEDANCE_TIMEOUT_S))
    long_break_after_blocks = tuple(
        int(v) for v in session_cfg.get("long_break_after_blocks", DEFAULT_MS_LONG_BREAK_AFTER_BLOCKS)
    )
    frequencies_hz = [float(f) for f in session_cfg.get("frequencies_hz", DEFAULT_MS_FREQUENCIES_HZ)]
    _fixed_cue_raw = session_cfg.get("fixed_cue_target_idx")
    fixed_cue_target_idx: Optional[int] = int(_fixed_cue_raw) if _fixed_cue_raw is not None else None
    target_positions_deg: List[Tuple[float, float]] = _load_target_positions(
        session_cfg,
        spacing_deg=target_spacing_deg,
    )
    target_position_labels = [
        str(lbl) for lbl in session_cfg.get(
            "target_position_labels", [_xy_label(p) for p in target_positions_deg]
        )
    ]
    session_cfg["target_spacing_deg"] = target_spacing_deg
    session_cfg["target_size_deg"] = target_size_deg
    if len(frequencies_hz) != MULTI_TARGET_MAX_TARGETS:
        raise ValueError(
            f"Expected exactly {MULTI_TARGET_MAX_TARGETS} frequencies, got {len(frequencies_hz)}"
        )
    if len(target_positions_deg) != MULTI_TARGET_MAX_TARGETS:
        raise ValueError(
            f"Expected exactly {MULTI_TARGET_MAX_TARGETS} target positions, got {len(target_positions_deg)}"
        )

    # ---- Step 2: pre-flight compliance ------------------------------------
    if skip_dialogs or skip_compliance:
        compliance_result = {
            "skipped": True,
            "items": {},
            "all_required_ok": True,
            "skip_reason": "skip_dialogs" if skip_dialogs else "skip_compliance",
        }
    else:
        compliance_result = run_compliance_checklist(schema, headless=headless, language=interface_language)
    if not compliance_result.get("all_required_ok", True):
        missing = compliance_result.get("missing_required", [])
        raise RuntimeError(
            "Pre-flight compliance not satisfied (each required item must be set to 'Yes'): "
            + ", ".join(missing or ["<unknown>"])
            + f"  [schema: {proband_schema_path}]"
        )

    # ---- Step 3: Q1/Q2/Q3 questionnaires ----------------------------------
    if skip_dialogs:
        q1_answers = cfg.get("q1_answers") or {}
        q2_answers = cfg.get("q2_answers") or {}
        q3_answers = cfg.get("q3_answers") or {}
    else:
        q1_answers = (
            cfg.get("q1_answers")
            or run_questionnaire_section(schema, "q1_trait", headless=headless, language=interface_language)
        )
        q2_answers = run_questionnaire_section(schema, "q2_state", headless=headless, language=interface_language)
        q3_answers = run_questionnaire_section(schema, "q3_environment", headless=headless, language=interface_language)

    exclusion_hits = check_exclusion_criteria(schema, q1_answers)
    if exclusion_hits:
        raise RuntimeError(
            "Hard exclusion criteria triggered: "
            + "; ".join(reason for _, reason in exclusion_hits)
        )

    # ---- Open output tree --------------------------------------------------
    out_dir = str(cfg.get("out_dir", DEFAULT_OUT_DIR))
    folder_name = cfg.get("folder_name")
    main_study_root_folder = str(session_cfg.get("main_study_root_folder", "main_study")).strip()
    if folder_name is None and main_study_root_folder:
        folder_name = str(Path(main_study_root_folder) / proband_id)
    run_timestamp = build_run_timestamp()
    session_id, _ = determine_next_session_id(proband_id, phase_short, out_dir, folder_name=folder_name)
    paths = build_output_paths(
        proband_id=proband_id,
        session_id=session_id,
        phase_short=phase_short,
        run_timestamp=run_timestamp,
        out_dir=out_dir,
        folder_name=folder_name,
    ).as_dict()
    session_logger = setup_logging(paths["log"], f"bci.{phase_short}")
    session_logger.info("Main study setup | proband=%s | session=%s", proband_id, session_id)

    # ---- Load stimuli -----------------------------------------------------
    stimuli_list = _load_json_list(stimuli_json_path)
    stimuli_by_id: Dict[int, Dict[str, Any]] = {}
    for stim in stimuli_list:
        if "id" not in stim:
            continue
        stim_id = int(stim["id"])
        if stim_id in stimuli_by_id:
            raise ValueError(f"Duplicate stimulus id in stimuli JSON: {stim_id}")
        if "luminance_depth" in stim:
            depth = float(stim["luminance_depth"])
            if not 0.0 <= depth <= 1.0:
                raise ValueError(f"stimulus {stim_id}: luminance_depth must be in [0, 1].")
        if "contrast_percent" in stim:
            contrast_value = float(stim["contrast_percent"])
            if contrast_value < 0.0 or contrast_value > 100.0:
                raise ValueError(f"stimulus {stim_id}: contrast_percent must be in [0, 100].")
        stimuli_by_id[stim_id] = stim

    block_order_spec = list(session_cfg.get("block_order") or [])
    if not block_order_spec:
        raise ValueError("Session config is missing a 'block_order' list.")
    missing_stimulus_ids = sorted(
        {
            int(block["stimulus_id"])
            for block in block_order_spec
            if block.get("stimulus_id") is not None and int(block["stimulus_id"]) not in stimuli_by_id
        }
    )
    if missing_stimulus_ids:
        raise KeyError(f"block_order references missing stimulus id(s): {missing_stimulus_ids}")
    has_idle_block = any(block.get("stimulus_id") is None for block in block_order_spec)
    idle_trials_per_source_block = (
        int(runtime_params.get("idle_trials_per_source_block", DEFAULT_MS_IDLE_TRIALS_PER_SOURCE_BLOCK))
        if has_idle_block else 0
    )
    idle_gaze_targets = _load_idle_gaze_targets(session_cfg) if has_idle_block else []
    idle_target_positions_by_target = (
        _load_idle_target_positions(session_cfg, spacing_deg=target_spacing_deg)
        if has_idle_block else []
    )
    idle_post_stop_hold_s = float(session_cfg.get("idle_post_stop_hold_s", 0.1)) if has_idle_block else 0.0
    canonical_stim_blocks_spec = [b for b in block_order_spec if b.get("stimulus_id") is not None]
    idle_blocks_spec = [b for b in block_order_spec if b.get("stimulus_id") is None]
    block_order_strategy = str(session_cfg.get("block_order_strategy", "balanced_rotation_by_proband"))
    stim_blocks_spec, block_order_meta = _resolve_active_block_order(
        canonical_stim_blocks_spec,
        proband_id=proband_id,
        strategy=block_order_strategy,
        seed_base=seed_base,
        lock_tail_count=int(session_cfg.get("block_order_lock_tail_count", 0)),
    )
    expected_idle_trials_total = len(stim_blocks_spec) * idle_trials_per_source_block
    idle_source_phase_s = idle_trials_per_source_block * trial_duration_s + idle_post_stop_hold_s

    # Deterministic frequency-to-position mapping per proband.
    freq_to_slot = _assign_frequency_mapping(frequencies_hz, target_positions_deg, proband_id)
    # Pre-compute the position tuple in target order (so target idx 0 == freq[0]).
    positions_by_target: List[Tuple[float, float]] = [
        target_positions_deg[freq_to_slot[t]] for t in range(len(frequencies_hz))
    ]
    position_label_by_target: List[str] = [
        target_position_labels[freq_to_slot[t]] if freq_to_slot[t] < len(target_position_labels)
        else _xy_label(target_positions_deg[freq_to_slot[t]])
        for t in range(len(frequencies_hz))
    ]

    # ---- Session metadata / protocol scaffold -----------------------------
    trial_rows: List[Dict[str, Any]] = []
    idle_rows: List[Dict[str, Any]] = []
    rating_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    protocol: Dict[str, Any] = {
        "study": "BCI_BA_MAIN_STUDY",
        "phase": phase_short,
        "phase_label": phase_label,
        "schema_version": session_cfg.get("schema_version"),
        "proband_id": proband_id,
        "session_id": session_id,
        "datetime_start": now_iso(),
        "datetime_end": None,
        "random_seed_base": seed_base,
        "acquisition_mode": acquisition_mode,
        "interface_language": interface_language,
        "runtime_parameters": runtime_params,
        "trials_per_target": trials_per_target,
        "trials_per_frequency": trials_per_target,
        "trials_per_block": trials_per_block,
        "session_config_path": str(session_cfg_path),
        "session_config": session_cfg,
        "stimuli_json_path": str(stimuli_json_path),
        "stimuli_definition": stimuli_list,
        "proband_schema_path": str(proband_schema_path),
        "proband_schema_version": schema.get("schema_version"),
        "pre_flight_compliance": compliance_result,
        "q1_trait": q1_answers,
        "q2_state": q2_answers,
        "q3_environment": q3_answers,
        "exclusion_hits": [{"key": k, "reason": r} for k, r in exclusion_hits],
        "frequencies_hz": frequencies_hz,
        "target_spacing_deg": target_spacing_deg,
        "target_size_deg": target_size_deg,
        "target_positions_deg": [list(p) for p in target_positions_deg],
        "target_position_labels": target_position_labels,
        "frequency_to_position_slot": freq_to_slot,
        "positions_by_target": [list(p) for p in positions_by_target],
        "contrast_percent": _contrast_percent({}, session_cfg),
        "block_order": block_order_meta,
        "pause_anchor": session_cfg.get("pause_anchor", "after_ratings"),
        "idle_trials_per_source_block": idle_trials_per_source_block,
        "idle_trials_total_expected": expected_idle_trials_total,
        "idle_post_stop_hold_s": idle_post_stop_hold_s,
        "idle_source_phase_duration_s": round(idle_source_phase_s, 6),
        "idle_mode": session_cfg.get("idle_mode", "corner_stimuli_center_fixation"),
        "idle_target_positions_deg": [list(p) for p in idle_target_positions_by_target],
        "idle_gaze_targets": idle_gaze_targets,
        "dataset_plan": [
            {
                "dataset_id": str(block.get("dataset_id", f"DS{idx}")),
                "active_block_id": str(block.get("block_id", f"B{idx}")),
                "active_trials_per_person": trials_per_block,
                "idle_trials_per_person": idle_trials_per_source_block,
                "total_trials_per_person": trials_per_block + idle_trials_per_source_block,
            }
            for idx, block in enumerate(stim_blocks_spec, start=1)
        ],
        "expected_refresh_hz": round(MONITOR_REFRESH_RATE, 3),
        "refresh_frequency_audit": _build_refresh_frequency_audit(
            frequencies_hz,
            refresh_hz=MONITOR_REFRESH_RATE,
            trial_duration_s=trial_duration_s,
        ),
        "design_constraints": {
            "monitor_refresh_rate_hz": 60.0,
            "eeg_sampling_rate_hz": 256.0,
            "headset": "EPOC X",
            "frequency_set_fixed": True,
            "idle_block_fixed": True,
            "nominal_modulation_depth_only": True,
            "modulation_depth_definition": (
                "luminance_depth is the remaining software RGB dark-bright "
                "difference. It is not a measured physical luminance contrast."
            ),
        },
        "impedance_events": [],
        "baseline_log": [],
        "block_log": [],
        "trial_log": [],
        "comfort_log": [],
        "stimulus_ranking": [],
        "break_log": [],
        "event_log": [],
        "post_session": None,
        "status": "running",
        "aborted": False,
        "export_ok": False,
        "output_folder": paths["folder"],
        "output_proband_folder": paths["proband_folder"],
        "output_main_study_folder": str(Path(out_dir) / main_study_root_folder) if main_study_root_folder else str(out_dir),
    }
    if has_idle_block:
        protocol["idle_trial_log"] = []

    def _append_event(row: Dict[str, Any]) -> None:
        row.setdefault("session_id", session_id)
        row.setdefault("proband_id", proband_id)
        row.setdefault("phase", phase_short)
        row.setdefault("timestamp_iso", now_iso())
        event_rows.append(row)
        protocol["event_log"].append(row)
        append_csv_rows(paths["events"], EVENT_FIELDS, [row], logger=session_logger)

    def _snapshot() -> None:
        safe_write_json(protocol, paths["protocol"], logger=session_logger)

    _snapshot()

    # ---- Headless dry run terminates here with protocol-only output ----
    if cfg.get("dry_run", False):
        protocol["status"] = "dry_run_completed"
        protocol["datetime_end"] = now_iso()
        _snapshot()
        return protocol

    # ---- Full runtime (PsychoPy + Cortex) ---------------------------------
    from psychopy import core, event, visual  # type: ignore

    from stimuli.display import (
        check_escape,
        make_info_stim,
        measure_refresh_rate,
        seconds_to_frames,
        show_blank,
        show_fixation,
        show_message,
    )
    from acquisition.impedance import ImpedanceMonitor, run_impedance_gate
    from acquisition.live_stream import CortexLiveStream
    from acquisition.marker import Marker
    from acquisition.recording import CortexRecording
    from acquisition.recovery import TrialConnectionLost, raise_if_disconnected, recover_after_disconnect
    from experiments.stimulus_screening import RatingDisplay, _collect_ratings
    from stimuli.generator import (
        StimulusGenerator,
        build_fixation_stimuli,
        make_window,
        normalize_stimulus_cfg,
    )

    headset_id = str(cfg.get("headset_id", DEFAULT_HEADSET_ID))
    screen_id = int(cfg.get("screen_id", DEFAULT_SCREEN_ID))
    fullscreen = bool(cfg.get("fullscreen", DEFAULT_FULLSCREEN))
    live_streams = [str(s).lower() for s in cfg.get("live_streams", list(DEFAULT_LIVE_STREAMS))]
    live_flush_every = int(cfg.get("live_flush_every", DEFAULT_LIVE_FLUSH_EVERY))
    background_color = session_cfg.get("background_color") or list(DEFAULT_MS_BACKGROUND_COLOR)
    stim_on_color = session_cfg.get("stim_on_color") or list(DEFAULT_MS_STIM_ON_COLOR)
    stim_off_color = session_cfg.get("stim_off_color") or list(DEFAULT_MS_STIM_OFF_COLOR)

    marker: Optional[Marker] = None
    recording: Optional[CortexRecording] = None
    live: Optional[CortexLiveStream] = None
    impedance_monitor: Optional[ImpedanceMonitor] = None
    win: Optional[visual.Window] = None
    aborted = False

    def _flip_and_capture(label: str, value: int, metadata: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """Schedule a marker on the next flip and return (monotonic_flip_time, marker_time_ms)."""
        flip_time_container: List[float] = [0.0]
        marker_ms_container: List[float] = [0.0]

        def _callback() -> None:
            mono = time.monotonic()
            ts_ms = marker.capture_flip_time_ms(mono) if marker is not None else mono * 1000.0
            flip_time_container[0] = mono
            marker_ms_container[0] = ts_ms
            if marker is not None and acquisition_mode == "record":
                try:
                    marker.inject_marker(
                        value=str(value),
                        label=label,
                        timestamp_ms=ts_ms,
                        capture_method="flip_callback",
                    )
                except Exception:
                    session_logger.exception("inject_marker failed | label=%s", label)
            if live is not None:
                meta = {
                    "session_id": session_id,
                    "proband_id": proband_id,
                }
                if metadata:
                    meta.update(metadata)
                try:
                    live.register_marker(
                        marker_value=int(value),
                        marker_label=str(label),
                        marker_time_ms=float(ts_ms),
                        marker_role="trial_marker",
                        metadata=meta,
                    )
                except Exception:
                    session_logger.exception("live.register_marker failed | label=%s", label)

        win.callOnFlip(_callback)
        win.flip()
        return flip_time_container[0], marker_ms_container[0]

    try:
        # ---- Step 4: Cortex / marker session --------------------------
        marker = Marker(debug_mode=False)
        marker.connect(headset_id=headset_id)
        if not marker.wait_for_session(timeout=90.0):
            raise RuntimeError("Cortex session could not be established.")
        session_logger.info("Marker session ready")

        record_title = build_cortex_record_title(proband_id, phase_short, session_id, run_timestamp)
        if acquisition_mode == "record":
            recording = CortexRecording(marker, log=session_logger)
            if not recording.start_record(record_title, description=f"BCI BA {phase_label}"):
                raise RuntimeError("Cortex recording could not be started.")
            session_logger.info("Cortex record started | title=%s", record_title)
        else:
            live = CortexLiveStream(
                marker=marker,
                output_dir=Path(paths["folder"]) / "live_stream",
                streams=live_streams,
                flush_every=live_flush_every,
                log=session_logger,
            )
            live.start(wait_labels_s=10.0)

        # ---- Window + generator ---------------------------------------
        win = make_window(fullscr=fullscreen, screen=screen_id, background_color=background_color)
        info_stim = make_info_stim(win)
        measured_refresh: List[float | None] = []
        refresh_hz = measure_refresh_rate(win, measured_out=measured_refresh)
        measured_hz = measured_refresh[0] if measured_refresh else None
        protocol["monitor_refresh_hz"] = round(refresh_hz, 3)
        protocol["measured_refresh_hz"] = round(measured_hz, 3) if measured_hz else None
        protocol["refresh_frequency_audit_measured"] = _build_refresh_frequency_audit(
            frequencies_hz,
            refresh_hz=refresh_hz,
            trial_duration_s=trial_duration_s,
        )
        session_logger.info(
            "Refresh rate expected=%.3f Hz | measured=%s Hz",
            MONITOR_REFRESH_RATE,
            f"{measured_hz:.3f}" if measured_hz else "unavailable",
        )
        if measured_hz is not None and abs(float(measured_hz) - float(MONITOR_REFRESH_RATE)) > 2.0:
            raise RuntimeError(
                "Measured display refresh rate differs from the fixed 60 Hz study setup "
                f"(expected {MONITOR_REFRESH_RATE:.3f} Hz, measured {measured_hz:.3f} Hz). "
                "Set the display to 60 Hz before recording."
            )

        generator = StimulusGenerator(
            win,
            monitor_refresh_rate=refresh_hz,
            luminance_depth=1.0,  # per-stimulus overrides are applied per run
            luminance_mod_kind="sine",
            background_color=background_color,
            stim_on_color=stim_on_color,
            stim_off_color=stim_off_color,
            abort_check=lambda: marker is not None and not marker.is_headset_connected(),
        )
        fixation = build_fixation_stimuli(win)
        rating_ui = RatingDisplay(win, language=interface_language)

        # ---- Step 5: initial impedance gate (5 s stable) --------------
        impedance_monitor = ImpedanceMonitor(marker.c, log=session_logger)
        impedance_monitor.start()

        def _impedance_event(name: str, payload: Dict[str, Any]) -> None:
            _append_event({
                "event_type": f"impedance_{name}",
                "impedance_mode": payload.get("mode"),
                "impedance_min_quality": payload.get("min_quality"),
                "planned_duration_s": payload.get("stable_s"),
                "actual_duration_s": payload.get("elapsed_s"),
                "marker_label": f"IMPEDANCE_{name.upper()}",
                "t_monotonic": round(time.monotonic(), 6),
            })

        imp_initial = run_impedance_gate(
            win,
            marker.c,
            stable_s=impedance_stable_s_initial,
            quality_threshold=impedance_quality_threshold,
            timeout_s=impedance_timeout_s,
            mode_label="initial",
            monitor=impedance_monitor,
            on_event=_impedance_event,
        )
        protocol["impedance_events"].append(imp_initial)
        if imp_initial.get("aborted"):
            aborted = True
            raise RuntimeError("Session aborted at initial impedance gate.")
        if imp_initial.get("timeout"):
            session_logger.warning("Initial impedance gate timed out - operator must intervene.")
        _snapshot()

        # ---- Step 6: baseline eyes-open (+ optional eyes-closed) -------
        est_min = _estimate_session_duration_min(
            baseline_open_s=baseline_open_s,
            baseline_closed_s=baseline_closed_s,
            include_eyes_closed_baseline=include_eyes_closed_baseline,
            active_block_count=len(stim_blocks_spec),
            trials_per_block=trials_per_block,
            cue_duration_s=cue_duration_s,
            trial_duration_s=trial_duration_s,
            fix_range_s=fix_range_s,
            iti_range_s=iti_range_s,
            block_break_s=block_break_s,
            long_break_s=long_break_s,
            long_break_after_blocks=long_break_after_blocks,
            idle_block_count=1 if idle_blocks_spec else 0,
            idle_trials_total=expected_idle_trials_total,
            idle_post_stop_hold_s=idle_post_stop_hold_s,
            impedance_stable_s_initial=impedance_stable_s_initial,
            impedance_stable_s_between=impedance_stable_s_between,
        )
        key = show_message(
            win, info_stim,
            _ui(
                interface_language,
                (
                    f"{phase_label}\n\n"
                    f"Participant: {proband_id}  |  Session: {session_id}\n"
                    f"Active blocks: {len(stim_blocks_spec)}   Idle block: {1 if idle_blocks_spec else 0}\n"
                    f"Trials/target: {trials_per_target}  ({trials_per_block} active trials/block)\n"
                    f"Idle trials: {expected_idle_trials_total}\n"
                    f"Estimated duration: ~{est_min:.0f} min\n\n"
                    f"After active blocks a short comfort questionnaire is shown.\n\n"
                    f"Press SPACE to start."
                ),
                (
                    f"{phase_label}\n\n"
                    f"Teilnehmer: {proband_id}  |  Sitzung: {session_id}\n"
                    f"Aktive Blöcke: {len(stim_blocks_spec)}   Idle-Block: {1 if idle_blocks_spec else 0}\n"
                    f"Trials/Ziel: {trials_per_target}  ({trials_per_block} aktive Trials/Block)\n"
                    f"Idle-Trials: {expected_idle_trials_total}\n"
                    f"Geschätzte Dauer: ca. {est_min:.0f} min\n\n"
                    f"Nach aktiven Blöcken folgen kurze Komfortfragen.\n\n"
                    f"Leertaste drücken zum Start."
                ),
            ),
        )
        if key == "escape":
            aborted = True
            raise RuntimeError("User aborted before baseline.")

        # Eyes-open
        key = show_message(
            win,
            info_stim,
            _ui(
                interface_language,
                f"Baseline - eyes open\n\nFixate the cross. Duration: {baseline_open_s:.0f} s.\n\nPress SPACE to start.",
                f"Baseline - Augen offen\n\nBitte das Kreuz fixieren. Dauer: {baseline_open_s:.0f} s.\n\nLeertaste drücken zum Start.",
            ),
        )
        if key == "escape":
            aborted = True
            raise RuntimeError("User aborted before eyes-open baseline.")
        for stim in fixation:
            stim.draw()
        bl_start_mono, bl_start_ms = _flip_and_capture(
            "BASELINE_OPEN_START", 10, {"baseline_type": "eyes_open"}
        )
        show_fixation(win, fixation, seconds_to_frames(baseline_open_s, refresh_hz) - 1)
        bl_stop_mono, bl_stop_ms = _flip_and_capture(
            "BASELINE_OPEN_STOP", 11, {"baseline_type": "eyes_open"}
        )
        protocol["baseline_log"].append({
            "type": "eyes_open",
            "start_marker": 10,
            "stop_marker": 11,
            "t_start_monotonic": round(bl_start_mono, 6),
            "t_stop_monotonic": round(bl_stop_mono, 6),
            "t_start_marker_ms": round(bl_start_ms, 3),
            "t_stop_marker_ms": round(bl_stop_ms, 3),
            "actual_duration_s": round(bl_stop_mono - bl_start_mono, 6),
        })
        _snapshot()

        if include_eyes_closed_baseline:
            key = show_message(
                win,
                info_stim,
                _ui(
                    interface_language,
                    f"Baseline - eyes closed\n\nClose your eyes after start.\nDuration: {baseline_closed_s:.0f} s.\n\nPress SPACE, then close.",
                    f"Baseline - Augen geschlossen\n\nNach dem Start bitte die Augen schließen. Dauer: {baseline_closed_s:.0f} s.\n\nLeertaste drücken, dann Augen schließen.",
                ),
            )
            if key == "escape":
                aborted = True
                raise RuntimeError("User aborted before eyes-closed baseline.")
            bl_start_mono, bl_start_ms = _flip_and_capture(
                "BASELINE_CLOSED_START", 20, {"baseline_type": "eyes_closed"}
            )
            show_blank(win, seconds_to_frames(baseline_closed_s, refresh_hz) - 1)
            bl_stop_mono, bl_stop_ms = _flip_and_capture(
                "BASELINE_CLOSED_STOP", 21, {"baseline_type": "eyes_closed"}
            )
            protocol["baseline_log"].append({
                "type": "eyes_closed",
                "start_marker": 20,
                "stop_marker": 21,
                "t_start_monotonic": round(bl_start_mono, 6),
                "t_stop_monotonic": round(bl_stop_mono, 6),
                "t_start_marker_ms": round(bl_start_ms, 3),
                "t_stop_marker_ms": round(bl_stop_ms, 3),
                "actual_duration_s": round(bl_stop_mono - bl_start_mono, 6),
            })
            _snapshot()

        # ---- Step 7: active block loop (counterbalanced active blocks) -
        global_trial = 0
        stim_blocks = list(stim_blocks_spec)
        idle_blocks = list(idle_blocks_spec)
        total_blocks_display = len(stim_blocks) + (1 if idle_blocks else 0)

        for block_idx, block_spec in enumerate(stim_blocks, start=1):
            if check_escape():
                aborted = True
                break

            block_id = str(block_spec.get("block_id", f"B{block_idx}"))
            dataset_id = str(block_spec.get("dataset_id", f"DS{block_idx}"))
            stim_id = int(block_spec["stimulus_id"])
            stim_cfg_raw = stimuli_by_id.get(stim_id)
            if stim_cfg_raw is None:
                raise KeyError(f"block {block_id}: stimulus_id={stim_id} not in stimulus JSON.")
            stim_cfg = normalize_stimulus_cfg(stim_cfg_raw)
            stim_name = str(stim_cfg.get("name", f"stimulus_{stim_id}"))
            stim_type = str(stim_cfg.get("type", "ssvep")).lower()
            stim_shape = str(stim_cfg.get("shape", ""))
            configured_lum_kind = str(stim_cfg.get("luminance_mod_kind", "sine"))
            configured_lum_depth = float(stim_cfg.get("luminance_depth", 1.0))
            lum_kind, lum_depth = resolve_active_modulation(
                stim_type,
                configured_lum_kind,
                configured_lum_depth,
            )
            contrast_percent = _contrast_percent(stim_cfg, session_cfg)
            block_background_color = stim_cfg.get("background_color") or background_color

            # Reconfigure the generator's modulation and colors for this block.
            generator.luminance_mod_kind = lum_kind
            generator.luminance_depth = lum_depth
            generator.set_colors(
                stim_on_color=stim_cfg.get("stim_on_color") or stim_on_color,
                stim_off_color=stim_cfg.get("stim_off_color") or stim_off_color,
                background_color=block_background_color,
            )
            win.color = block_background_color

            trial_start_marker = CONDITION_MARKER_START_BASE + block_idx
            trial_stop_marker = CONDITION_MARKER_STOP_BASE + block_idx
            block_start_marker = _main_study_block_marker(block_idx, role="start")
            block_stop_marker = _main_study_block_marker(block_idx, role="stop")

            block_log_entry = {
                "block_idx": block_idx,
                "block_id": block_id,
                "dataset_id": dataset_id,
                "stimulus_id": stim_id,
                "stimulus_name": stim_name,
                "stimulus_shape": stim_shape,
                "luminance_mod_kind": lum_kind,
                "luminance_depth": lum_depth,
                "contrast_percent": contrast_percent,
                "trials_per_target": trials_per_target,
                "trials_per_frequency": trials_per_target,
                "trials_per_block": trials_per_block,
                "trial_start_marker": trial_start_marker,
                "trial_stop_marker": trial_stop_marker,
                "block_start_marker": block_start_marker,
                "block_stop_marker": block_stop_marker,
                "t_block_monotonic": round(time.monotonic(), 6),
            }
            protocol["block_log"].append(block_log_entry)
            _snapshot()
            session_logger.info(
                "Block %d/%d (%s) | stim=%s | shape=%s | mod=%s | depth=%.2f | contrast=%.1f%%",
                block_idx, len(stim_blocks), block_id, stim_name, stim_shape, lum_kind, lum_depth, contrast_percent,
            )

            key = show_message(
                win,
                info_stim,
                _ui(
                    interface_language,
                    (
                        f"Block {block_idx}/{total_blocks_display} - {block_id}\n\n"
                        f"Stimulus: {_stimulus_display_name(stim_name, lum_kind, contrast_percent, interface_language)}\n\n"
                        f"Fixate the cue, then the indicated target.\n\n"
                        f"Press SPACE to start."
                    ),
                    (
                        f"Block {block_idx}/{total_blocks_display} - {block_id}\n\n"
                        f"Stimulus: {_stimulus_display_name(stim_name, lum_kind, contrast_percent, interface_language)}\n\n"
                        f"Bitte zuerst den Cue und danach das angezeigte Ziel fixieren.\n\n"
                        f"Leertaste drücken zum Start."
                    ),
                ),
            )
            if key == "escape":
                aborted = True
                break

            # ---- Block start marker ----------------------------------
            block_start_mono, block_start_ms = _flip_and_capture(
                f"BLOCK_ON_{block_id}", block_start_marker, {"block_idx": block_idx, "block_id": block_id}
            )
            block_log_entry["t_block_start_monotonic"] = round(block_start_mono, 6)
            block_log_entry["t_block_start_marker_ms"] = round(block_start_ms, 3)

            # ---- Trial-cue sequence ----------------------------------
            cue_seed = seed_base + block_idx * 1000
            cue_seq = _build_cue_sequence(
                trials_per_target=trials_per_target,
                n_targets=len(frequencies_hz),
                seed=cue_seed,
                fixed_target_idx=fixed_cue_target_idx,
            )
            trial_rng = random.Random(cue_seed + 7)

            cue_index = 0
            while cue_index < len(cue_seq):
                trial_in_block = cue_index + 1
                cue_target_idx = cue_seq[cue_index]
                if check_escape():
                    aborted = True
                    break
                planned_global_trial = global_trial + 1

                fix_s = trial_rng.uniform(*fix_range_s)
                iti_s = trial_rng.uniform(*iti_range_s)
                cue_position = positions_by_target[cue_target_idx]
                cue_position_label = position_label_by_target[cue_target_idx]
                cue_frequency_hz = frequencies_hz[cue_target_idx]
                cue_marker_value = multi_target_cue_marker(cue_target_idx)
                cue_marker_lbl = multi_target_cue_label(cue_target_idx, cue_frequency_hz)

                # Fixation delay.
                if show_fixation(win, fixation, seconds_to_frames(fix_s, refresh_hz)):
                    aborted = True
                    break
                try:
                    raise_if_disconnected(marker)
                except TrialConnectionLost:
                    recovery = recover_after_disconnect(
                        win=win,
                        info_stim=info_stim,
                        marker=marker,
                        marker_cortex=marker.c,
                        run_impedance_gate=run_impedance_gate,
                        stable_s=impedance_stable_s_between,
                        quality_threshold=impedance_quality_threshold,
                        timeout_s=impedance_timeout_s,
                        mode_label=f"recovery_{block_id}_trial_{trial_in_block}",
                        monitor=impedance_monitor,
                        on_event=_impedance_event,
                        log=session_logger,
                    )
                    protocol.setdefault("recovery_events", []).append(recovery)
                    _snapshot()
                    continue
                global_trial = planned_global_trial

                # Cue phase: arrow/marker on the cued target, for cue_duration_s.
                cue_dot = visual.Circle(
                    win, radius=0.5, fillColor=(1.0, 1.0, 1.0), lineColor=None,
                    pos=cue_position, units="deg",
                )
                cue_frames = max(1, seconds_to_frames(cue_duration_s, refresh_hz))
                cue_flip_mono: Optional[float] = None
                cue_flip_ms: Optional[float] = None
                for frame_i in range(cue_frames):
                    if check_escape():
                        aborted = True
                        break
                    for stim in fixation:
                        stim.draw()
                    cue_dot.draw()
                    if frame_i == 0:
                        cue_flip_mono, cue_flip_ms = _flip_and_capture(
                            cue_marker_lbl, cue_marker_value,
                            {
                                "block_idx": block_idx,
                                "block_id": block_id,
                                "trial_idx_global": global_trial,
                                "cue_target_idx": cue_target_idx,
                                "cue_frequency_hz": cue_frequency_hz,
                                "marker_role": "cue",
                            },
                        )
                    else:
                        win.flip()
                if aborted:
                    break

                # Stimulation phase: four targets running concurrently.
                onset_mono_container: List[float] = [0.0]
                onset_ms_container: List[float] = [0.0]

                def _onset_callback(flip_time: float) -> None:
                    # Invoked by run_multi_target_trial on first flip.
                    mono = time.monotonic()
                    ts_ms = marker.capture_flip_time_ms(mono) if marker is not None else mono * 1000.0
                    onset_mono_container[0] = mono
                    onset_ms_container[0] = ts_ms
                    if marker is not None and acquisition_mode == "record":
                        try:
                            marker.inject_marker(
                                value=str(trial_start_marker),
                                label=f"TRIAL_ON_{block_id}",
                                timestamp_ms=ts_ms,
                                capture_method="flip_callback",
                            )
                        except Exception:
                            session_logger.exception("inject_marker failed (trial onset)")
                    if live is not None:
                        try:
                            live.register_marker(
                                marker_value=int(trial_start_marker),
                                marker_label=f"TRIAL_ON_{block_id}",
                                marker_time_ms=float(ts_ms),
                                marker_role="trial_on",
                                metadata={
                                    "session_id": session_id,
                                    "proband_id": proband_id,
                                    "block_idx": block_idx,
                                    "block_id": block_id,
                                    "trial_idx_global": global_trial,
                                    "cue_target_idx": cue_target_idx,
                                    "cue_frequency_hz": cue_frequency_hz,
                                },
                            )
                        except Exception:
                            session_logger.exception("live.register_marker failed (trial onset)")

                trial_result = generator.run_multi_target_trial(
                    base_cfg={**stim_cfg, "size_deg": target_size_deg},
                    frequencies_hz=frequencies_hz,
                    positions_deg=positions_by_target,
                    duration_s=trial_duration_s,
                    on_first_flip=_onset_callback,
                )
                if trial_result.get("aborted") and marker is not None and not marker.is_headset_connected():
                    global_trial = planned_global_trial - 1
                    recovery = recover_after_disconnect(
                        win=win,
                        info_stim=info_stim,
                        marker=marker,
                        marker_cortex=marker.c,
                        run_impedance_gate=run_impedance_gate,
                        stable_s=impedance_stable_s_between,
                        quality_threshold=impedance_quality_threshold,
                        timeout_s=impedance_timeout_s,
                        mode_label=f"recovery_{block_id}_trial_{trial_in_block}",
                        monitor=impedance_monitor,
                        on_event=_impedance_event,
                        log=session_logger,
                    )
                    protocol.setdefault("recovery_events", []).append(recovery)
                    _snapshot()
                    continue
                if trial_result.get("aborted"):
                    aborted = True

                offset_mono, offset_ms = _flip_and_capture(
                    f"TRIAL_OFF_{block_id}", trial_stop_marker,
                    {
                        "block_idx": block_idx,
                        "block_id": block_id,
                        "trial_idx_global": global_trial,
                        "marker_role": "trial_off",
                    },
                )

                # ITI
                iti_frames = seconds_to_frames(iti_s, refresh_hz)
                if iti_frames > 1 and show_blank(win, iti_frames - 1):
                    aborted = True

                trial_row = {
                    "session_id": session_id,
                    "proband_id": proband_id,
                    "phase": phase_short,
                    "trial_idx_global": global_trial,
                    "block_idx": block_idx,
                    "trial_idx_in_block": trial_in_block,
                    "block_id": block_id,
                    "dataset_id": dataset_id,
                    "stimulus_id": stim_id,
                    "stimulus_name": stim_name,
                    "stimulus_type": stim_type,
                    "stimulus_shape": stim_shape,
                    "luminance_mod_kind": lum_kind,
                    "luminance_depth": lum_depth,
                    "contrast_percent": contrast_percent,
                    "cue_target_idx": cue_target_idx,
                    "cue_position_label": cue_position_label,
                    "cue_position_deg_x": cue_position[0],
                    "cue_position_deg_y": cue_position[1],
                    "cue_frequency_hz": cue_frequency_hz,
                    "frequencies_hz": list(frequencies_hz),
                    "positions_deg": [list(p) for p in positions_by_target],
                    "fix_duration_s": round(fix_s, 6),
                    "cue_duration_s": round(cue_duration_s, 6),
                    "iti_duration_s": round(iti_s, 6),
                    "nominal_duration_s": round(trial_duration_s, 6),
                    "actual_duration_s": round(offset_mono - onset_mono_container[0], 6) if onset_mono_container[0] else None,
                    "cue_marker_value": cue_marker_value,
                    "cue_marker_label": cue_marker_lbl,
                    "start_marker": trial_start_marker,
                    "stop_marker": trial_stop_marker,
                    "t_cue_monotonic": round(cue_flip_mono, 6) if cue_flip_mono is not None else None,
                    "t_onset_monotonic": round(onset_mono_container[0], 6) if onset_mono_container[0] else None,
                    "t_offset_monotonic": round(offset_mono, 6),
                    "t_cue_marker_ms": round(cue_flip_ms, 3) if cue_flip_ms is not None else None,
                    "t_onset_marker_ms": round(onset_ms_container[0], 3) if onset_ms_container[0] else None,
                    "t_offset_marker_ms": round(offset_ms, 3),
                }
                trial_rows.append(trial_row)
                protocol["trial_log"].append(trial_row)
                append_csv_rows(paths["trials"], TRIAL_FIELDS, [trial_row], logger=session_logger)
                cue_index += 1

                if aborted:
                    break

            # Block stop marker
            block_stop_mono, block_stop_ms = _flip_and_capture(
                f"BLOCK_OFF_{block_id}", block_stop_marker,
                {"block_idx": block_idx, "block_id": block_id},
            )
            block_log_entry["t_block_stop_monotonic"] = round(block_stop_mono, 6)
            block_log_entry["t_block_stop_marker_ms"] = round(block_stop_ms, 3)
            _snapshot()
            win.color = background_color

            if aborted:
                break

            # ---- Comfort ratings (reuses stimulus_screening RatingDisplay) --
            block_ratings = _collect_ratings(
                win, rating_ui,
                session_id=session_id, proband_id=proband_id,
                block_idx=block_idx, condition_id=block_id,
                stimulus_id=stim_id, stimulus_name=stim_name,
                language=interface_language,
            )
            if block_ratings is None:
                aborted = True
                break
            for row in block_ratings:
                row.setdefault("block_id", block_id)
                row.setdefault("contrast_percent", contrast_percent)
            rating_rows.extend(block_ratings)
            protocol["comfort_log"].extend(block_ratings)
            append_csv_rows(paths["ratings"], RATING_FIELDS, block_ratings, logger=session_logger)
            event.clearEvents(eventType="keyboard")
            _snapshot()

            # ---- Countdown rest time (clock starts after comfort ratings)
            is_last_stim_block = block_idx == len(stim_blocks) and not idle_blocks
            if not is_last_stim_block:
                is_long = block_idx in long_break_after_blocks
                pause_total_s = long_break_s if is_long else block_break_s
                break_anchor_mono = time.monotonic()
                elapsed_since_block_end, remaining_s = _compute_remaining_break_s(
                    break_total_s=pause_total_s,
                    block_end_monotonic=break_anchor_mono,
                    now_monotonic=break_anchor_mono,
                )
                break_type = "long_break" if is_long else "short_break"
                next_block = stim_blocks[block_idx] if block_idx < len(stim_blocks) else None
                if next_block is not None:
                    next_block_id = str(next_block.get("block_id", f"B{block_idx + 1}"))
                    next_stim_id = int(next_block["stimulus_id"])
                    next_stim = normalize_stimulus_cfg(stimuli_by_id[next_stim_id])
                    next_name = str(next_stim.get("name", f"stimulus_{next_stim_id}"))
                    next_type = str(next_stim.get("type", "ssvep")).lower()
                    next_lum_kind, _next_lum_depth = resolve_active_modulation(
                        next_type,
                        str(next_stim.get("luminance_mod_kind", "sine")),
                        float(next_stim.get("luminance_depth", 1.0)),
                    )
                    next_contrast_percent = _contrast_percent(next_stim, session_cfg)
                    next_display = _stimulus_display_name(
                        next_name, next_lum_kind, next_contrast_percent, interface_language
                    )
                    next_lines = (
                        [
                            f"Completed: {block_id} - Block {block_idx}/{total_blocks_display}",
                            f"Next: {next_block_id} - {next_display}",
                            "Impedance check follows.",
                        ]
                        if interface_language == "en" else
                        [
                            f"Abgeschlossen: {block_id} - Block {block_idx}/{total_blocks_display}",
                            f"Nächster Block: {next_block_id} - {next_display}",
                            "Impedanz-Check folgt.",
                        ]
                    )
                else:
                    _idle_label = str(idle_blocks[0].get("block_id", "B8")) if idle_blocks else "Idle"
                    next_lines = (
                        [
                            f"Completed: {block_id} - Block {block_idx}/{total_blocks_display}",
                            f"Next: {_idle_label} - Idle",
                            "Impedance check follows.",
                        ]
                        if interface_language == "en" else
                        [
                            f"Abgeschlossen: {block_id} - Block {block_idx}/{total_blocks_display}",
                            f"Nächster Block: {_idle_label} - Idle",
                            "Impedanz-Check folgt.",
                        ]
                    )
                aborted_countdown = _countdown_remaining(
                    win=win, info_stim=info_stim,
                    title=_ui(interface_language, ("Long break" if is_long else "Short break"), ("Lange Pause" if is_long else "Kurze Pause")),
                    body_lines=next_lines,
                    remaining_s=remaining_s,
                    countdown_label=_ui(interface_language, "Starts in", "Startet in"),
                    ready_text=_ui(interface_language, "Press SPACE to continue.", "Leertaste drücken zum Fortfahren."),
                )
                break_elapsed_total_s, _ = _compute_remaining_break_s(
                    break_total_s=pause_total_s,
                    block_end_monotonic=break_anchor_mono,
                    now_monotonic=time.monotonic(),
                )
                _append_event({
                    "event_type": break_type,
                    "block_idx": block_idx,
                    "block_id": block_id,
                    "after_block_idx": block_idx,
                    "break_type": break_type,
                    "break_anchor": "after_ratings",
                    "planned_duration_s": round(pause_total_s, 3),
                    "actual_duration_s": round(break_elapsed_total_s, 3),
                })
                _snapshot()
                if aborted_countdown:
                    aborted = True
                    break

                # Between-blocks impedance gate (3 s).
                imp_between = run_impedance_gate(
                    win,
                    marker.c,
                    stable_s=impedance_stable_s_between,
                    quality_threshold=impedance_quality_threshold,
                    timeout_s=impedance_timeout_s,
                    mode_label=f"between_block_{block_idx}",
                    monitor=impedance_monitor,
                    on_event=_impedance_event,
                )
                protocol["impedance_events"].append(imp_between)
                if imp_between.get("aborted"):
                    aborted = True
                    break
                _snapshot()

        # ---- Step 8: paired idle block (B8) ---------------------------
        if not aborted and idle_blocks:
            idle_spec = idle_blocks[0]
            idle_block_id = str(idle_spec.get("block_id", "B8"))
            idle_block_idx = len(stim_blocks) + 1
            idle_start_marker = _main_study_block_marker(idle_block_idx, role="start")
            idle_stop_marker = _main_study_block_marker(idle_block_idx, role="stop")

            key = show_message(
                win,
                info_stim,
                _ui(
                    interface_language,
                    (
                        f"Idle phase - {idle_block_id} ({idle_block_idx}/{total_blocks_display})\n\n"
                        f"Please relax and look at the center fixation cross.\n\n"
                        f"Press SPACE to start."
                    ),
                    (
                        f"Idle-Phase - {idle_block_id} ({idle_block_idx}/{total_blocks_display})\n\n"
                        f"Bitte entspannen und auf das Fixationskreuz in der Mitte schauen.\n\n"
                        f"Leertaste drücken zum Start."
                    ),
                ),
            )
            if key == "escape":
                aborted = True
            else:
                _, _ = _flip_and_capture(
                    f"BLOCK_ON_{idle_block_id}", idle_start_marker,
                    {"block_idx": idle_block_idx, "block_id": idle_block_id},
                )

                idle_global_index = 0

                for source_block_idx, source_block in enumerate(stim_blocks, start=1):
                    if aborted:
                        break
                    source_block_id = str(source_block.get("block_id", f"B{source_block_idx}"))
                    dataset_id = str(source_block.get("dataset_id", f"DS{source_block_idx}"))
                    source_stim_id = int(source_block["stimulus_id"])
                    source_stim_raw = stimuli_by_id.get(source_stim_id)
                    if source_stim_raw is None:
                        raise KeyError(f"idle source {source_block_id}: stimulus_id={source_stim_id} not in stimulus JSON.")
                    source_stim = normalize_stimulus_cfg(source_stim_raw)
                    source_type = str(source_stim.get("type", "ssvep")).lower()
                    source_shape = str(source_stim.get("shape", ""))
                    source_lum_kind, source_lum_depth = resolve_active_modulation(
                        source_type,
                        str(source_stim.get("luminance_mod_kind", "sine")),
                        float(source_stim.get("luminance_depth", 1.0)),
                    )
                    source_contrast_percent = _contrast_percent(source_stim, session_cfg)

                    generator.luminance_mod_kind = source_lum_kind
                    generator.luminance_depth = source_lum_depth
                    generator.set_colors(
                        stim_on_color=source_stim.get("stim_on_color") or stim_on_color,
                        stim_off_color=source_stim.get("stim_off_color") or stim_off_color,
                        background_color=source_stim.get("background_color") or background_color,
                    )

                    if check_escape():
                        aborted = True
                        break

                    gaze_idx = 0
                    angle_deg = 0.0
                    gaze_id = "center_fixation"
                    gaze_label = _ui(interface_language, "center fixation cross", "Fixationskreuz in der Mitte")

                    try:
                        raise_if_disconnected(marker)
                    except TrialConnectionLost:
                        recovery = recover_after_disconnect(
                            win=win,
                            info_stim=info_stim,
                            marker=marker,
                            marker_cortex=marker.c,
                            run_impedance_gate=run_impedance_gate,
                            stable_s=impedance_stable_s_between,
                            quality_threshold=impedance_quality_threshold,
                            timeout_s=impedance_timeout_s,
                            mode_label=f"recovery_{idle_block_id}_{source_block_id}",
                            monitor=impedance_monitor,
                            on_event=_impedance_event,
                            log=session_logger,
                        )
                        protocol.setdefault("recovery_events", []).append(recovery)
                        _snapshot()
                        continue

                    idle_trial_onsets: Dict[int, Tuple[float, float, int, str]] = {}
                    idle_trial_offsets: Dict[int, Tuple[float, float, int, str]] = {}

                    def _send_idle_trial_marker(
                        *,
                        trial_idx: int,
                        marker_value: int,
                        label: str,
                        marker_role: str,
                    ) -> Tuple[float, float]:
                        mono = time.monotonic()
                        ts_ms = marker.capture_flip_time_ms(mono) if marker is not None else mono * 1000.0
                        metadata = {
                            "session_id": session_id,
                            "proband_id": proband_id,
                            "block_id": idle_block_id,
                            "dataset_id": dataset_id,
                            "source_block_idx": source_block_idx,
                            "source_block_id": source_block_id,
                            "idle_trial_idx": int(trial_idx),
                            "angle_idx": gaze_idx,
                            "angle_deg": angle_deg,
                            "idle_gaze_id": gaze_id,
                        }
                        if marker is not None and acquisition_mode == "record":
                            try:
                                marker.inject_marker(
                                    value=str(marker_value),
                                    label=label,
                                    timestamp_ms=ts_ms,
                                    capture_method="flip_callback",
                                )
                            except Exception:
                                session_logger.exception("inject_marker failed (idle trial)")
                        if live is not None:
                            try:
                                live.register_marker(
                                    marker_value=int(marker_value),
                                    marker_label=label,
                                    marker_time_ms=float(ts_ms),
                                    marker_role=marker_role,
                                    metadata=metadata,
                                )
                            except Exception:
                                session_logger.exception("live.register_marker failed (idle trial)")
                        return mono, ts_ms

                    def _idle_trial_start(trial_idx: int, _flip_time: float | None = None) -> None:
                        marker_value = idle_trial_start_marker(
                            source_block_idx,
                            trial_idx,
                            trials_per_source=idle_trials_per_source_block,
                        )
                        label = idle_trial_marker_label(source_block_id, trial_idx, role="START")
                        mono, ts_ms = _send_idle_trial_marker(
                            trial_idx=trial_idx,
                            marker_value=marker_value,
                            label=label,
                            marker_role="idle_trial_start",
                        )
                        idle_trial_onsets[int(trial_idx)] = (mono, ts_ms, marker_value, label)

                    def _idle_trial_stop(trial_idx: int, _flip_time: float | None = None) -> None:
                        marker_value = idle_trial_stop_marker(
                            source_block_idx,
                            trial_idx,
                            trials_per_source=idle_trials_per_source_block,
                        )
                        label = idle_trial_marker_label(source_block_id, trial_idx, role="STOP")
                        mono, ts_ms = _send_idle_trial_marker(
                            trial_idx=trial_idx,
                            marker_value=marker_value,
                            label=label,
                            marker_role="idle_trial_stop",
                        )
                        idle_trial_offsets[int(trial_idx)] = (mono, ts_ms, marker_value, label)

                    idle_result = generator.run_multi_target_continuous_trials(
                        base_cfg={**source_stim, "size_deg": target_size_deg},
                        frequencies_hz=frequencies_hz,
                        positions_deg=idle_target_positions_by_target,
                        trial_duration_s=trial_duration_s,
                        trial_count=idle_trials_per_source_block,
                        on_trial_start=_idle_trial_start,
                        on_trial_stop=_idle_trial_stop,
                        post_stop_hold_s=idle_post_stop_hold_s,
                    )
                    if idle_result.get("aborted"):
                        if marker is not None and not marker.is_headset_connected():
                            recovery = recover_after_disconnect(
                                win=win,
                                info_stim=info_stim,
                                marker=marker,
                                marker_cortex=marker.c,
                                run_impedance_gate=run_impedance_gate,
                                stable_s=impedance_stable_s_between,
                                quality_threshold=impedance_quality_threshold,
                                timeout_s=impedance_timeout_s,
                                mode_label=f"recovery_{idle_block_id}_{source_block_id}",
                                monitor=impedance_monitor,
                                on_event=_impedance_event,
                                log=session_logger,
                            )
                            protocol.setdefault("recovery_events", []).append(recovery)
                            _snapshot()
                            continue
                        aborted = True

                    for trial_in_source in range(1, idle_trials_per_source_block + 1):
                        if trial_in_source not in idle_trial_onsets:
                            continue
                        idle_global_index += 1
                        onset_mono, onset_ms, start_marker, start_label = idle_trial_onsets[trial_in_source]
                        stop_info = idle_trial_offsets.get(trial_in_source)
                        if stop_info is not None:
                            offset_mono, offset_ms, stop_marker, stop_label = stop_info
                        else:
                            offset_mono = onset_mono + float(trial_duration_s)
                            offset_ms = onset_ms + float(trial_duration_s) * 1000.0
                            stop_marker = idle_trial_stop_marker(
                                source_block_idx,
                                trial_in_source,
                                trials_per_source=idle_trials_per_source_block,
                            )
                            stop_label = idle_trial_marker_label(source_block_id, trial_in_source, role="STOP")

                        idle_row = {
                            "session_id": session_id,
                            "proband_id": proband_id,
                            "phase": phase_short,
                            "trial_idx_in_idle": idle_global_index,
                            "trial_idx_in_idle_source": trial_in_source,
                            "idle_block_idx": idle_block_idx,
                            "idle_block_id": idle_block_id,
                            "dataset_id": dataset_id,
                            "source_block_idx": source_block_idx,
                            "source_block_id": source_block_id,
                            "angle_idx": gaze_idx,
                            "angle_deg": angle_deg,
                            "idle_gaze_id": gaze_id,
                            "idle_gaze_label": gaze_label,
                            "stimulus_id": source_stim_id,
                            "stimulus_name": source_stim.get("name", ""),
                            "stimulus_shape": source_shape,
                            "luminance_mod_kind": source_lum_kind,
                            "luminance_depth": source_lum_depth,
                            "contrast_percent": source_contrast_percent,
                            "frequencies_hz": list(frequencies_hz),
                            "positions_deg": [list(p) for p in idle_target_positions_by_target],
                            "fix_duration_s": 0.0,
                            "iti_duration_s": 0.0,
                            "nominal_duration_s": round(trial_duration_s, 6),
                            "actual_duration_s": round(offset_mono - onset_mono, 6),
                            "start_marker": start_marker,
                            "start_marker_label": start_label,
                            "stop_marker": stop_marker,
                            "stop_marker_label": stop_label,
                            "t_onset_monotonic": round(onset_mono, 6),
                            "t_offset_monotonic": round(offset_mono, 6),
                            "t_onset_marker_ms": round(onset_ms, 3),
                            "t_offset_marker_ms": round(offset_ms, 3),
                        }
                        idle_rows.append(idle_row)
                        protocol["idle_trial_log"].append(idle_row)

                    write_csv(
                        idle_rows,
                        str(Path(paths["folder"]) / f"{paths['stem']}_idle_trials.csv"),
                        IDLE_TRIAL_FIELDS,
                    )
                    if aborted:
                        break

                _, _ = _flip_and_capture(
                    f"BLOCK_OFF_{idle_block_id}", idle_stop_marker,
                    {"block_idx": idle_block_idx, "block_id": idle_block_id},
                )

        # ---- Step 9: end-of-session comfort ranking -------------------
        if not aborted:
            ranking_rows = _collect_end_ranking(
                win=win,
                info_stim=info_stim,
                stim_blocks=stim_blocks,
                stimuli_by_id=stimuli_by_id,
                session_cfg=session_cfg,
                generator_cls=StimulusGenerator,
                base_generator=generator,
                frequencies_hz=frequencies_hz,
                refresh_hz=refresh_hz,
                target_size_deg=target_size_deg,
                background_color=background_color,
                stim_on_color=stim_on_color,
                stim_off_color=stim_off_color,
                interface_language=interface_language,
                session_id=session_id,
                proband_id=proband_id,
                phase_short=phase_short,
            ) or []
            if not ranking_rows:
                session_logger.warning("Stimulus ranking was cancelled or empty.")
            else:
                protocol["stimulus_ranking"] = ranking_rows
                write_csv(
                    ranking_rows,
                    str(Path(paths["folder"]) / f"{paths['stem']}_stimulus_ranking.csv"),
                    RANKING_FIELDS,
                )
                _snapshot()

        # ---- Step 10: no separate post-session popup -----------------
        if not aborted:
            protocol["post_session"] = {
                "skipped": True,
                "reason": "comfort ranking is collected with active stimulus previews",
            }
            _snapshot()

    except Exception:
        session_logger.exception("Critical error in %s", phase_label)
        protocol["status"] = "error"
        protocol["aborted"] = True
        protocol["datetime_end"] = now_iso()
        _snapshot()
        raise
    finally:
        # ---- Step 10: stop acquisition + final export ----------------
        try:
            if impedance_monitor is not None:
                impedance_monitor.stop()
        except Exception:
            session_logger.exception("Impedance monitor stop failed")

        if marker is not None:
            try:
                if acquisition_mode == "record":
                    if recording is None:
                        raise RuntimeError("Recording controller is not available.")
                    recording.stop_and_export(
                        export_folder=paths["folder"],
                        data_types=["EEG", "MOTION", "PM", "BP"],
                        export_format="CSV",
                        version="V2",
                        timeout=300.0,
                    )
                    protocol["export_ok"] = True
                elif live is not None:
                    live.stop()
                    protocol["live_summary"] = live.summary()
            except Exception:
                session_logger.exception("Stop/export failed")

            try:
                protocol["marker_timing_log"] = marker.stats.log
            except Exception:
                pass

            try:
                marker.close()
            except Exception:
                session_logger.exception("Marker close failed")

        protocol["aborted"] = aborted
        protocol["status"] = "aborted" if aborted else protocol.get("status", "completed")
        if protocol["status"] == "running":
            protocol["status"] = "completed"
        protocol["datetime_end"] = now_iso()
        _snapshot()

        if trial_rows:
            write_csv(trial_rows, paths["trials"], TRIAL_FIELDS)
        if idle_rows:
            write_csv(
                idle_rows,
                str(Path(paths["folder"]) / f"{paths['stem']}_idle_trials.csv"),
                IDLE_TRIAL_FIELDS,
            )
        if rating_rows:
            write_csv(rating_rows, paths["ratings"], RATING_FIELDS)
        if ranking_rows:
            write_csv(
                ranking_rows,
                str(Path(paths["folder"]) / f"{paths['stem']}_stimulus_ranking.csv"),
                RANKING_FIELDS,
            )

        if win is not None:
            try:
                win.close()
            except Exception:
                pass
        try:
            core.quit()
        except Exception:
            pass

    return protocol


__all__ = [
    "PHASE_SHORT",
    "PHASE_LABEL",
    "TRIAL_FIELDS",
    "IDLE_TRIAL_FIELDS",
    "RATING_FIELDS",
    "RANKING_FIELDS",
    "EVENT_FIELDS",
    "run_main_study_session",
]


if __name__ == "__main__":
    print("Main study module demo")
    print("This demo does not open a PsychoPy window and does not start Cortex recording.")
    print(f"Phase: {PHASE_LABEL} ({PHASE_SHORT})")
    print(f"Cue marker base: {MULTI_TARGET_CUE_MARKER_BASE}")
    print(f"Maximum targets: {MULTI_TARGET_MAX_TARGETS}")



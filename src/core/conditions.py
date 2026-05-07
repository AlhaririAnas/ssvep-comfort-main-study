from __future__ import annotations

"""
Deterministic condition IDs and marker values.

This module does not send markers. It only creates the IDs, labels, and marker
numbers that experiment runners can use later.
"""

from typing import Any, Iterable, Mapping, MutableMapping, Sequence


CONDITION_SCHEMA_VERSION = "condition_hierarchy_v1"
CONDITION_MARKER_START_BASE = 100
CONDITION_MARKER_STOP_BASE = 150
KNOWN_STIMULUS_TYPE_ORDER: tuple[str, ...] = ("ssvep", "ssmvep", "hybrid")

# These marker ranges do not overlap with condition start/stop markers.
MULTI_TARGET_CUE_MARKER_BASE = 200
MULTI_TARGET_MAX_TARGETS = 4
IDLE_ANGLE_MARKER_BASE = 210
IDLE_ANGLE_MAX = 32
IDLE_TRIAL_START_MARKER_BASE = 300
IDLE_TRIAL_STOP_MARKER_BASE = 400
IDLE_SEGMENT_MARKER_BASE = IDLE_TRIAL_START_MARKER_BASE
IDLE_SEGMENT_MAX_SOURCE_BLOCKS = 32
IDLE_SEGMENT_MAX_PER_SOURCE = 32


def _clean_text(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _normalize_text(value: Any, *, default: str = "") -> str:
    text = _clean_text(value, default=default)
    return " ".join(text.split())


def _normalize_key_text(value: Any, *, default: str = "") -> str:
    text = _normalize_text(value, default=default).lower()
    safe_chars = []
    for char in text:
        if char.isalnum():
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    compact = "".join(safe_chars).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or default


def _normalize_float(value: Any, *, default: float = 0.0) -> float:
    if value in (None, ""):
        return float(default)
    return round(float(value), 6)


def _format_frequency_hz(value: Any) -> str:
    return f"{_normalize_float(value):.2f}"


def _format_depth(value: Any) -> str:
    return f"{_normalize_float(value):.3f}"


def stimulus_type_sort_key(stimulus_type: Any) -> tuple[int, str]:
    """Return a stable sort key for known and unknown stimulus types."""

    value = _normalize_text(stimulus_type, default="unknown").lower()
    try:
        return KNOWN_STIMULUS_TYPE_ORDER.index(value), value
    except ValueError:
        return len(KNOWN_STIMULUS_TYPE_ORDER), value


def build_condition_key(spec: Mapping[str, Any]) -> str:
    """Build the stable semantic key for one experimental condition."""

    return "|".join(
        [
            _normalize_key_text(spec.get("stimulus_type"), default="unknown"),
            _format_frequency_hz(spec.get("frequency_hz")),
            _normalize_key_text(spec.get("stimulus_name"), default="unnamed"),
            _normalize_key_text(spec.get("stimulus_shape"), default="unknown"),
            _normalize_key_text(spec.get("luminance_mod_kind"), default="unknown"),
            _format_depth(spec.get("luminance_depth")),
        ]
    )


def build_condition_label(spec: Mapping[str, Any], condition_id: int | None = None) -> str:
    """Build a human-readable label for logs and exported protocol files."""

    prefix = f"C{int(condition_id):03d} | " if condition_id is not None else ""
    return (
        f"{prefix}"
        f"{_normalize_text(spec.get('stimulus_type'), default='unknown').upper()} | "
        f"{_format_frequency_hz(spec.get('frequency_hz'))} Hz | "
        f"{_normalize_text(spec.get('stimulus_name'), default='Unnamed stimulus')} | "
        f"shape={_normalize_text(spec.get('stimulus_shape'), default='unknown')} | "
        f"mod={_normalize_text(spec.get('luminance_mod_kind'), default='unknown')} | "
        f"depth={_format_depth(spec.get('luminance_depth'))}"
    )


def build_condition_spec(
    *,
    frequency_hz: Any,
    stimulus_type: Any,
    stimulus_name: Any,
    stimulus_shape: Any,
    luminance_mod_kind: Any,
    luminance_depth: Any,
    source_stimulus_id: Any = None,
    legacy_condition_id: Any = None,
    legacy_start_marker: Any = None,
    legacy_stop_marker: Any = None,
) -> dict[str, Any]:
    """Build the normalized condition record used for ID assignment."""

    spec = {
        "frequency_hz": _normalize_float(frequency_hz),
        "stimulus_type": _normalize_text(stimulus_type, default="unknown").lower(),
        "stimulus_name": _normalize_text(stimulus_name, default="Unnamed stimulus"),
        "stimulus_shape": _normalize_text(stimulus_shape, default="unknown").lower(),
        "luminance_mod_kind": _normalize_text(luminance_mod_kind, default="unknown").lower(),
        "luminance_depth": _normalize_float(luminance_depth),
        "source_stimulus_id": None if source_stimulus_id in (None, "") else int(source_stimulus_id),
        "legacy_condition_id": None if legacy_condition_id in (None, "") else str(legacy_condition_id),
        "legacy_start_marker": None if legacy_start_marker in (None, "") else int(float(legacy_start_marker)),
        "legacy_stop_marker": None if legacy_stop_marker in (None, "") else int(float(legacy_stop_marker)),
    }
    spec["condition_key"] = build_condition_key(spec)
    return spec


def build_frequency_pretest_condition_spec(
    *,
    frequency_hz: Any,
    luminance_mod_kind: Any,
    luminance_depth: Any,
    stimulus_name: str = "SSVEP Frequency Pretest Flicker",
    stimulus_shape: str = "flicker",
) -> dict[str, Any]:
    """Build a condition spec for the frequency pretest flicker stimulus."""

    return build_condition_spec(
        frequency_hz=frequency_hz,
        stimulus_type="ssvep",
        stimulus_name=stimulus_name,
        stimulus_shape=stimulus_shape,
        luminance_mod_kind=luminance_mod_kind,
        luminance_depth=luminance_depth,
    )


def resolve_active_modulation(
    stimulus_type: Any,
    default_luminance_mod_kind: Any,
    default_luminance_depth: Any,
) -> tuple[str, float]:
    """Return the active luminance modulation for a stimulus."""

    stim_type = _normalize_text(stimulus_type, default="unknown").lower()
    if stim_type == "ssmvep":
        return "sine", 1.0
    return (
        _normalize_text(default_luminance_mod_kind, default="unknown").lower(),
        _normalize_float(default_luminance_depth),
    )


def build_stimulus_condition_spec(
    stimulus: Mapping[str, Any],
    *,
    default_luminance_mod_kind: Any,
    default_luminance_depth: Any,
) -> dict[str, Any]:
    """Build a condition spec from one stimulus definition."""

    active_kind, active_depth = resolve_active_modulation(
        stimulus.get("type"),
        default_luminance_mod_kind,
        default_luminance_depth,
    )
    return build_condition_spec(
        frequency_hz=stimulus.get("freq"),
        stimulus_type=stimulus.get("type"),
        stimulus_name=stimulus.get("name"),
        stimulus_shape=stimulus.get("shape"),
        luminance_mod_kind=active_kind,
        luminance_depth=active_depth,
        source_stimulus_id=stimulus.get("id"),
    )


def condition_sort_key(spec: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the stable sort key used before assigning condition IDs."""

    return (
        stimulus_type_sort_key(spec.get("stimulus_type")),
        _normalize_float(spec.get("frequency_hz")),
        _normalize_text(spec.get("stimulus_name"), default="").lower(),
        _normalize_text(spec.get("stimulus_shape"), default="").lower(),
        _normalize_text(spec.get("luminance_mod_kind"), default="").lower(),
        _normalize_float(spec.get("luminance_depth")),
        _normalize_text(spec.get("condition_key"), default=""),
    )


def assign_condition_ids(condition_specs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Assign deterministic IDs and marker values to unique conditions."""

    unique_by_key: dict[str, dict[str, Any]] = {}
    for raw in condition_specs:
        if "condition_key" in raw:
            spec = dict(raw)
        else:
            spec = build_condition_spec(
                frequency_hz=raw.get("frequency_hz"),
                stimulus_type=raw.get("stimulus_type"),
                stimulus_name=raw.get("stimulus_name"),
                stimulus_shape=raw.get("stimulus_shape"),
                luminance_mod_kind=raw.get("luminance_mod_kind"),
                luminance_depth=raw.get("luminance_depth"),
                source_stimulus_id=raw.get("source_stimulus_id"),
                legacy_condition_id=raw.get("legacy_condition_id"),
                legacy_start_marker=raw.get("legacy_start_marker"),
                legacy_stop_marker=raw.get("legacy_stop_marker"),
            )
        unique_by_key.setdefault(spec["condition_key"], spec)

    ordered_specs = sorted(unique_by_key.values(), key=condition_sort_key)
    conditions: list[dict[str, Any]] = []
    condition_map: dict[str, dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}

    for idx, spec in enumerate(ordered_specs, start=1):
        condition = dict(spec)
        condition["condition_id"] = idx
        condition["condition_numeric_id"] = idx
        condition["condition_label"] = build_condition_label(condition, condition_id=idx)
        condition["start_marker"] = CONDITION_MARKER_START_BASE + idx
        condition["stop_marker"] = CONDITION_MARKER_STOP_BASE + idx
        conditions.append(condition)
        condition_map[str(idx)] = condition
        by_key[condition["condition_key"]] = condition

    return {
        "schema_version": CONDITION_SCHEMA_VERSION,
        "condition_sorting_rule": [
            "stimulus_type",
            "frequency_hz",
            "stimulus_name",
            "stimulus_shape",
            "luminance_mod_kind",
            "luminance_depth",
        ],
        "condition_id_formula": "deterministic_rank_in_sorted_condition_set",
        "marker_formula": {
            "start_marker": "100 + condition_id",
            "stop_marker": "150 + condition_id",
        },
        "conditions": conditions,
        "condition_map": condition_map,
        "by_key": by_key,
    }


def condition_marker_label(prefix: str, condition_id: Any) -> str:
    """Build a compact condition marker label."""

    return f"{prefix}_C{int(condition_id):03d}"


def multi_target_cue_marker(target_idx: int) -> int:
    """Return the cue marker for a zero-based target index."""

    idx = int(target_idx)
    if idx < 0 or idx >= MULTI_TARGET_MAX_TARGETS:
        raise ValueError(
            f"target_idx must be in [0, {MULTI_TARGET_MAX_TARGETS - 1}], got {target_idx!r}"
        )
    return MULTI_TARGET_CUE_MARKER_BASE + idx + 1


def multi_target_cue_label(target_idx: int, frequency_hz: Any) -> str:
    """Return the cue marker label for one target."""

    return f"CUE_T{int(target_idx):d}_{_format_frequency_hz(frequency_hz)}Hz"


def idle_angle_marker(angle_idx: int) -> int:
    """Return the idle-trial marker for a zero-based gaze-angle index."""

    idx = int(angle_idx)
    if idx < 0 or idx >= IDLE_ANGLE_MAX:
        raise ValueError(f"angle_idx must be in [0, {IDLE_ANGLE_MAX - 1}], got {angle_idx!r}")
    return IDLE_ANGLE_MARKER_BASE + idx + 1


def idle_angle_label(angle_idx: int, angle_deg: Any) -> str:
    """Return the idle-trial marker label for one gaze angle."""

    return f"IDLE_A{int(angle_idx):d}_{_normalize_float(angle_deg):+.1f}deg"


def idle_segment_marker(source_block_idx: int, segment_idx: int, *, segments_per_source: int = 6) -> int:
    """Return the start marker for one continuous idle trial.

    Kept as a compatibility alias for earlier analysis scripts.
    """

    return idle_trial_start_marker(
        source_block_idx,
        segment_idx,
        trials_per_source=segments_per_source,
    )


def idle_trial_start_marker(source_block_idx: int, trial_idx: int, *, trials_per_source: int = 6) -> int:
    """Return a unique start marker for one continuous idle trial."""

    source_idx = int(source_block_idx)
    trial = int(trial_idx)
    trial_count = int(trials_per_source)
    if source_idx < 1 or source_idx > IDLE_SEGMENT_MAX_SOURCE_BLOCKS:
        raise ValueError(
            f"source_block_idx must be in [1, {IDLE_SEGMENT_MAX_SOURCE_BLOCKS}], got {source_block_idx!r}"
        )
    if trial < 1 or trial > trial_count or trial_count > IDLE_SEGMENT_MAX_PER_SOURCE:
        raise ValueError(
            f"trial_idx must be in [1, {trial_count}], got {trial_idx!r}"
        )
    return IDLE_TRIAL_START_MARKER_BASE + (source_idx - 1) * trial_count + trial


def idle_trial_stop_marker(source_block_idx: int, trial_idx: int, *, trials_per_source: int = 6) -> int:
    """Return a unique stop marker for one continuous idle trial."""

    source_idx = int(source_block_idx)
    trial = int(trial_idx)
    trial_count = int(trials_per_source)
    if source_idx < 1 or source_idx > IDLE_SEGMENT_MAX_SOURCE_BLOCKS:
        raise ValueError(
            f"source_block_idx must be in [1, {IDLE_SEGMENT_MAX_SOURCE_BLOCKS}], got {source_block_idx!r}"
        )
    if trial < 1 or trial > trial_count or trial_count > IDLE_SEGMENT_MAX_PER_SOURCE:
        raise ValueError(
            f"trial_idx must be in [1, {trial_count}], got {trial_idx!r}"
        )
    return IDLE_TRIAL_STOP_MARKER_BASE + (source_idx - 1) * trial_count + trial


def idle_segment_label(source_block_id: Any, segment_idx: int) -> str:
    """Return the start label for one continuous idle trial.

    Kept as a compatibility alias for earlier analysis scripts.
    """

    return idle_trial_marker_label(source_block_id, segment_idx, role="START")


def idle_trial_marker_label(source_block_id: Any, trial_idx: int, *, role: str) -> str:
    """Return a compact label for one continuous idle-trial marker."""

    role_text = _normalize_key_text(role, default="marker").upper()
    return f"IDLE_{_normalize_key_text(source_block_id, default='source').upper()}_TRIAL{int(trial_idx):02d}_{role_text}"


def attach_condition_fields(
    row: MutableMapping[str, Any],
    condition: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Attach condition fields to a protocol or trial row."""

    row["condition_id"] = int(condition["condition_id"])
    row["condition_numeric_id"] = int(condition["condition_id"])
    row["condition_label"] = condition["condition_label"]
    row["condition_key"] = condition["condition_key"]
    row["frequency_hz"] = condition["frequency_hz"]
    row["stimulus_type"] = condition["stimulus_type"]
    row["stimulus_name"] = condition["stimulus_name"]
    row["stimulus_shape"] = condition["stimulus_shape"]
    row["luminance_mod_kind"] = condition["luminance_mod_kind"]
    row["luminance_depth"] = condition["luminance_depth"]
    row["start_marker"] = condition["start_marker"]
    row["stop_marker"] = condition["stop_marker"]

    # Backward-compatible aliases used by earlier exports and analysis code.
    row["stim_type"] = condition["stimulus_type"]
    row["stim_shape"] = condition["stimulus_shape"]
    row["lum_mod_kind"] = condition["luminance_mod_kind"]
    row["lum_depth"] = condition["luminance_depth"]
    return row


def build_legacy_mapping_entries(
    legacy_rows: Sequence[Mapping[str, Any]],
    by_condition_key: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build rows that map older marker values to the new condition IDs."""

    seen: dict[tuple[str, Any, Any], dict[str, Any]] = {}
    for row in legacy_rows:
        key = _clean_text(row.get("condition_key"))
        if not key or key not in by_condition_key:
            continue
        condition = by_condition_key[key]
        legacy_condition_id = row.get("legacy_condition_id")
        legacy_start_marker = row.get("legacy_start_marker")
        legacy_stop_marker = row.get("legacy_stop_marker")
        lookup = (
            str(legacy_condition_id),
            legacy_start_marker,
            legacy_stop_marker,
        )
        seen[lookup] = {
            "legacy_condition_id": legacy_condition_id,
            "legacy_start_marker": legacy_start_marker,
            "legacy_stop_marker": legacy_stop_marker,
            "new_condition_id": condition["condition_id"],
            "new_condition_label": condition["condition_label"],
            "new_start_marker": condition["start_marker"],
            "new_stop_marker": condition["stop_marker"],
            "condition_key": condition["condition_key"],
        }
    return list(seen.values())


if __name__ == "__main__":
    demo_specs = [
        build_frequency_pretest_condition_spec(
            frequency_hz=8.57,
            luminance_mod_kind="square",
            luminance_depth=1.0,
        ),
        build_frequency_pretest_condition_spec(
            frequency_hz=12.0,
            luminance_mod_kind="square",
            luminance_depth=1.0,
        ),
    ]
    assignment = assign_condition_ids(demo_specs)

    print("Condition demo")
    for condition in assignment["conditions"]:
        print(
            f"C{condition['condition_id']:03d}: "
            f"start={condition['start_marker']}, "
            f"stop={condition['stop_marker']}, "
            f"label={condition['condition_label']}"
        )
    print(f"Target 0 cue marker: {multi_target_cue_marker(0)}")
    print(f"Idle angle 0 marker: {idle_angle_marker(0)}")

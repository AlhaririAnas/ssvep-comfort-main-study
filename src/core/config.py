from __future__ import annotations

"""
Central configuration for the experiment software.

This module only reads text configuration. It does not open PsychoPy windows,
connect to Cortex, start recordings, start streams, or print during import.
"""

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - used only before dependencies are installed
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing or invalid."""


@dataclass(frozen=True)
class ProjectSettings:
    """Basic project paths and proband defaults."""

    project_root: Path
    data_dir: Path
    config_dir: Path
    stimuli_json: Path
    proband_id: str
    headset_id: str


@dataclass(frozen=True)
class EmotivSettings:
    """Cortex credentials and headset selection."""

    client_id: str
    client_secret: str
    headset_id: str


@dataclass(frozen=True)
class DisplaySettings:
    """Display values used by PsychoPy setup code."""

    screen_id: int
    fullscreen: bool
    monitor_name: str
    refresh_rate_hz: float
    screen_size_px: tuple[int, int]
    screen_width_cm: float
    viewing_distance_cm: float


def load_project_env(project_root: Path | None = None) -> Path | None:
    """Load the first project .env file that exists and return its path."""

    root = project_root or PROJECT_ROOT
    candidates = (
        root / ".env",
        root / ".env.local",
    )
    for candidate in candidates:
        if candidate.is_file():
            if load_dotenv is not None:
                load_dotenv(candidate, override=False)
            return candidate
    return None


LOADED_ENV_FILE = load_project_env()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {value!r}") from exc


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return float(default)
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {value!r}") from exc


def _env_path(key: str, default: str | Path) -> Path:
    return resolve_project_path(_env_str(key, str(default)))


def _env_csv_floats(key: str, default: str) -> list[float]:
    value = _env_str(key, default)
    try:
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ConfigError(f"{key} must contain comma-separated numbers") from exc


def _env_csv_ints(key: str, default: str) -> tuple[int, ...]:
    value = _env_str(key, default)
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ConfigError(f"{key} must contain comma-separated integers") from exc


def _env_csv_strings(key: str, default: str) -> tuple[str, ...]:
    value = _env_str(key, default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _env_rgb_tuple(key: str, default: str) -> tuple[float, float, float]:
    value = _env_str(key, default)
    try:
        parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ConfigError(f"{key} must contain three comma-separated numbers") from exc
    if len(parts) != 3:
        raise ConfigError(f"{key} must have exactly three values, got {value!r}")
    return parts[0], parts[1], parts[2]


def _env_position_pairs(key: str, default: str) -> tuple[tuple[float, float], ...]:
    """Parse x,y;x,y into position pairs."""

    value = _env_str(key, default)
    pairs: list[tuple[float, float]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            xy = [float(part.strip()) for part in chunk.split(",") if part.strip()]
        except ValueError as exc:
            raise ConfigError(f"{key} must contain numeric x,y pairs") from exc
        if len(xy) != 2:
            raise ConfigError(f"{key} pair {chunk!r} must have exactly two values")
        pairs.append((xy[0], xy[1]))
    return tuple(pairs)


def target_positions_from_spacing(spacing_deg: float) -> tuple[tuple[float, float], ...]:
    """Return the standard four-target cross layout in degrees."""

    spacing = float(spacing_deg)
    return (
        (-spacing, 0.0),
        (0.0, spacing),
        (spacing, 0.0),
        (0.0, -spacing),
    )


def _format_position_pairs(positions: tuple[tuple[float, float], ...]) -> str:
    return ";".join(f"{x},{y}" for x, y in positions)


def dva_to_px(angle_deg: float) -> float:
    """Convert visual angle in degrees to pixels for the configured screen."""

    size_cm = 2.0 * DIST_CM * math.tan(math.radians(float(angle_deg) / 2.0))
    px_per_cm = SCREEN_SIZE_PX[0] / SCREEN_W_CM
    return size_cm * px_per_cm


def require_emotiv_credentials() -> EmotivSettings:
    """Return Cortex credentials or raise a clear error."""

    missing = []
    if not EMOTIV_CLIENT_ID:
        missing.append("EMOTIV_CLIENT_ID")
    if not EMOTIV_CLIENT_SECRET:
        missing.append("EMOTIV_CLIENT_SECRET")
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required Cortex setting(s): {joined}")
    return get_emotiv_settings()


def load_json_config(path: str | Path) -> Any:
    """Load a JSON config file with a clear error message."""

    json_path = resolve_project_path(path)
    if not json_path.is_file():
        raise ConfigError(f"JSON config file not found: {json_path}")
    try:
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON config file: {json_path}") from exc


def load_json_dict(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file and require an object at the top level."""

    data = load_json_config(path)
    if not isinstance(data, dict):
        raise ConfigError(f"JSON config must be an object: {resolve_project_path(path)}")
    return data


def load_json_list(path: str | Path) -> list[Any]:
    """Load a JSON config file and require a list at the top level."""

    data = load_json_config(path)
    if not isinstance(data, list):
        raise ConfigError(f"JSON config must be a list: {resolve_project_path(path)}")
    return data


def validate_basic_config(require_emotiv: bool = False) -> None:
    """Validate values that can be checked without hardware."""

    if DEFAULT_DATA_DIR == PROJECT_ROOT:
        raise ConfigError("BCI_DATA_DIR must not point to the project root")
    if SCREEN_SIZE_PX[0] <= 0 or SCREEN_SIZE_PX[1] <= 0:
        raise ConfigError("Screen width and height in pixels must be positive")
    if SCREEN_W_CM <= 0:
        raise ConfigError("BCI_SCREEN_WIDTH_CM must be positive")
    if DIST_CM <= 0:
        raise ConfigError("BCI_VIEWING_DISTANCE_CM must be positive")
    if MONITOR_REFRESH_RATE <= 0:
        raise ConfigError("BCI_MONITOR_REFRESH_RATE must be positive")
    if DEFAULT_FIX_RANGE_S[0] < 0 or DEFAULT_FIX_RANGE_S[1] < DEFAULT_FIX_RANGE_S[0]:
        raise ConfigError("Fixation range must be positive and ordered")
    if DEFAULT_ITI_RANGE_S[0] < 0 or DEFAULT_ITI_RANGE_S[1] < DEFAULT_ITI_RANGE_S[0]:
        raise ConfigError("ITI range must be positive and ordered")
    if require_emotiv:
        require_emotiv_credentials()


def get_project_settings() -> ProjectSettings:
    """Return the current project settings."""

    return ProjectSettings(
        project_root=PROJECT_ROOT,
        data_dir=DEFAULT_DATA_DIR,
        config_dir=DEFAULT_CONFIG_DIR,
        stimuli_json=DEFAULT_STIMULI_JSON,
        proband_id=DEFAULT_PROBAND_ID,
        headset_id=DEFAULT_HEADSET_ID,
    )


def get_emotiv_settings() -> EmotivSettings:
    """Return Cortex settings without validating credentials."""

    return EmotivSettings(
        client_id=EMOTIV_CLIENT_ID,
        client_secret=EMOTIV_CLIENT_SECRET,
        headset_id=DEFAULT_HEADSET_ID,
    )


def get_display_settings() -> DisplaySettings:
    """Return display settings used by PsychoPy code."""

    return DisplaySettings(
        screen_id=DEFAULT_SCREEN_ID,
        fullscreen=DEFAULT_FULLSCREEN,
        monitor_name=MONITOR_NAME,
        refresh_rate_hz=MONITOR_REFRESH_RATE,
        screen_size_px=SCREEN_SIZE_PX,
        screen_width_cm=SCREEN_W_CM,
        viewing_distance_cm=DIST_CM,
    )


# General project settings
DEFAULT_CONFIG_DIR = _env_path("BCI_CONFIG_DIR", "configs")
DEFAULT_DATA_DIR = _env_path("BCI_DATA_DIR", "data")
DEFAULT_OUT_DIR = str(DEFAULT_DATA_DIR)
DEFAULT_STIMULI_JSON = _env_path("BCI_STIMULI_JSON", DEFAULT_CONFIG_DIR / "stimuli.json")

DEFAULT_PROBAND_ID = _env_str("BCI_PROBAND_ID", "P01")
DEFAULT_HEADSET_ID = _env_str("BCI_HEADSET_ID", "")

EMOTIV_CLIENT_ID = _env_str("EMOTIV_CLIENT_ID", "")
EMOTIV_CLIENT_SECRET = _env_str("EMOTIV_CLIENT_SECRET", "")


# Display settings
DEFAULT_SCREEN_ID = _env_int("BCI_SCREEN_ID", _env_int("BCI_SCREEN_INDEX", 1))
DEFAULT_FULLSCREEN = _env_bool("BCI_FULLSCREEN", True)
MONITOR_NAME = _env_str("BCI_MONITOR_NAME", "bci_monitor")
MONITOR_REFRESH_RATE = _env_float("BCI_MONITOR_REFRESH_RATE", 60.0)
SCREEN_SIZE_PX = (
    _env_int("BCI_SCREEN_WIDTH_PX", 1920),
    _env_int("BCI_SCREEN_HEIGHT_PX", 1080),
)
SCREEN_W_CM = _env_float("BCI_SCREEN_WIDTH_CM", 80.0)
DIST_CM = _env_float("BCI_VIEWING_DISTANCE_CM", 60.0)

TEXT_COLOR = _env_str("BCI_TEXT_COLOR", "white")
TEXT_HEIGHT_DEG = _env_float("BCI_TEXT_HEIGHT_DEG", 0.55)
TEXT_WRAP_DEG = _env_float("BCI_TEXT_WRAP_DEG", 18.0)


# Frequency pretest defaults
DEFAULT_FREQUENCIES_HZ = _env_csv_floats("BCI_FREQUENCIES_HZ", "7.5,8.57,10,12,15,20,30")
DEFAULT_TRIALS_PER_FREQ = _env_int("BCI_TRIALS_PER_FREQ", 5)
DEFAULT_STIM_DURATION_S = _env_float("BCI_STIM_DURATION_S", 5.0)
DEFAULT_BASELINE_OPEN_S = _env_float("BCI_BASELINE_OPEN_S", 30.0)

DEFAULT_FIX_RANGE_S = (
    _env_float("BCI_FIX_MIN_S", 1.5),
    _env_float("BCI_FIX_MAX_S", 2.5),
)
DEFAULT_ITI_RANGE_S = (
    _env_float("BCI_ITI_MIN_S", 1.0),
    _env_float("BCI_ITI_MAX_S", 2.0),
)


# Stimulus screening defaults
DEFAULT_TRIALS_PER_STIM = _env_int("BCI_TRIALS_PER_STIM", 5)
DEFAULT_BASELINE_CLOSED_S = _env_float("BCI_BASELINE_CLOSED_S", 60.0)
DEFAULT_INCLUDE_EYES_CLOSED_BASELINE = _env_bool("BCI_INCLUDE_EYES_CLOSED_BASELINE", False)
DEFAULT_BLOCK_BREAK_S = _env_float("BCI_BLOCK_BREAK_S", 10.0)
DEFAULT_LONG_BREAK_AFTER_BLOCKS = _env_csv_ints("BCI_LONG_BREAK_AFTER_BLOCKS", "7")
DEFAULT_LONG_BREAK_S = _env_float("BCI_LONG_BREAK_S", 180.0)

DEFAULT_LUM_MOD_KIND = _env_str("BCI_LUM_MOD_KIND", "square")
DEFAULT_LUM_DEPTH = _env_float("BCI_LUM_DEPTH", 1.0)


# Acquisition defaults
DEFAULT_ACQUISITION_MODE = _env_str("BCI_ACQUISITION_MODE", "record").lower()
DEFAULT_LIVE_STREAMS = _env_csv_strings("BCI_LIVE_STREAMS", "eeg,mot,pow,dev")
DEFAULT_LIVE_FLUSH_EVERY = _env_int("BCI_LIVE_FLUSH_EVERY", 25)
DEFAULT_LIVE_TEST_DURATION_S = _env_float("BCI_LIVE_TEST_DURATION_S", 15.0)


# Main study config files
DEFAULT_MS_SESSION_CONFIG = _env_path(
    "BCI_MS_SESSION_CONFIG",
    DEFAULT_CONFIG_DIR / "main_study_session.json",
)
DEFAULT_MS_STIMULI_JSON = _env_path(
    "BCI_MS_STIMULI_JSON",
    DEFAULT_CONFIG_DIR / "main_study_stimuli.json",
)
DEFAULT_MS_PROBAND_SCHEMA = _env_path(
    "BCI_MS_PROBAND_SCHEMA",
    DEFAULT_CONFIG_DIR / "main_study_proband_schema.json",
)
DEFAULT_MS_INTERFACE_LANGUAGE = _env_str("BCI_MS_INTERFACE_LANGUAGE", "en")


# Main study timing and layout
DEFAULT_MS_FREQUENCIES_HZ = _env_csv_floats("BCI_MS_FREQUENCIES_HZ", "8.57,12.0,15.0,20.0")
DEFAULT_MS_TARGET_SPACING_DEG = _env_float("BCI_MS_TARGET_SPACING_DEG", 9.0)
DEFAULT_MS_TARGET_POSITIONS_DEG = _env_position_pairs(
    "BCI_MS_TARGET_POSITIONS_DEG",
    _format_position_pairs(target_positions_from_spacing(DEFAULT_MS_TARGET_SPACING_DEG)),
)
DEFAULT_MS_TARGET_SIZE_DEG = _env_float("BCI_MS_TARGET_SIZE_DEG", 5.0)

_DEFAULT_MS_TRIALS_PER_TARGET = max(
    1,
    _env_int("BCI_MS_TRIALS_PER_BLOCK", 20) // max(1, len(DEFAULT_MS_FREQUENCIES_HZ)),
)
DEFAULT_MS_TRIALS_PER_TARGET = _env_int("BCI_MS_TRIALS_PER_TARGET", _DEFAULT_MS_TRIALS_PER_TARGET)
DEFAULT_MS_TRIALS_PER_BLOCK = _env_int(
    "BCI_MS_TRIALS_PER_BLOCK",
    DEFAULT_MS_TRIALS_PER_TARGET * len(DEFAULT_MS_FREQUENCIES_HZ),
)
DEFAULT_MS_TRIAL_DURATION_S = _env_float("BCI_MS_TRIAL_DURATION_S", 5.0)
DEFAULT_MS_CUE_DURATION_S = _env_float("BCI_MS_CUE_DURATION_S", 1.0)
DEFAULT_MS_FIX_RANGE_S = (
    _env_float("BCI_MS_FIX_MIN_S", 1.5),
    _env_float("BCI_MS_FIX_MAX_S", 2.5),
)
DEFAULT_MS_ITI_RANGE_S = (
    _env_float("BCI_MS_ITI_MIN_S", 1.0),
    _env_float("BCI_MS_ITI_MAX_S", 2.0),
)


# Main study baseline, breaks, and impedance gate
DEFAULT_MS_BASELINE_OPEN_S = _env_float("BCI_MS_BASELINE_OPEN_S", 30.0)
DEFAULT_MS_BASELINE_CLOSED_S = _env_float("BCI_MS_BASELINE_CLOSED_S", 60.0)
DEFAULT_MS_INCLUDE_EYES_CLOSED_BASELINE = _env_bool(
    "BCI_MS_INCLUDE_EYES_CLOSED_BASELINE",
    False,
)
DEFAULT_MS_BLOCK_BREAK_S = _env_float("BCI_MS_BLOCK_BREAK_S", 45.0)
DEFAULT_MS_LONG_BREAK_AFTER_BLOCKS = _env_csv_ints("BCI_MS_LONG_BREAK_AFTER_BLOCKS", "3,5")
DEFAULT_MS_LONG_BREAK_S = _env_float("BCI_MS_LONG_BREAK_S", 180.0)

DEFAULT_MS_IMPEDANCE_STABLE_S_INITIAL = _env_float("BCI_MS_IMPEDANCE_STABLE_S_INITIAL", 5.0)
DEFAULT_MS_IMPEDANCE_STABLE_S_BETWEEN = _env_float("BCI_MS_IMPEDANCE_STABLE_S_BETWEEN", 3.0)
DEFAULT_MS_IMPEDANCE_QUALITY_THRESHOLD = _env_int("BCI_MS_IMPEDANCE_QUALITY_THRESHOLD", 3)
DEFAULT_MS_IMPEDANCE_TIMEOUT_S = _env_float("BCI_MS_IMPEDANCE_TIMEOUT_S", 180.0)


# Main study idle block and counterbalancing
DEFAULT_MS_IDLE_ANGLES_DEG = _env_csv_floats(
    "BCI_MS_IDLE_ANGLES_DEG",
    "-30,-18,18,30,90,-90",
)
DEFAULT_MS_IDLE_TRIALS_PER_ANGLE = _env_int("BCI_MS_IDLE_TRIALS_PER_ANGLE", 1)
DEFAULT_MS_IDLE_TRIALS_PER_SOURCE_BLOCK = _env_int("BCI_MS_IDLE_TRIALS_PER_SOURCE_BLOCK", 6)
DEFAULT_MS_SEED_BASE = _env_int("BCI_MS_SEED_BASE", 42)


# Main study colors
DEFAULT_MS_BACKGROUND_COLOR = _env_rgb_tuple("BCI_MS_BACKGROUND_COLOR", "-1.0,-1.0,-1.0")
DEFAULT_MS_STIM_ON_COLOR = _env_rgb_tuple("BCI_MS_STIM_ON_COLOR", "1.0,1.0,1.0")
DEFAULT_MS_STIM_OFF_COLOR = _env_rgb_tuple("BCI_MS_STIM_OFF_COLOR", "-1.0,-1.0,-1.0")


if __name__ == "__main__":
    validate_basic_config(require_emotiv=False)
    project = get_project_settings()
    display = get_display_settings()
    print("Configuration demo")
    print(f"Project root: {project.project_root}")
    print(f"Data dir: {project.data_dir}")
    print(f"Config dir: {project.config_dir}")
    print(f"Stimuli JSON: {project.stimuli_json}")
    print(f"Participant ID: {project.proband_id}")
    print(f"Screen: {display.screen_size_px[0]} x {display.screen_size_px[1]} px")
    print(f"Refresh rate: {display.refresh_rate_hz:g} Hz")


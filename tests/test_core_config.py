from __future__ import annotations

from pathlib import Path

import pytest

from core import config


def test_resolve_project_path_keeps_absolute_path(tmp_path: Path) -> None:
    """Absolute paths should not be changed."""

    absolute_path = tmp_path / "file.txt"

    assert config.resolve_project_path(absolute_path) == absolute_path


def test_resolve_project_path_uses_project_root() -> None:
    """Relative paths should resolve below the project root."""

    resolved = config.resolve_project_path("configs/stimuli.json")

    assert resolved == config.PROJECT_ROOT / "configs" / "stimuli.json"


def test_target_positions_from_spacing_returns_standard_cross() -> None:
    """The four-target layout should be left, top, right, bottom."""

    assert config.target_positions_from_spacing(9.0) == (
        (-9.0, 0.0),
        (0.0, 9.0),
        (9.0, 0.0),
        (0.0, -9.0),
    )


def test_load_json_list_accepts_stimuli_config() -> None:
    """The default stimulus config should be a non-empty list."""

    stimuli = config.load_json_list(config.DEFAULT_STIMULI_JSON)

    assert stimuli
    assert isinstance(stimuli[0], dict)


def test_load_json_dict_accepts_main_study_session_config() -> None:
    """The main study session config should be a JSON object."""

    session_config = config.load_json_dict(config.DEFAULT_MS_SESSION_CONFIG)

    assert session_config["schema_version"] == "main_study_session_v4"
    assert isinstance(session_config["block_order"], list)
    assert session_config["block_order_strategy"] == "shuffle_first_n_keep_tail"
    assert session_config["block_order_lock_tail_count"] == 1


def test_main_study_plan_matches_final_block_structure() -> None:
    """The final main study uses seven active blocks and one paired idle block."""

    session_config = config.load_json_dict(config.DEFAULT_MS_SESSION_CONFIG)
    active_blocks = [block for block in session_config["block_order"] if block["stimulus_id"] is not None]
    idle_blocks = [block for block in session_config["block_order"] if block["stimulus_id"] is None]

    assert len(active_blocks) == 7
    assert len(idle_blocks) == 1
    assert session_config["idle_trials_per_source_block"] == 6
    assert session_config["idle_post_stop_hold_s"] == 0.1
    assert session_config["long_break_after_blocks"] == [3, 5]
    assert session_config["idle_target_positions_deg"] == [
        [-10.5, 10.5],
        [10.5, 10.5],
        [-10.5, -10.5],
        [10.5, -10.5],
    ]
    assert active_blocks[0]["stimulus_id"] == 311
    assert active_blocks[-1]["stimulus_id"] == 299


def test_load_json_dict_rejects_list_config() -> None:
    """A list JSON file should not be accepted as a dict config."""

    with pytest.raises(config.ConfigError, match="must be an object"):
        config.load_json_dict(config.DEFAULT_STIMULI_JSON)


def test_basic_config_validation_passes_without_emotiv_credentials() -> None:
    """Hardware credentials are optional for local config validation."""

    config.validate_basic_config(require_emotiv=False)

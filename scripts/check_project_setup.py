from __future__ import annotations

"""
Check the local project setup without hardware or PsychoPy windows.

This script validates configuration files, important folders, and safe imports.
It does not connect to Cortex, does not start recording, and does not open a
PsychoPy window.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


SAFE_IMPORTS = (
    "core.config",
    "core.conditions",
    "core.logging_utils",
    "core.session",
    "data_io.writers",
    "acquisition.recording",
    "acquisition.live_stream",
    "acquisition.impedance",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(
        description="Check local BCI project setup without hardware.",
    )
    parser.add_argument(
        "--require-emotiv",
        action="store_true",
        help="Require EMOTIV_CLIENT_ID and EMOTIV_CLIENT_SECRET in the environment.",
    )
    return parser


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _check_file(path: Path, label: str) -> bool:
    if path.is_file():
        _ok(f"{label}: {path}")
        return True
    _fail(f"{label} missing: {path}")
    return False


def _check_dir(path: Path, label: str) -> bool:
    if path.is_dir():
        _ok(f"{label}: {path}")
        return True
    _fail(f"{label} missing: {path}")
    return False


def _check_json(path: Path, label: str, expected_type: type) -> bool:
    if not _check_file(path, label):
        return False
    try:
        data = _load_json(path)
    except json.JSONDecodeError as exc:
        _fail(f"{label} invalid JSON: {exc}")
        return False
    if not isinstance(data, expected_type):
        _fail(f"{label} has wrong top-level type: expected {expected_type.__name__}")
        return False
    _ok(f"{label} JSON type: {expected_type.__name__}")
    return True


def _check_imports() -> bool:
    passed = True
    for module_name in SAFE_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            _fail(f"Import failed: {module_name} ({exc})")
            passed = False
        else:
            _ok(f"Import: {module_name}")
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run the project setup check."""

    parser = build_parser()
    args = parser.parse_args(argv)

    from core.config import (
        DEFAULT_CONFIG_DIR,
        DEFAULT_DATA_DIR,
        DEFAULT_MS_PROBAND_SCHEMA,
        DEFAULT_MS_SESSION_CONFIG,
        DEFAULT_MS_STIMULI_JSON,
        DEFAULT_STIMULI_JSON,
        LOADED_ENV_FILE,
        PROJECT_ROOT as CONFIG_PROJECT_ROOT,
        validate_basic_config,
    )

    passed = True

    print("Project setup check")
    print(f"Project root: {PROJECT_ROOT}")

    if CONFIG_PROJECT_ROOT != PROJECT_ROOT:
        _fail(f"Config project root mismatch: {CONFIG_PROJECT_ROOT}")
        passed = False
    else:
        _ok("Config project root matches script project root")

    if LOADED_ENV_FILE is None:
        _ok("No .env file loaded; using built-in defaults")
    else:
        _ok(f"Loaded env file: {LOADED_ENV_FILE}")

    try:
        validate_basic_config(require_emotiv=args.require_emotiv)
    except Exception as exc:
        _fail(f"Basic config validation failed: {exc}")
        passed = False
    else:
        _ok("Basic config validation")

    passed = _check_dir(SRC_DIR, "Source directory") and passed
    passed = _check_dir(DEFAULT_CONFIG_DIR, "Config directory") and passed
    passed = _check_dir(DEFAULT_DATA_DIR, "Data directory") and passed

    passed = _check_json(DEFAULT_STIMULI_JSON, "Stimuli config", list) and passed
    passed = _check_json(DEFAULT_MS_STIMULI_JSON, "Main study stimuli config", list) and passed
    passed = _check_json(DEFAULT_MS_SESSION_CONFIG, "Main study session config", dict) and passed
    passed = _check_json(DEFAULT_MS_PROBAND_SCHEMA, "Main study proband schema", dict) and passed

    passed = _check_imports() and passed

    if passed:
        print("Setup check passed.")
        return 0

    print("Setup check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

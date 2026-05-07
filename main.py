from __future__ import annotations

"""
Central command line launcher for the BCI project.

This file only routes commands to the small script entry points. It does not
contain marker, recording, stimulus, questionnaire, or experiment logic.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


COMMAND_MODULES = {
    "setup-check": "check_project_setup",
    "impedance-check": "run_impedance_check",
    "frequency-pretest": "run_frequency_pretest",
    "stimulus-screening": "run_stimulus_screening",
    "preview-stimuli": "preview_stimuli",
    "main-study": "run_main_study",
    "idle-test": "run_idle_block_test",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""

    parser = argparse.ArgumentParser(
        description="Central launcher for the low-amplitude visual SSVEP project.",
    )
    parser.add_argument(
        "command",
        choices=tuple(COMMAND_MODULES),
        help="Command to run. Use '<command> --help' for command-specific options.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a project command."""

    parser = build_parser()
    parsed = parser.parse_args(argv)

    module_name = COMMAND_MODULES[parsed.command]
    module = __import__(module_name)
    command_main = getattr(module, "main")
    return int(command_main(parsed.args))


if __name__ == "__main__":
    raise SystemExit(main())

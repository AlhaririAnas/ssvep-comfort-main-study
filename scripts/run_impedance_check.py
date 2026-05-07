from __future__ import annotations

"""
Manual impedance check for the Emotiv headset.

This script opens a Cortex session and shows the impedance gate UI. It does
not start recording, does not inject markers, and does not run an experiment.
"""

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import (  # noqa: E402
    DEFAULT_FULLSCREEN,
    DEFAULT_HEADSET_ID,
    DEFAULT_MS_BACKGROUND_COLOR,
    DEFAULT_MS_IMPEDANCE_QUALITY_THRESHOLD,
    DEFAULT_MS_IMPEDANCE_STABLE_S_INITIAL,
    DEFAULT_MS_IMPEDANCE_TIMEOUT_S,
    DEFAULT_SCREEN_ID,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(
        description="Run a manual headset impedance check without recording.",
    )
    parser.add_argument("--headset-id", default=DEFAULT_HEADSET_ID, help="Cortex headset ID.")
    parser.add_argument("--screen-id", type=int, default=DEFAULT_SCREEN_ID, help="PsychoPy screen index.")
    parser.add_argument("--windowed", action="store_true", help="Use a window instead of fullscreen.")
    parser.add_argument(
        "--stable-s",
        type=float,
        default=DEFAULT_MS_IMPEDANCE_STABLE_S_INITIAL,
        help="Required stable duration in seconds.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=DEFAULT_MS_IMPEDANCE_QUALITY_THRESHOLD,
        help="Minimum Cortex contact quality per channel.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_MS_IMPEDANCE_TIMEOUT_S,
        help="Maximum wait time in seconds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the manual impedance check."""

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("bci.impedance_check")

    from acquisition.impedance import run_impedance_gate
    from acquisition.marker import Marker
    from psychopy import core
    from stimuli.generator import make_window

    marker: Marker | None = None
    win = None
    try:
        marker = Marker(debug_mode=False)
        marker.connect(headset_id=args.headset_id)
        log.info("Waiting for Cortex session")
        if not marker.wait_for_session(timeout=90.0):
            raise RuntimeError("Cortex session could not be established.")

        win = make_window(
            fullscr=(DEFAULT_FULLSCREEN and not args.windowed),
            screen=args.screen_id,
            background_color=DEFAULT_MS_BACKGROUND_COLOR,
        )
        result = run_impedance_gate(
            win,
            marker.c,
            stable_s=args.stable_s,
            quality_threshold=args.quality_threshold,
            timeout_s=args.timeout_s,
            mode_label="manual_check",
        )

        print(f"Passed: {result.get('passed')}")
        print(f"Forced: {result.get('forced')}")
        print(f"Timeout: {result.get('timeout')}")
        print(f"Aborted: {result.get('aborted')}")
        print(f"Elapsed seconds: {result.get('elapsed_s')}")
        return 0 if result.get("passed") or result.get("forced") else 1
    finally:
        if win is not None:
            try:
                win.close()
            except Exception:
                log.exception("Window close failed")
        if marker is not None:
            marker.close()
        try:
            core.quit()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""
Connection recovery helpers for experiment runners.

The helpers keep recovery logic out of marker and recording modules.
They do not change marker timing. They only pause the experiment, wait for
the headset, run an impedance gate, and let the caller repeat the trial.
"""

import logging
import time
from typing import Any, Callable, Dict


logger = logging.getLogger(__name__)


class TrialConnectionLost(RuntimeError):
    """Raised when the headset disconnects during an active trial."""


def headset_is_connected(marker: Any) -> bool:
    """Return the current headset connection state from the marker owner."""

    if marker is None:
        return True
    checker = getattr(marker, "is_headset_connected", None)
    if checker is None:
        return True
    return bool(checker())


def raise_if_disconnected(marker: Any) -> None:
    """Raise TrialConnectionLost if Cortex reported a headset disconnect."""

    if not headset_is_connected(marker):
        raise TrialConnectionLost("Headset connection was lost during the trial.")


def show_recovery_message(win: Any, info_stim: Any, message: str) -> None:
    """Show a persistent recovery message without closing the PsychoPy window."""

    info_stim.text = message
    info_stim.draw()
    win.flip()


def recover_after_disconnect(
    *,
    win: Any,
    info_stim: Any,
    marker: Any,
    marker_cortex: Any,
    run_impedance_gate: Callable[..., Dict[str, Any]],
    stable_s: float,
    quality_threshold: int,
    timeout_s: float,
    mode_label: str,
    monitor: Any = None,
    on_event: Callable[[str, Dict[str, Any]], None] | None = None,
    log: logging.Logger | None = None,
) -> Dict[str, Any]:
    """Wait for reconnection, re-sync, run impedance, and return the gate result."""

    run_log = log or logger
    show_recovery_message(
        win,
        info_stim,
        "Headset connection lost.\n\n"
        "Please contact the experiment operator.\n"
        "The current trial will be repeated after reconnection.",
    )
    run_log.error("Headset connection lost. Waiting for reconnection.")

    while not marker.wait_for_headset_reconnected(timeout=1.0):
        show_recovery_message(
            win,
            info_stim,
            "Headset connection lost.\n\n"
            "Please contact the experiment operator.\n"
            "Waiting for the headset to reconnect ...",
        )
        time.sleep(0.1)

    show_recovery_message(
        win,
        info_stim,
        "Headset connected again.\n\n"
        "Checking signal quality before the trial is repeated ...",
    )
    run_log.warning("Headset reconnected. Re-syncing marker clock.")
    try:
        marker.resync_clock(timeout=10.0)
    except Exception:
        run_log.exception("Clock re-sync after reconnect failed")

    result = run_impedance_gate(
        win,
        marker_cortex,
        stable_s=stable_s,
        quality_threshold=quality_threshold,
        timeout_s=timeout_s,
        mode_label=mode_label,
        monitor=monitor,
        on_event=on_event,
    )
    return result


if __name__ == "__main__":
    print("Recovery module demo")
    print("This demo does not connect to Cortex and does not open a PsychoPy window.")

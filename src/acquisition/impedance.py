from __future__ import annotations

"""
impedance.py
============
Live electrode-contact-quality visualisation and stability gate for the
Emotiv EPOC X 14-channel headset, rendered in a PsychoPy window.

Usage
-----
    from acquisition.impedance import run_impedance_gate

    result = run_impedance_gate(
        win,
        cortex_client,                # an opened Cortex client
        stable_s=5.0,                 # how long all channels must stay >= threshold
        quality_threshold=3,          # Cortex contactQuality level 0..4
        timeout_s=180.0,              # hard cap, operator intervention after this
        mode_label="initial",         # arbitrary string for logs / protocol
    )
    # result: {"passed": bool, "elapsed_s": float, "final_quality": {label: q}, ...}

Design
------
- Subscribes to the Cortex `dev` stream (if not already subscribed), binds
  to `new_dev_data`, and reads `contactQuality` per channel.
- Renders a schematic head (circle) with 14 electrode circles at
  approximate 10-20 positions; each circle is coloured 0 (dark red) -> 4
  (green). A progress bar fills while all channels stay >= threshold and
  resets to zero the moment any channel drops below.
- Keyboard controls:
    * `escape` aborts the gate (returns `passed=False, aborted=True`).
    * `space`  force-accepts the gate (operator override; logged).
    * `enter`/`return` confirms once the gate is reached (required after
      the bar is full so the operator can visually verify).

This module intentionally avoids importing psychopy at module load (tests
and non-GUI callers can still import `run_impedance_gate` metadata).
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# 14 EPOC X channels in canonical Cortex order.
# (Cortex emits `dev` labels matching these. The code still checks labels
#  and only falls back to this list if labels are unavailable.)
EPOC_X_CHANNELS: Tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)

# Approximate 2D (x, y) positions on a unit circle head, viewed from above,
# nose at +y. Values drawn from the standard 10-20 layout projected onto
# the scalp sphere; fine enough for a schematic UI.
EPOC_X_POSITIONS: Dict[str, Tuple[float, float]] = {
    "AF3": (-0.33,  0.85),
    "AF4": ( 0.33,  0.85),
    "F7":  (-0.80,  0.58),
    "F3":  (-0.35,  0.55),
    "F4":  ( 0.35,  0.55),
    "F8":  ( 0.80,  0.58),
    "FC5": (-0.55,  0.30),
    "FC6": ( 0.55,  0.30),
    "T7":  (-0.95,  0.00),
    "T8":  ( 0.95,  0.00),
    "P7":  (-0.75, -0.55),
    "P8":  ( 0.75, -0.55),
    "O1":  (-0.30, -0.85),
    "O2":  ( 0.30, -0.85),
}

# PsychoPy `rgb` tuples in [-1, 1]. A 5-step gradient for contactQuality 0..4.
QUALITY_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (-0.4, -1.0, -1.0),  # 0 = no contact (dark red)
    ( 1.0, -1.0, -1.0),  # 1 = very poor (red)
    ( 1.0,  0.2, -1.0),  # 2 = poor (orange)
    ( 0.7,  1.0, -1.0),  # 3 = fair (yellow-green)
    (-0.5,  1.0, -0.5),  # 4 = good (green)
)

QUALITY_MIN = 0
QUALITY_MAX = 4


class ImpedanceMonitor:
    """Threadsafe collector of the latest per-channel contactQuality values.

    Binds to a Cortex client's `new_dev_data` and `new_data_labels` signals.
    """

    def __init__(self, cortex_client: Any, log: Optional[logging.Logger] = None) -> None:
        self.client = cortex_client
        self.log = log or logger
        self._lock = threading.Lock()
        self._labels: List[str] = list(EPOC_X_CHANNELS)
        self._labels_from_cortex: Optional[List[str]] = None
        self._last_quality: Dict[str, int] = {c: 0 for c in EPOC_X_CHANNELS}
        self._last_update_monotonic: float = 0.0
        self._last_signal: Optional[float] = None
        self._last_battery: Optional[float] = None
        self._bound = False

    def start(self) -> None:
        if self._bound:
            return
        self.client.bind(
            new_dev_data=self._on_dev_data,
            new_data_labels=self._on_labels,
        )
        self._bound = True
        try:
            self.client.sub_request(["dev"])
        except Exception:
            self.log.exception("Cortex dev subscribe failed")

    def stop(self) -> None:
        if not self._bound:
            return
        try:
            self.client.unsub_request(["dev"])
        except Exception:
            self.log.debug("Cortex dev unsubscribe failed (non-fatal)")
        self._bound = False

    # --- callbacks -------------------------------------------------------

    def _on_labels(self, *args: Any, **kwargs: Any) -> None:
        data = kwargs.get("data") or {}
        if str(data.get("streamName", "")).lower() != "dev":
            return
        labels = [str(x) for x in (data.get("labels") or [])]
        if labels:
            with self._lock:
                self._labels_from_cortex = labels
                for label in labels:
                    self._last_quality.setdefault(label, 0)

    def _on_dev_data(self, *args: Any, **kwargs: Any) -> None:
        data = kwargs.get("data") or {}
        values = data.get("dev") or []
        if not values:
            return
        with self._lock:
            labels = self._labels_from_cortex or self._labels
            n = min(len(labels), len(values))
            for i in range(n):
                label = labels[i]
                try:
                    q = int(values[i])
                except (TypeError, ValueError):
                    continue
                q = max(QUALITY_MIN, min(QUALITY_MAX, q))
                self._last_quality[label] = q
            self._last_update_monotonic = time.monotonic()
            if "signal" in data:
                try:
                    self._last_signal = float(data["signal"])
                except (TypeError, ValueError):
                    pass
            if "batteryPercent" in data:
                try:
                    self._last_battery = float(data["batteryPercent"])
                except (TypeError, ValueError):
                    pass

    # --- snapshot --------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "quality": dict(self._last_quality),
                "last_update_monotonic": self._last_update_monotonic,
                "signal": self._last_signal,
                "battery": self._last_battery,
                "labels": list(self._labels_from_cortex or self._labels),
            }


def _quality_color(q: int) -> Tuple[float, float, float]:
    q = max(QUALITY_MIN, min(QUALITY_MAX, int(q)))
    return QUALITY_COLORS[q]


def _min_quality(snapshot: Dict[str, Any]) -> int:
    qs = snapshot.get("quality") or {}
    if not qs:
        return 0
    return int(min(qs.values()))


def run_impedance_gate(
    win: Any,
    cortex_client: Any,
    *,
    stable_s: float = 5.0,
    quality_threshold: int = 3,
    timeout_s: float = 180.0,
    mode_label: str = "initial",
    head_radius_deg: float = 8.5,
    monitor: Optional[ImpedanceMonitor] = None,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Block until contactQuality >= `quality_threshold` on all channels for `stable_s`.

    Parameters
    ----------
    win : psychopy.visual.Window
        Fullscreen or windowed PsychoPy window. The function draws directly
        to it; the caller is responsible for clearing / flipping afterwards.
    cortex_client : Cortex
        An opened Cortex client (create_session() already called).
    stable_s : float
        Required stable duration above threshold.
    quality_threshold : int
        Required contactQuality per channel (0..4). 3 = yellow-green.
    timeout_s : float
        Hard cap. If not met by then, returns `passed=False` with `timeout=True`.
    mode_label : str
        Free-form label used in logs / result dict ("initial", "between", ...).
    head_radius_deg : float
        Radius of the schematic head. Electrode positions scale from this.
    monitor : ImpedanceMonitor, optional
        Reuse an existing monitor (e.g. across blocks). If None, a fresh one
        is created, started, and stopped automatically.
    on_event : callable, optional
        `on_event(name, payload)` is called for "started", "stable_tick",
        "dropped", "passed", "timeout", "aborted", "force_accepted".
        Useful for marker injection into the EEG stream.
    """
    # Imports deferred so the module stays importable in test environments.
    from psychopy import visual, event  # type: ignore

    owns_monitor = monitor is None
    if monitor is None:
        monitor = ImpedanceMonitor(cortex_client)
        monitor.start()

    try:
        return _run_impedance_gate_core(
            win,
            visual,
            event,
            monitor,
            stable_s=float(stable_s),
            quality_threshold=int(quality_threshold),
            timeout_s=float(timeout_s),
            mode_label=str(mode_label),
            head_radius_deg=float(head_radius_deg),
            on_event=on_event,
        )
    finally:
        if owns_monitor:
            monitor.stop()


def _run_impedance_gate_core(
    win: Any,
    visual: Any,
    event: Any,
    monitor: ImpedanceMonitor,
    *,
    stable_s: float,
    quality_threshold: int,
    timeout_s: float,
    mode_label: str,
    head_radius_deg: float,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]],
) -> Dict[str, Any]:
    head = visual.Circle(
        win, radius=head_radius_deg, edges=96, lineColor="white",
        fillColor=None, lineWidth=2.5, units="deg",
    )
    nose = visual.Polygon(
        win, edges=3, radius=head_radius_deg * 0.09,
        fillColor="white", lineColor="white",
        pos=(0.0, head_radius_deg + head_radius_deg * 0.05), units="deg",
        ori=0.0,
    )

    electrode_circles: Dict[str, Any] = {}
    electrode_labels: Dict[str, Any] = {}
    electrode_numbers: Dict[str, Any] = {}
    electrode_radius_deg = head_radius_deg * 0.11

    for name, (nx, ny) in EPOC_X_POSITIONS.items():
        pos = (nx * head_radius_deg * 0.90, ny * head_radius_deg * 0.90)
        electrode_circles[name] = visual.Circle(
            win, radius=electrode_radius_deg, edges=48,
            lineColor="white", fillColor=QUALITY_COLORS[0], lineWidth=1.5,
            pos=pos, units="deg",
        )
        electrode_labels[name] = visual.TextStim(
            win, text=name, height=electrode_radius_deg * 0.55,
            color="white", pos=(pos[0], pos[1] + electrode_radius_deg * 1.25),
            units="deg",
        )
        electrode_numbers[name] = visual.TextStim(
            win, text="0", height=electrode_radius_deg * 0.9,
            color="black", pos=pos, units="deg", bold=True,
        )

    title = visual.TextStim(
        win,
        text=f"Impedance check ({mode_label}) - press gently until all electrodes are green",
        height=0.6, color="white", pos=(0.0, head_radius_deg + 2.0),
        units="deg", wrapWidth=28.0,
    )
    instruction = visual.TextStim(
        win,
        text=(
            f"Required: all {len(EPOC_X_POSITIONS)} channels at quality >= {quality_threshold} "
            f"(yellow-green) for {stable_s:.1f} s.\n"
            "space = force accept   |   enter = start when full   |   escape = abort"
        ),
        height=0.5, color="white", pos=(0.0, -(head_radius_deg + 2.6)),
        units="deg", wrapWidth=28.0,
    )

    bar_width = head_radius_deg * 2.0
    bar_height = 0.6
    bar_back = visual.Rect(
        win, width=bar_width, height=bar_height,
        lineColor="white", fillColor=(-0.6, -0.6, -0.6),
        pos=(0.0, -(head_radius_deg + 1.3)), units="deg",
    )
    bar_fill = visual.Rect(
        win, width=0.001, height=bar_height - 0.1,
        lineColor=None, fillColor=(-0.5, 1.0, -0.5),
        pos=(-bar_width / 2.0, -(head_radius_deg + 1.3)), units="deg",
    )
    status = visual.TextStim(
        win, text="", height=0.5, color="white",
        pos=(0.0, head_radius_deg + 1.3), units="deg",
    )

    def _emit(name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if on_event is None:
            return
        try:
            on_event(name, dict(payload or {}))
        except Exception:
            logger.exception("on_event callback failed | name=%s", name)

    event.clearEvents()
    start_mono = time.monotonic()
    stable_since: Optional[float] = None
    force_accepted = False
    aborted = False
    passed = False
    timed_out = False
    await_enter = False

    _emit("started", {
        "mode": mode_label,
        "stable_s": stable_s,
        "threshold": quality_threshold,
        "timeout_s": timeout_s,
    })

    while True:
        now = time.monotonic()
        snap = monitor.snapshot()
        qualities = snap.get("quality") or {}
        qmin = _min_quality(snap) if qualities else 0

        # Gate logic
        if qualities and qmin >= quality_threshold:
            if stable_since is None:
                stable_since = now
            hold = now - stable_since
            if hold >= stable_s and not await_enter:
                await_enter = True
                _emit("stable_reached", {"hold_s": hold, "mode": mode_label})
        else:
            if stable_since is not None and not await_enter:
                _emit("dropped", {"min_quality": qmin, "mode": mode_label})
            stable_since = None
            await_enter = False

        # Render
        for name, circ in electrode_circles.items():
            q = int(qualities.get(name, 0))
            circ.fillColor = _quality_color(q)
            electrode_numbers[name].text = str(q)
            electrode_numbers[name].color = "black" if q >= 3 else "white"

        if await_enter:
            status.text = "READY - press ENTER to start"
            status.color = (-0.5, 1.0, -0.5)
            fill_frac = 1.0
        elif stable_since is not None:
            fill_frac = min(1.0, (now - stable_since) / max(stable_s, 1e-6))
            status.text = f"Holding {now - stable_since:0.1f} / {stable_s:0.1f} s"
            status.color = (0.7, 1.0, -1.0)
        else:
            fill_frac = 0.0
            status.text = f"Min quality {qmin}/4 - need all at {quality_threshold}+"
            status.color = (1.0, 0.2, -1.0)

        bar_fill.width = max(0.001, bar_width * float(fill_frac))
        bar_fill.pos = (-bar_width / 2.0 + bar_fill.width / 2.0, bar_fill.pos[1])

        head.draw()
        nose.draw()
        for name in EPOC_X_POSITIONS:
            electrode_circles[name].draw()
            electrode_labels[name].draw()
            electrode_numbers[name].draw()
        title.draw()
        instruction.draw()
        bar_back.draw()
        bar_fill.draw()
        status.draw()
        win.flip()

        # Keyboard
        for key in event.getKeys():
            if key in ("escape",):
                aborted = True
                _emit("aborted", {"mode": mode_label, "elapsed_s": time.monotonic() - start_mono})
                break
            if key in ("space",):
                force_accepted = True
                passed = True
                _emit("force_accepted", {"mode": mode_label, "min_quality": qmin})
                break
            if key in ("return", "num_enter") and await_enter:
                passed = True
                _emit("passed", {"mode": mode_label, "hold_s": now - (stable_since or now)})
                break
        if aborted or passed:
            break

        # Timeout
        if (time.monotonic() - start_mono) >= timeout_s:
            timed_out = True
            _emit("timeout", {"mode": mode_label, "elapsed_s": time.monotonic() - start_mono})
            break

        time.sleep(1.0 / 60.0)

    elapsed = time.monotonic() - start_mono
    final_snap = monitor.snapshot()
    result = {
        "mode": mode_label,
        "passed": bool(passed),
        "aborted": bool(aborted),
        "timeout": bool(timed_out),
        "force_accepted": bool(force_accepted),
        "elapsed_s": round(elapsed, 3),
        "quality_threshold": int(quality_threshold),
        "stable_s": float(stable_s),
        "final_quality": dict(final_snap.get("quality") or {}),
        "final_min_quality": _min_quality(final_snap),
        "battery_pct": final_snap.get("battery"),
    }
    logger.info(
        "Impedance gate | mode=%s | passed=%s | min=%s | elapsed=%.1fs | forced=%s",
        mode_label, result["passed"], result["final_min_quality"], elapsed, result["force_accepted"],
    )
    return result


__all__ = [
    "EPOC_X_CHANNELS",
    "EPOC_X_POSITIONS",
    "QUALITY_COLORS",
    "QUALITY_MIN",
    "QUALITY_MAX",
    "ImpedanceMonitor",
    "run_impedance_gate",
]


if __name__ == "__main__":
    class DemoClient:
        def bind(self, **kwargs: Any) -> None:
            self.callbacks = kwargs

        def sub_request(self, streams: List[str]) -> None:
            self.streams = streams

        def unsub_request(self, streams: List[str]) -> None:
            self.unsubscribed = streams

    client = DemoClient()
    monitor = ImpedanceMonitor(client)
    monitor.start()
    monitor._on_dev_data(data={"dev": [4] * len(EPOC_X_CHANNELS), "time": 1.0})
    snapshot = monitor.snapshot()
    monitor.stop()

    print("Impedance module demo")
    print("This demo does not open a PsychoPy window and does not connect to Cortex.")
    print(f"Channels: {len(EPOC_X_CHANNELS)}")
    print(f"Minimum quality: {_min_quality(snapshot)}")

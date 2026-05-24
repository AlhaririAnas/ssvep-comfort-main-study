"""
marker.py - precision marker library for the Emotiv Cortex API
===========================================================

Purpose
-------
Thread-safe wrapper around the Emotiv Cortex API for timing-critical
SSVEP / SSMVEP experiments with PsychoPy and the Emotiv EPOC X.

Handles:
  - session creation and headset-clock synchronisation
  - marker injection with sub-millisecond timestamp resolution
  - optional periodic re-sync for long sessions (>20 min)
  - live RTT and jitter statistics for QA

Clock synchronisation - how it works
--------------------------------------
cortex.py sends `syncWithHeadsetClock` with:
    systemTime   = time.time()      (Unix epoch seconds, wall clock)
    monotonicTime = time.monotonic() (arbitrary-start monotonic seconds)

Cortex computes the offset between the local monotonic clock and the
headset's sample clock (which is epoch-based) and returns:
    adjustment ~= time.time() - time.monotonic()   (+/- headset offset)

Confirmed from log:
    time.monotonic()      ~= 143 644 s   (system uptime)
    adjustment            ~= 1 775 358 032 s  (~2026-04-06 UTC epoch)
    -> marker timestamp    ~= 1 919 002 676 855 ms  (correct epoch ms)

Correct timestamp formula (the ONLY correct formula):
    marker_time_ms = (time.monotonic() + adjustment) * 1000

Do NOT use time.time() here - it would add the epoch offset twice.

PsychoPy flip coupling (critical for SSVEP precision)
-------------------------------------------------------
The timestamp must be captured at the earliest possible moment after
the physical pixel change, which is when win.flip() returns.
PsychoPy stores the exact flip time in win.lastFrameT (seconds,
monotonic-clock-based, measured by the OpenGL buffer-swap callback).

Recommended pattern in your stimulus loop:
    win.flip()
    ts_ms = marker.capture_flip_time_ms(win.lastFrameT)
    marker.inject_marker(value="1", label="TRIAL_ON", timestamp_ms=ts_ms)

If win.lastFrameT is unavailable (e.g. non-PsychoPy context):
    win.flip()
    ts_ms = marker.capture_marker_time_ms()   # captures monotonic NOW
    marker.inject_marker(value="1", label="TRIAL_ON", timestamp_ms=ts_ms)

Known systematic offsets (offline-correction only - do NOT bake in):
    EPOC X / EPOC+ at 128 Hz: ~60 ms anti-aliasing filter delay
        (Badcock et al., 2021, PMC7879951, Table 1)
    WebSocket API round-trip jitter: typically 10-40 ms
        -> use stats.roundtrip_times_ms() to measure in the session

References
----------
Emotiv Cortex API - syncWithHeadsetClock
    https://emotiv.gitbook.io/cortex-api/headset/syncwithheadsetclock
Emotiv Cortex API - injectMarker
    https://emotiv.gitbook.io/cortex-api/markers/injectmarker
Badcock, N. et al. (2021). It's all about time: precision and accuracy
    of Emotiv event-marking for ERP research. PMC7879951.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
try:
    from .cortex import Cortex
except ImportError:  # pragma: no cover - supports direct module demos
    from cortex import Cortex

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loading (looks next to this file, then one level up)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _env_candidate in (
    os.path.join(_SCRIPT_DIR, ".env"),
    os.path.join(os.path.dirname(_SCRIPT_DIR), ".env"),
):
    if os.path.isfile(_env_candidate):
        load_dotenv(_env_candidate)
        break


# ---------------------------------------------------------------------------
# MarkerStatistics
# ---------------------------------------------------------------------------
class MarkerStatistics:
    """
    Per-session marker quality tracker.

    Records:
      - local perf_counter at injection call time
      - local perf_counter at API confirmation callback time
      - the exact marker_time_ms that was sent to Cortex

    From these, round-trip times (RTT) are derived as a proxy for
    WebSocket latency + Cortex processing time.  RTT is NOT the same
    as timestamp error (the timestamp is captured before the WS send),
    but large or variable RTT indicates infrastructure problems.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: List[Dict] = []
        self._confirmations: List[float] = []   # perf_counter at callback

    def record_injection(
        self,
        idx: int,
        label: str,
        value: str,
        marker_time_ms: float,
        capture_method: str,
    ) -> float:
        """Called in the experiment thread immediately before WS send."""
        t = time.perf_counter()
        with self._lock:
            self._entries.append(
                {
                    "idx": idx,
                    "label": label,
                    "value": value,
                    "marker_time_ms": marker_time_ms,
                    "perf_counter_sent": t,
                    "capture_method": capture_method,
                    "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                }
            )
        return t

    def record_confirmation(self, marker_id: str, cortex_start_datetime: str) -> None:
        """Called in the WebSocket callback thread."""
        t = time.perf_counter()
        with self._lock:
            idx = len(self._confirmations)
            self._confirmations.append(t)
            # Annotate matching entry if it exists
            if idx < len(self._entries):
                self._entries[idx]["perf_counter_confirmed"] = t
                self._entries[idx]["cortex_marker_id"] = marker_id
                self._entries[idx]["cortex_start_datetime"] = cortex_start_datetime

    def roundtrip_times_ms(self) -> List[float]:
        """Return paired RTT values in milliseconds (sent -> confirmed)."""
        with self._lock:
            n = min(len(self._entries), len(self._confirmations))
            result = []
            for i in range(n):
                sent = self._entries[i].get("perf_counter_sent")
                conf = self._confirmations[i]
                if sent is not None:
                    result.append((conf - sent) * 1000.0)
            return result

    def inter_marker_intervals_ms(self) -> List[float]:
        """Return intervals between consecutive marker_time_ms values."""
        with self._lock:
            times = [e["marker_time_ms"] for e in self._entries]
        return [times[i + 1] - times[i] for i in range(len(times) - 1)]

    @property
    def log(self) -> List[Dict]:
        with self._lock:
            return list(self._entries)

    def summary(self) -> str:
        rtts = self.roundtrip_times_ms()
        if not rtts:
            return f"Markers sent: {len(self._entries)}, confirmed: {len(self._confirmations)}, no RTT pairs yet."
        mean_rtt = sum(rtts) / len(rtts)
        return (
            f"Markers sent={len(self._entries)} confirmed={len(self._confirmations)} | "
            f"RTT min/mean/max = {min(rtts):.1f}/{mean_rtt:.1f}/{max(rtts):.1f} ms"
        )


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------
class Marker:
    """
    Thread-safe, experiment-ready Emotiv Cortex marker controller.

    Typical PsychoPy experiment usage
    ----------------------------------
        marker = Marker()
        marker.connect()
        if not marker.wait_for_session(timeout=60):
            raise RuntimeError("Session setup timed out.")

        # --- in your stimulus loop ---
        win.flip()
        ts_ms = marker.capture_flip_time_ms(win.lastFrameT)
        marker.inject_marker(value="1", label="TRIAL_ON", timestamp_ms=ts_ms)

        # --- after last trial ---
        win.flip()
        ts_ms = marker.capture_flip_time_ms(win.lastFrameT)
        marker.inject_marker(value="0", label="TRIAL_OFF", timestamp_ms=ts_ms)

        marker.close()

    Long sessions (>20 min)
    -----------------------
    Clock drift between time.monotonic() and the headset clock can
    accumulate.  Call resync_clock() periodically, or enable the
    background timer:
        marker.periodic_resync_start(interval_s=600)   # every 10 min
        # ... experiment ...
        marker.periodic_resync_stop()
    """

    # EPOC X / EPOC+ at 128 Hz, firmware 3.x (Badcock et al. 2021, Table 1).
    # Documented for offline ERP correction - do NOT subtract from timestamps.
    EPOC_FILTER_DELAY_S: float = 0.060

    def __init__(self, debug_mode: bool = False) -> None:
        client_id = os.getenv("EMOTIV_CLIENT_ID", "").strip()
        client_secret = os.getenv("EMOTIV_CLIENT_SECRET", "").strip()
        if not client_id:
            raise ValueError(
                "EMOTIV_CLIENT_ID is missing. Add it to your .env file."
            )
        if not client_secret:
            raise ValueError(
                "EMOTIV_CLIENT_SECRET is missing. Add it to your .env file."
            )

        self.c = Cortex(client_id, client_secret, debug_mode=debug_mode)
        self.c.bind(
            create_session_done=self._on_create_session_done,
            sync_with_headset_clock_done=self._on_sync_with_headset_clock_done,
            inject_marker_done=self._on_inject_marker_done,
            update_marker_done=self._on_update_marker_done,
            headset_connected=self._on_headset_connected,
            headset_disconnected=self._on_headset_disconnected,
            new_eeg_data=self._on_stream_data,
            new_mot_data=self._on_stream_data,
            new_dev_data=self._on_stream_data,
            new_met_data=self._on_stream_data,
            new_pow_data=self._on_stream_data,
            new_sys_data=self._on_stream_data,
            inform_error=self._on_inform_error,
        )

        # Threading events
        self._session_ready = threading.Event()
        self._resync_done = threading.Event()
        self._headset_connected = threading.Event()
        self._headset_connected.set()

        # State - always access via property (protected by _state_lock)
        self._state_lock = threading.Lock()
        self._headset_clock_adjustment: float = 0.0
        self._sync_count: int = 0           # how many times clock was synced
        self._last_marker_id: str = ""
        self._last_marker_label: str = ""
        self._marker_counter: int = 0       # monotonic injection counter
        self._last_stream_monotonic: float = 0.0
        self.stream_timeout_s: float = 3.0

        # Periodic re-sync
        self._resync_timer: Optional[threading.Timer] = None

        self.stats = MarkerStatistics()

    # ------------------------------------------------------------------
    # Properties (all protected by _state_lock for thread safety)
    # ------------------------------------------------------------------
    @property
    def headset_clock_adjustment(self) -> float:
        with self._state_lock:
            return self._headset_clock_adjustment

    @headset_clock_adjustment.setter
    def headset_clock_adjustment(self, value: float) -> None:
        with self._state_lock:
            self._headset_clock_adjustment = float(value)

    @property
    def last_marker_id(self) -> str:
        with self._state_lock:
            return self._last_marker_id

    @property
    def sync_count(self) -> int:
        with self._state_lock:
            return self._sync_count

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "Marker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API - lifecycle
    # ------------------------------------------------------------------
    def connect(self, headset_id: str = "") -> None:
        """
        Start the Cortex WebSocket in a background daemon thread.

        cortex.py.open() blocks internally via websock_thread.join().
        Running it in a daemon thread keeps the main thread free for
        the experiment loop.
        """
        if headset_id:
            self.c.set_wanted_headset(headset_id)
        thread = threading.Thread(
            target=self.c.open,
            name="CortexWebSocketThread",
            daemon=True,
        )
        thread.start()
        logger.info("[Marker] WebSocket thread started.")

    def wait_for_session(self, timeout: float = 60.0) -> bool:
        """
        Block until session creation AND headset-clock sync are complete.

        Both steps are required before any marker timestamp is computed.
        Returns True on success, False on timeout.
        """
        logger.info("[Marker] Waiting for session + headset clock sync ...")
        ok = self._session_ready.wait(timeout=timeout)
        if ok:
            adj = self.headset_clock_adjustment
            logger.info(
                "[Marker] Session ready | adjustment=%.6f s | "
                "verified epoch_now=%.3f s | EPOC filter delay=%.0f ms",
                adj,
                time.monotonic() + adj,
                self.EPOC_FILTER_DELAY_S * 1000.0,
            )
        else:
            logger.error("[Marker] Session timeout after %.1f s.", timeout)
        return ok

    def close(self) -> None:
        """Close the Cortex WebSocket."""
        self.periodic_resync_stop()
        try:
            self.c.close()
        except Exception as exc:
            logger.error("[Marker] Error closing connection: %s", exc)

    def is_headset_connected(self) -> bool:
        """Return True if Cortex has not reported a headset disconnect."""

        if not self._headset_connected.is_set():
            return False
        with self._state_lock:
            last_stream = self._last_stream_monotonic
        if last_stream > 0.0 and (time.monotonic() - last_stream) > self.stream_timeout_s:
            self._headset_connected.clear()
            logger.error(
                "[Marker] Headset stream timeout after %.1f s without data.",
                self.stream_timeout_s,
            )
            return False
        return True

    def wait_for_headset_reconnected(self, timeout: float | None = None) -> bool:
        """Wait until Cortex reports the headset as connected again."""

        return self._headset_connected.wait(timeout=timeout)

    def request_reconnect_probe(self) -> None:
        """Ask Cortex to refresh the headset state while recovery is waiting."""

        try:
            self.c.refresh_headset_list()
            self.c.query_headset()
        except Exception:
            logger.exception("[Marker] Reconnect probe failed.")

    # ------------------------------------------------------------------
    # Public API - clock
    # ------------------------------------------------------------------
    def capture_marker_time_ms(self) -> float:
        """
        Return a synchronised marker timestamp in milliseconds NOW.

        Call this immediately after win.flip() returns when
        win.lastFrameT is not available.

        Formula:
            (time.monotonic() + headset_clock_adjustment) * 1000
        """
        return (time.monotonic() + self.headset_clock_adjustment) * 1000.0

    def capture_flip_time_ms(self, monotonic_time: float) -> float:
        """
        Convert a time.monotonic() value to a Cortex-compatible timestamp (ms).

        The argument MUST come from time.monotonic(), which is the same clock
        used by cortex.py's syncWithHeadsetClock calibration.

        DO NOT pass win.lastFrameT here.  PsychoPy's internal clock is based
        on time.perf_counter() starting from PsychoPy initialisation, while
        the adjustment was calibrated against time.monotonic() (system boot
        epoch).  On Windows these clocks have DIFFERENT epochs - mixing them
        produces timestamps ~system-uptime hours in the past, causing
        Cortex error -32115 ("timestamp too far in the past").

        Correct usage inside a callOnFlip callback:
            def _callback():
                t     = time.monotonic()          # <- must be this clock
                ts_ms = marker.capture_flip_time_ms(t)
                marker.inject_marker(..., timestamp_ms=ts_ms)
            win.callOnFlip(_callback)

        Returns:
            Marker timestamp in milliseconds, aligned to the headset clock.
        """
        return (monotonic_time + self.headset_clock_adjustment) * 1000.0

    def resync_clock(self, timeout: float = 10.0) -> bool:
        """
        Re-run headset-clock synchronisation and update the adjustment.

        Recommended for sessions longer than ~20 minutes to compensate
        for monotonic-clock vs headset-clock drift (typically <1 ms/min
        but cumulative over long sessions).

        Returns True if re-sync completed within timeout.
        """
        self._resync_done.clear()
        logger.info("[Marker] Re-syncing headset clock (sync #%d) ...", self.sync_count + 1)
        self.c.sync_with_headset_clock()
        ok = self._resync_done.wait(timeout=timeout)
        if not ok:
            logger.error("[Marker] Clock re-sync timeout after %.1f s.", timeout)
        return ok

    def periodic_resync_start(self, interval_s: float = 600.0) -> None:
        """
        Start a background timer that calls resync_clock() every interval_s.

        Default 600 s (10 min) is appropriate for sessions up to ~2 hours.
        Call periodic_resync_stop() before ending the experiment.
        """
        if interval_s <= 0:
            raise ValueError("interval_s must be positive.")
        self._resync_interval_s = interval_s
        self._schedule_resync()
        logger.info("[Marker] Periodic re-sync enabled every %.0f s.", interval_s)

    def periodic_resync_stop(self) -> None:
        """Cancel any pending periodic re-sync timer."""
        if self._resync_timer is not None and self._resync_timer.is_alive():
            self._resync_timer.cancel()
            self._resync_timer = None
            logger.info("[Marker] Periodic re-sync stopped.")

    # ------------------------------------------------------------------
    # Public API - markers
    # ------------------------------------------------------------------
    def inject_marker(
        self,
        value: str,
        label: str,
        port: str = "python_app",
        timestamp_ms: Optional[float] = None,
        capture_method: str = "auto",
        **kwargs,
    ) -> bool:
        """
        Inject an instance marker into the currently running record.

        Args:
            value:          Marker value string (e.g. stimulus ID).
            label:          Marker label string (e.g. "TRIAL_ON").
            port:           Source label shown in the Cortex export.
            timestamp_ms:   Pre-computed timestamp in ms (preferred).
                            Computed NOW if omitted.
            capture_method: Human-readable description of how the
                            timestamp was captured (for QA logging).
                            Defaults to "auto" when timestamp computed
                            here, or "flip" when win.lastFrameT was used.
            **kwargs:       Extra Cortex injectMarker parameters.

        Returns:
            True if the request was sent successfully, False otherwise.
        """
        # Guard: session must be ready
        if not self._session_ready.is_set():
            logger.error(
                "[Marker] inject_marker() called before session ready - "
                "label='%s' DISCARDED.",
                label,
            )
            return False

        # Capture timestamp
        if timestamp_ms is not None:
            t_ms = float(timestamp_ms)
            method = capture_method if capture_method != "auto" else "pre-captured"
        else:
            t_ms = self.capture_marker_time_ms()
            method = "auto (no pre-capture)"

        # Increment injection counter (thread-safe via GIL for int addition,
        # The lock keeps the request order correct on all platforms.
        with self._state_lock:
            idx = self._marker_counter
            self._marker_counter += 1

        self.stats.record_injection(idx, label, str(value), t_ms, method)

        try:
            self.c.inject_marker_request(t_ms, value, label, port=port, **kwargs)
            logger.debug(
                "[Marker] #%d sent | label='%s' | value='%s' | t=%.3f ms | method=%s",
                idx, label, value, t_ms, method,
            )
            return True
        except Exception as exc:
            logger.error("[Marker] inject_marker #%d ('%s') failed: %s", idx, label, exc)
            return False

    def update_marker(
        self,
        marker_id: str,
        timestamp_ms: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """
        Update a previously injected marker (e.g. to set its end time).

        Args:
            marker_id:    UUID returned by the inject_marker_done callback
                          (available as marker.last_marker_id after inject).
            timestamp_ms: End-time in ms.  Captured NOW if omitted.
        """
        if not marker_id:
            logger.error("[Marker] update_marker() requires a non-empty marker_id.")
            return False
        try:
            t_ms = float(timestamp_ms) if timestamp_ms is not None else self.capture_marker_time_ms()
            self.c.update_marker_request(marker_id, t_ms, **kwargs)
            logger.debug("[Marker] update sent | id=%s | t=%.3f ms", marker_id, t_ms)
            return True
        except Exception as exc:
            logger.error("[Marker] update_marker('%s') failed: %s", marker_id, exc)
            return False

    # ------------------------------------------------------------------
    # Callbacks (all run in the WebSocket thread)
    # ------------------------------------------------------------------
    def _on_create_session_done(self, *args, **kwargs) -> None:
        logger.info("[Marker] Session created -> syncing headset clock ...")
        self._headset_connected.set()
        self.c.sync_with_headset_clock()

    def _on_sync_with_headset_clock_done(self, *args, **kwargs) -> None:
        """
        Update headset_clock_adjustment from the Cortex response.

        The `data` dict from cortex.py._handle_sync_with_headset_clock
        contains the raw result_dic from the JSON-RPC response.
        Field name is "adjustment" per the Cortex API spec.
        """
        data = kwargs.get("data") or {}
        adj = float(data.get("adjustment", 0.0))

        with self._state_lock:
            old_adj = self._headset_clock_adjustment
            self._headset_clock_adjustment = adj
            self._sync_count += 1
            count = self._sync_count

        drift_ms = (adj - old_adj) * 1000.0 if count > 1 else 0.0
        logger.info(
            "[Marker] Clock sync #%d done | adjustment=%.6f s | "
            "epoch_now~=%.3f s | drift_since_last=%.3f ms",
            count, adj, time.monotonic() + adj, drift_ms,
        )

        self._resync_done.set()
        if not self._session_ready.is_set():
            self._session_ready.set()

    def _on_inject_marker_done(self, *args, **kwargs) -> None:
        data = kwargs.get("data") or {}
        marker_id = data.get("uuid", "")
        marker_label = data.get("label", "")
        cortex_start = data.get("startDatetime", "")

        # Update last marker info (thread-safe via lock)
        with self._state_lock:
            self._last_marker_id = marker_id
            self._last_marker_label = marker_label

        self.stats.record_confirmation(marker_id, cortex_start)
        logger.debug(
            "[Marker] Confirmed | id=%s | label='%s' | cortex_start=%s | %s",
            marker_id, marker_label, cortex_start, self.stats.summary(),
        )

    def _on_update_marker_done(self, *args, **kwargs) -> None:
        data = kwargs.get("data") or {}
        logger.debug(
            "[Marker] Update confirmed | id=%s | end=%s",
            data.get("uuid", ""),
            data.get("endDatetime", ""),
        )

    def _on_headset_connected(self, *args, **kwargs) -> None:
        self._headset_connected.set()
        with self._state_lock:
            self._last_stream_monotonic = time.monotonic()
        logger.warning("[Marker] Headset connected again: %s", kwargs.get("data"))

    def _on_headset_disconnected(self, *args, **kwargs) -> None:
        self._headset_connected.clear()
        self._session_ready.clear()
        logger.error("[Marker] Headset disconnected: %s", kwargs.get("data"))

    def _on_stream_data(self, *args, **kwargs) -> None:
        with self._state_lock:
            self._last_stream_monotonic = time.monotonic()
        if not self._headset_connected.is_set():
            self._headset_connected.set()
            logger.warning("[Marker] Headset stream data received again.")

    def _on_inform_error(self, *args, **kwargs) -> None:
        logger.error("[Marker] Cortex error: %s", kwargs.get("error_data"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _schedule_resync(self) -> None:
        """Schedule the next periodic re-sync (called recursively by timer)."""
        interval = getattr(self, "_resync_interval_s", 600.0)
        self._resync_timer = threading.Timer(interval, self._periodic_resync_tick)
        self._resync_timer.daemon = True
        self._resync_timer.start()

    def _periodic_resync_tick(self) -> None:
        """Timer callback: re-sync then reschedule."""
        if self._session_ready.is_set():
            self.resync_clock(timeout=10.0)
        self._schedule_resync()

if __name__ == "__main__":
    stats = MarkerStatistics()
    sent_at = stats.record_injection(
        idx=0,
        label="DEMO_MARKER",
        value="1",
        marker_time_ms=123456.0,
        capture_method="demo",
    )
    stats.record_confirmation("demo-marker-id", "demo-start-time")

    print("Marker module demo")
    print("This demo does not connect to Cortex and does not send markers.")
    print(f"Sent perf counter recorded: {sent_at > 0}")
    print(stats.summary())


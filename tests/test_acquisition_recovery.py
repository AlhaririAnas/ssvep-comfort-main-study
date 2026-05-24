from __future__ import annotations

from pathlib import Path

import pytest

from acquisition.cortex import Cortex, ERR_SESSION_DOES_NOT_EXIST
from acquisition.impedance import ImpedanceMonitor
from acquisition.live_stream import CortexLiveStream
from acquisition.recovery import recover_after_disconnect


class _FakeInfoStim:
    def __init__(self) -> None:
        self.text = ""

    def draw(self) -> None:
        return


class _FakeWindow:
    def __init__(self) -> None:
        self.flips = 0

    def flip(self) -> None:
        self.flips += 1


class _FakeMarker:
    def __init__(self) -> None:
        self.wait_calls = 0
        self.probe_calls = 0
        self.session_waits: list[float] = []
        self.resync_calls = 0

    def wait_for_headset_reconnected(self, timeout: float | None = None) -> bool:
        self.wait_calls += 1
        return self.wait_calls >= 2

    def request_reconnect_probe(self) -> None:
        self.probe_calls += 1

    def wait_for_session(self, timeout: float = 60.0) -> bool:
        self.session_waits.append(timeout)
        return True

    def resync_clock(self, timeout: float = 10.0) -> bool:
        self.resync_calls += 1
        return True


class _FakeCortexClient:
    def __init__(self) -> None:
        self.callbacks = {}
        self.subscribed: list[tuple[str, ...]] = []
        self.unsubscribed: list[tuple[str, ...]] = []

    def bind(self, **kwargs) -> None:
        self.callbacks.update(kwargs)

    def sub_request(self, streams) -> None:
        self.subscribed.append(tuple(streams))

    def unsub_request(self, streams) -> None:
        self.unsubscribed.append(tuple(streams))


class _FakeLiveMarker:
    def __init__(self) -> None:
        self.c = _FakeCortexClient()


def test_recovery_probes_and_waits_for_session_after_reconnect(monkeypatch) -> None:
    marker = _FakeMarker()
    gate_calls = []

    def _gate(*args, **kwargs):
        gate_calls.append(kwargs)
        return {"passed": True}

    monkeypatch.setattr("acquisition.recovery.time.sleep", lambda _: None)

    result = recover_after_disconnect(
        win=_FakeWindow(),
        info_stim=_FakeInfoStim(),
        marker=marker,
        marker_cortex=object(),
        run_impedance_gate=_gate,
        stable_s=1.0,
        quality_threshold=3,
        timeout_s=5.0,
        mode_label="test",
    )

    assert result == {"passed": True}
    assert marker.probe_calls == 1
    assert marker.session_waits == [30.0]
    assert marker.resync_calls == 1
    assert gate_calls[0]["mode_label"] == "test"


def test_live_stream_resubscribes_after_session_recreation(tmp_path: Path) -> None:
    marker = _FakeLiveMarker()
    stream = CortexLiveStream(marker, tmp_path, streams=("eeg", "dev"))
    stream._started = True

    marker.c.callbacks["create_session_done"](data="new-session")

    assert marker.c.subscribed == [("eeg", "dev")]


def test_impedance_monitor_resubscribes_after_session_recreation() -> None:
    client = _FakeCortexClient()
    monitor = ImpedanceMonitor(client)

    monitor.start()
    client.subscribed.clear()
    client.callbacks["create_session_done"](data="new-session")

    assert client.subscribed == [("dev",)]


def test_existing_cortex_session_is_reannounced_after_reconnect() -> None:
    cortex = Cortex("client-id", "client-secret")
    events = []
    cortex.session_id = "session-1"
    cortex.emit = lambda event, **kwargs: events.append((event, kwargs))

    with pytest.warns(UserWarning, match="There is existed session session-1"):
        cortex.create_session()

    assert events == [("create_session_done", {"data": "session-1"})]


def test_cortex_forgets_session_when_cortex_reports_it_missing() -> None:
    cortex = Cortex("client-id", "client-secret")
    events = []
    cortex.session_id = "old-session"
    cortex.emit = lambda event, **kwargs: events.append((event, kwargs))

    cortex.handle_error({"id": 6, "error": {"code": ERR_SESSION_DOES_NOT_EXIST}})

    assert cortex.session_id == ""
    assert events == [("inform_error", {"error_data": {"code": ERR_SESSION_DOES_NOT_EXIST}})]

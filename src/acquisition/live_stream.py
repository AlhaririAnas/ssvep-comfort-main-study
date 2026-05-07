from __future__ import annotations

"""
live_stream.py
==============
Live Cortex stream helper with background CSV writing.

Purpose
-------
- Subscribe to selected Cortex streams after a session is ready.
- Write one CSV per stream continuously in a background thread.
- Optionally embed local marker events directly into EEG rows.
- Save a compact summary JSON and a dedicated markers CSV on stop.

Notes on marker embedding
-------------------------
Marker events are registered with local flip-synchronised timestamps (ms).
For the EEG CSV, each marker is attached to the first EEG sample whose
`cortex_time * 1000` is greater than or equal to the marker timestamp.
This keeps the EEG CSV analysis-friendly while preserving a dedicated
marker log as a separate stream/file.
"""

import csv
import json
import logging
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamPacket:
    stream_name: str
    row: Dict[str, Any]


@dataclass
class PendingMarker:
    marker_value: int
    marker_label: str
    marker_time_ms: float
    marker_role: str = "event"


class LiveCsvWriter:
    """Background CSV writer with one file per stream."""

    def __init__(self, output_dir: Path, flush_every: int = 25, log: Optional[logging.Logger] = None) -> None:
        self.output_dir = Path(output_dir)
        self.flush_every = max(1, int(flush_every))
        self.log = log or logger

        self._queue: "queue.Queue[Optional[StreamPacket]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, name="LiveCsvWriter", daemon=True)

        self._files: Dict[str, Any] = {}
        self._writers: Dict[str, csv.DictWriter] = {}
        self._fieldnames: Dict[str, List[str]] = {}
        self._rows_since_flush: Dict[str, int] = defaultdict(int)
        self._count_by_stream: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log.info("LiveCsvWriter start | folder=%s", self.output_dir)
        self._thread.start()

    def stop(self) -> None:
        self.log.info("LiveCsvWriter stop requested")
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=10.0)
        for stream_name, fh in list(self._files.items()):
            try:
                fh.flush()
                fh.close()
                self.log.info("LiveCsvWriter closed | stream=%s", stream_name)
            except Exception:
                self.log.exception("LiveCsvWriter close failed | stream=%s", stream_name)

    def push_row(self, stream_name: str, row: Dict[str, Any]) -> None:
        self._queue.put(StreamPacket(stream_name=stream_name, row=row))

    def summary(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._count_by_stream)

    def write_summary(self, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "counts": self.summary(),
            "finished_at_unix": time.time(),
        }
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        self.log.info("LiveCsvWriter summary saved | path=%s", path)

    def _open_writer_if_needed(self, stream_name: str, row: Dict[str, Any]) -> csv.DictWriter:
        writer = self._writers.get(stream_name)
        if writer is not None:
            return writer

        path = self.output_dir / f"{stream_name}.csv"
        fieldnames = list(row.keys())
        fh = open(path, "w", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        self._files[stream_name] = fh
        self._writers[stream_name] = writer
        self._fieldnames[stream_name] = fieldnames
        self.log.info("LiveCsvWriter opened | stream=%s | path=%s | cols=%d", stream_name, path, len(fieldnames))
        return writer

    def _writer_loop(self) -> None:
        self.log.info("LiveCsvWriter thread started")
        while True:
            item = self._queue.get()
            if item is None:
                break

            writer = self._open_writer_if_needed(item.stream_name, item.row)
            writer.writerow(item.row)
            fh = self._files[item.stream_name]

            with self._lock:
                self._count_by_stream[item.stream_name] += 1
                self._rows_since_flush[item.stream_name] += 1
                count = self._count_by_stream[item.stream_name]
                if self._rows_since_flush[item.stream_name] >= self.flush_every:
                    fh.flush()
                    self._rows_since_flush[item.stream_name] = 0
                    self.log.debug("LiveCsvWriter flush | stream=%s | rows=%d", item.stream_name, count)

        for stream_name, fh in self._files.items():
            try:
                fh.flush()
                self.log.debug("LiveCsvWriter final flush | stream=%s", stream_name)
            except Exception:
                self.log.exception("LiveCsvWriter final flush failed | stream=%s", stream_name)
        self.log.info("LiveCsvWriter thread finished")


class CortexLiveStream:
    """Bind Cortex callbacks, persist live streams, and embed markers into EEG rows."""

    _SUPPORTED_STREAMS = ("eeg", "mot", "pow", "met", "dev", "sys")

    def __init__(
        self,
        marker: Any,
        output_dir: Path,
        streams: Iterable[str] = ("eeg", "mot", "pow", "dev"),
        flush_every: int = 25,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.marker = marker
        self.output_dir = Path(output_dir)
        self.streams = [str(s).lower() for s in streams]
        self.log = log or logger
        self.writer = LiveCsvWriter(self.output_dir, flush_every=flush_every, log=self.log)

        self.labels_by_stream: Dict[str, List[str]] = {}
        self._labels_ready = threading.Event()
        self._started = False

        self._pending_markers: Deque[PendingMarker] = deque()
        self._marker_lock = threading.Lock()
        self._marker_rows: List[Dict[str, Any]] = []

        unsupported = sorted(set(self.streams) - set(self._SUPPORTED_STREAMS))
        if unsupported:
            raise ValueError(f"Unsupported live streams: {unsupported}")

        self.marker.c.bind(
            new_data_labels=self._on_new_data_labels,
            new_eeg_data=self._on_eeg_data,
            new_mot_data=self._on_mot_data,
            new_pow_data=self._on_pow_data,
            new_met_data=self._on_met_data,
            new_dev_data=self._on_dev_data,
            new_sys_data=self._on_sys_data,
            inform_error=self._on_error,
        )

    def start(self, wait_labels_s: float = 10.0) -> None:
        if self._started:
            return
        self.log.info("Live stream start | streams=%s", ", ".join(self.streams))
        self.writer.start()
        self.marker.c.sub_request(self.streams)
        self._labels_ready.wait(timeout=max(0.0, float(wait_labels_s)))
        self._started = True
        self.log.info("Live stream subscribed")

    def stop(self) -> None:
        if not self._started:
            return
        self.log.info("Live stream stop requested")
        try:
            self.marker.c.unsub_request(self.streams)
            self.log.info("Live stream unsubscribe sent")
        except Exception:
            self.log.exception("Live stream unsubscribe failed")
        time.sleep(1.0)
        self._flush_remaining_markers_to_marker_stream()
        self.writer.stop()
        self.writer.write_summary(
            self.output_dir / "summary.json",
            extra={"streams": self.streams, "markers_registered": len(self._marker_rows)},
        )
        self._started = False
        self.log.info("Live stream stopped")

    def summary(self) -> Dict[str, int]:
        return self.writer.summary()

    def register_marker(
        self,
        *,
        marker_value: int,
        marker_label: str,
        marker_time_ms: float,
        marker_role: str = "event",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        row = {
            "local_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "local_monotonic": round(time.monotonic(), 6),
            "marker_value": marker_value,
            "marker_label": str(marker_label),
            "marker_time_ms": round(float(marker_time_ms), 3),
            "marker_role": marker_role,
        }
        if metadata:
            row.update(metadata)
        with self._marker_lock:
            self._pending_markers.append(
                PendingMarker(
                    marker_value=int(marker_value),
                    marker_label=str(marker_label),
                    marker_time_ms=float(marker_time_ms),
                    marker_role=str(marker_role),
                )
            )
            self._marker_rows.append(dict(row))
        self.writer.push_row("markers", row)
        self.log.info(
            "Live marker registered | value=%s | label=%s | t_ms=%.3f",
            marker_value,
            marker_label,
            marker_time_ms,
        )

    def _base_row(self, cortex_time: Any) -> Dict[str, Any]:
        return {
            "local_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "local_monotonic": round(time.monotonic(), 6),
            "cortex_time": cortex_time,
        }

    def _row_from_values(self, prefix: str, values: List[Any], cortex_time: Any) -> Dict[str, Any]:
        row = self._base_row(cortex_time)
        labels = self.labels_by_stream.get(prefix) or []
        if labels and len(labels) == len(values):
            for label, value in zip(labels, values):
                safe_label = str(label).replace("/", "_")
                row[f"{prefix}_{safe_label}"] = value
        else:
            for i, value in enumerate(values):
                row[f"{prefix}_{i}"] = value
        if prefix == "eeg":
            self._attach_markers_to_eeg_row(row)
        return row

    def _attach_markers_to_eeg_row(self, row: Dict[str, Any]) -> None:
        row_ms = None
        try:
            row_ms = float(row.get("cortex_time")) * 1000.0
        except Exception:
            row_ms = None

        values: List[str] = []
        labels: List[str] = []
        times: List[str] = []
        roles: List[str] = []
        if row_ms is not None:
            with self._marker_lock:
                while self._pending_markers and self._pending_markers[0].marker_time_ms <= row_ms:
                    marker = self._pending_markers.popleft()
                    values.append(str(marker.marker_value))
                    labels.append(marker.marker_label)
                    times.append(f"{marker.marker_time_ms:.3f}")
                    roles.append(marker.marker_role)

        row["marker_value"] = "|".join(values) if values else ""
        row["marker_label"] = "|".join(labels) if labels else ""
        row["marker_time_ms"] = "|".join(times) if times else ""
        row["marker_role"] = "|".join(roles) if roles else ""

    def _flush_remaining_markers_to_marker_stream(self) -> None:
        # Marker rows are already written to markers.csv on registration.
        # This method is intentionally a no-op placeholder for symmetry.
        return

    def _push(self, stream_name: str, row: Dict[str, Any]) -> None:
        if stream_name in self.streams:
            self.writer.push_row(stream_name, row)

    def _on_new_data_labels(self, *args, **kwargs) -> None:
        data = kwargs.get("data") or {}
        stream_name = str(data.get("streamName", "")).lower()
        labels = [str(x) for x in data.get("labels", [])]
        if stream_name:
            self.labels_by_stream[stream_name] = labels
            self._labels_ready.set()
            self.log.info("Live labels | stream=%s | columns=%d", stream_name, len(labels))

    def _on_eeg_data(self, *args, **kwargs) -> None:
        data = kwargs.get("data") or {}
        values = list(data.get("eeg") or [])
        self._push("eeg", self._row_from_values("eeg", values, data.get("time")))

    def _on_mot_data(self, *args, **kwargs) -> None:
        data = kwargs.get("data") or {}
        values = list(data.get("mot") or [])
        self._push("mot", self._row_from_values("mot", values, data.get("time")))

    def _on_pow_data(self, *args, **kwargs) -> None:
        data = kwargs.get("data") or {}
        values = list(data.get("pow") or [])
        self._push("pow", self._row_from_values("pow", values, data.get("time")))

    def _on_met_data(self, *args, **kwargs) -> None:
        data = kwargs.get("data") or {}
        values = list(data.get("met") or [])
        self._push("met", self._row_from_values("met", values, data.get("time")))

    def _on_dev_data(self, *args, **kwargs) -> None:
        data = kwargs.get("data") or {}
        values = list(data.get("dev") or [])
        self._push("dev", self._row_from_values("dev", values, data.get("time")))

    def _on_sys_data(self, *args, **kwargs) -> None:
        data = kwargs.get("data") or {}
        values = list(data.get("sys") or [])
        self._push("sys", self._row_from_values("sys", values, data.get("time")))

    def _on_error(self, *args, **kwargs) -> None:
        self.log.error("Live stream Cortex error | %s", kwargs.get("error_data"))


if __name__ == "__main__":
    demo_dir = Path("data") / "_demo_live_stream"
    writer = LiveCsvWriter(demo_dir, flush_every=1)
    writer.start()
    writer.push_row("eeg", {"cortex_time": 1.0, "eeg_AF3": 0.1, "marker_value": ""})
    writer.push_row("dev", {"cortex_time": 1.0, "dev_quality": 4})
    writer.stop()
    print("Live stream module demo")
    print("This demo does not connect to Cortex and does not subscribe to streams.")
    print(f"Demo row counts: {writer.summary()}")

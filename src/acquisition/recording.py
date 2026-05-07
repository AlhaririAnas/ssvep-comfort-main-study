from __future__ import annotations

"""
Cortex recording helper.

This module owns record start, record stop, post-processing wait, and export.
It does not send markers and does not know any stimulus or trial structure.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingExportSettings:
    """Export settings for one Cortex recording."""

    folder: str
    data_types: tuple[str, ...] = ("EEG", "MOTION", "PM", "BP")
    export_format: str = "CSV"
    version: str = "V2"


def generate_record_title(stim_name: str, freq: float | None = None) -> str:
    """Return a filesystem-safe Cortex record title."""

    now = time.localtime()
    safe_name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(stim_name))
    date_part = time.strftime("%d%m%Y", now)
    time_part = time.strftime("%H%M%S", now)
    if freq is None:
        return f"{safe_name}_{date_part}_{time_part}"
    return f"{safe_name}_{float(freq):.1f}Hz_{date_part}_{time_part}"


def _resolve_cortex_client(cortex_or_owner: Any) -> Any:
    """Return a Cortex client from a Cortex instance or an owner with `.c`."""

    return getattr(cortex_or_owner, "c", cortex_or_owner)


class CortexRecording:
    """Manage Cortex record start, stop, post-processing, and export."""

    def __init__(self, cortex_or_owner: Any, log: logging.Logger | None = None) -> None:
        self.cortex = _resolve_cortex_client(cortex_or_owner)
        self.log = log or logger

        self._record_started = threading.Event()
        self._record_stopped = threading.Event()
        self._export_done = threading.Event()
        self._state_lock = threading.Lock()

        self._record_id = ""
        self._export_settings = RecordingExportSettings(folder="")

        self.cortex.bind(
            create_record_done=self._on_create_record_done,
            stop_record_done=self._on_stop_record_done,
            warn_record_post_processing_done=self._on_warn_record_post_processing_done,
            export_record_done=self._on_export_record_done,
            inform_error=self._on_inform_error,
        )

    @property
    def record_id(self) -> str:
        """Return the last confirmed Cortex record ID."""

        with self._state_lock:
            return self._record_id

    def start_record(self, title: str, description: str = "", timeout: float = 15.0) -> bool:
        """Start a Cortex record and wait until Cortex confirms it."""

        if not title:
            raise ValueError("Record title must not be empty.")

        self._record_started.clear()
        self._record_stopped.clear()
        self._export_done.clear()
        self.cortex.create_record(title, description=description)

        ok = self._record_started.wait(timeout=timeout)
        if not ok:
            self.log.error("Record start timeout after %.1f s.", timeout)
        return ok

    def stop_and_export(
        self,
        export_folder: str | Path,
        data_types: list[str] | tuple[str, ...] | None = None,
        export_format: str = "CSV",
        version: str = "V2",
        timeout: float = 120.0,
    ) -> bool:
        """Stop the active record, wait for post-processing, and export it."""

        settings = RecordingExportSettings(
            folder=str(export_folder),
            data_types=tuple(data_types or ("EEG", "MOTION", "PM", "BP")),
            export_format=str(export_format),
            version=str(version),
        )
        with self._state_lock:
            self._export_settings = settings

        self._record_stopped.clear()
        self._export_done.clear()

        self.log.info("Stopping recording and waiting for Cortex post-processing.")
        self.cortex.stop_record()

        ok = self._export_done.wait(timeout=timeout)
        if ok:
            self.log.info("Recording export complete: %s", settings.folder)
        else:
            self.log.error("Recording export timeout after %.1f s.", timeout)
        return ok

    def wait_for_export(self, timeout: float = 30.0) -> bool:
        """Wait for a pending export to finish."""

        return self._export_done.wait(timeout=timeout)

    def _on_create_record_done(self, *args: Any, **kwargs: Any) -> None:
        data = kwargs.get("data") or {}
        with self._state_lock:
            self._record_id = data.get("uuid", "")
        self.log.info(
            "Record started | id=%s | title=%s | start=%s",
            self.record_id,
            data.get("title", ""),
            data.get("startDatetime", ""),
        )
        self._record_started.set()

    def _on_stop_record_done(self, *args: Any, **kwargs: Any) -> None:
        data = kwargs.get("data") or {}
        self.log.info(
            "Record stopped | id=%s | end=%s | waiting for post-processing.",
            data.get("uuid", self.record_id),
            data.get("endDatetime", ""),
        )
        self._record_stopped.set()

    def _on_warn_record_post_processing_done(self, *args: Any, **kwargs: Any) -> None:
        raw_data = kwargs.get("data")
        if isinstance(raw_data, str) and raw_data:
            record_id = raw_data
        elif isinstance(raw_data, dict):
            record_id = raw_data.get("uuid", "") or raw_data.get("recordId", "")
        else:
            record_id = ""

        if not record_id:
            record_id = self.record_id
            self.log.warning("Post-processing warning had no record ID. Using %s.", record_id)

        with self._state_lock:
            settings = self._export_settings

        self.log.info("Post-processing done for record %s. Exporting to %s.", record_id, settings.folder)
        self.cortex.export_record(
            settings.folder,
            list(settings.data_types),
            settings.export_format,
            [record_id],
            settings.version,
        )

    def _on_export_record_done(self, *args: Any, **kwargs: Any) -> None:
        self.log.info("Record export finished: %s", kwargs.get("data"))
        self._export_done.set()

    def _on_inform_error(self, *args: Any, **kwargs: Any) -> None:
        self.log.error("Cortex recording error: %s", kwargs.get("error_data"))


if __name__ == "__main__":
    print("Recording module demo")
    print("This demo does not connect to Cortex and does not start a recording.")
    print(f"Example record title: {generate_record_title('SSVEP_Flicker', 10.0)}")

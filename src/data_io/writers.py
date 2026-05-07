from __future__ import annotations

"""
Safe JSON and CSV writers.

This module only writes files. It does not build experiment logic, does not
connect to Cortex, and does not know any stimulus or trial structure.
"""

import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def _recovery_path(path: str | Path) -> Path:
    """Return a timestamped fallback path for failed atomic writes."""

    target = Path(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return target.with_name(f"{target.stem}.recovery_{stamp}{target.suffix}")


def safe_write_json(
    data: Any,
    path: str | Path,
    *,
    logger: logging.Logger | None = None,
    retries: int = 5,
    retry_delay_s: float = 0.15,
) -> Path:
    """
    Write data as JSON using a temporary file and atomic replace.

    Permission errors are retried because Windows may lock files while they
    are open in another program. If all retries fail, a recovery file is
    written instead.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        tmp = target.with_suffix(target.suffix + f".tmp.{attempt}")
        try:
            with tmp.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False, default=str)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, target)
            return target
        except PermissionError as exc:
            last_error = exc
            if tmp.exists():
                try:
                    tmp.unlink(missing_ok=True)
                except PermissionError:
                    if logger is not None:
                        logger.warning("JSON temp cleanup blocked | path=%s", tmp)
            if logger is not None:
                logger.warning("JSON write blocked | path=%s | attempt=%d/%d", target, attempt, retries)
            time.sleep(retry_delay_s)

    recovery = _recovery_path(target)
    with recovery.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)
        file.write("\n")
    if logger is not None:
        logger.error("JSON write fallback used | target=%s | recovery=%s | error=%s", target, recovery, last_error)
    return recovery


def write_json(data: Any, path: str | Path) -> Path:
    """Write JSON and return the actual written path."""

    return safe_write_json(data, path)


def write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Write rows to a new CSV file, overwriting an existing file."""

    if not rows:
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or rows[0].keys())

    with target.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    logger: logging.Logger | None = None,
    retries: int = 5,
    retry_delay_s: float = 0.1,
) -> None:
    """Append rows to a CSV file and create the header if needed."""

    if not rows:
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            new_file = not target.exists()
            encoding = "utf-8-sig" if new_file else "utf-8"
            with target.open("a", newline="", encoding=encoding) as file:
                writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerows(rows)
                file.flush()
                os.fsync(file.fileno())
            return
        except PermissionError as exc:
            last_error = exc
            if logger is not None:
                logger.warning("CSV append blocked | path=%s | attempt=%d/%d", target, attempt, retries)
            time.sleep(retry_delay_s)

    if logger is not None:
        logger.error("CSV append failed after retries | path=%s | error=%s", target, last_error)
    raise last_error or RuntimeError(f"Could not append CSV rows to {target}")


def write_markerlog_csv(marker: Any, path: str | Path) -> None:
    """Write marker.stats.log to CSV if marker timing rows exist."""

    rows = marker.stats.log
    if not rows:
        return
    write_csv(rows, path, fieldnames=list(rows[0].keys()))


def save_session_outputs(
    *,
    protocol: dict[str, Any],
    trial_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    paths: Mapping[str, str | Path],
    marker: Any | None = None,
    rating_rows: Sequence[Mapping[str, Any]] | None = None,
    trial_fields: Sequence[str] | None = None,
    rating_fields: Sequence[str] | None = None,
) -> None:
    """Write standard session output files."""

    if marker is not None:
        protocol["marker_timing_log"] = marker.stats.log
        write_markerlog_csv(marker, paths["markerlog"])

    safe_write_json(protocol, paths["protocol"])

    if trial_rows:
        write_csv(trial_rows, paths["trials"], fieldnames=trial_fields)

    if event_rows:
        event_fields = sorted({key for row in event_rows for key in row.keys()})
        write_csv(event_rows, paths["events"], fieldnames=event_fields)

    if rating_rows:
        write_csv(rating_rows, paths["ratings"], fieldnames=rating_fields)


if __name__ == "__main__":
    demo_dir = Path("data") / "_demo_writers"
    json_path = write_json({"demo": True, "rows": 2}, demo_dir / "demo.json")
    write_csv(
        [{"trial": 1, "value": "a"}, {"trial": 2, "value": "b"}],
        demo_dir / "demo.csv",
    )
    append_csv_rows(
        demo_dir / "append.csv",
        ["event", "value"],
        [{"event": "start", "value": 1}, {"event": "stop", "value": 0}],
    )

    print("Writers module demo")
    print("This demo writes small local files only.")
    print(f"JSON path: {json_path}")

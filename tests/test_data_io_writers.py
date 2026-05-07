from __future__ import annotations

import csv
import json
from pathlib import Path

from data_io import writers


def test_safe_write_json_writes_readable_json(tmp_path: Path) -> None:
    """JSON writer should create a readable target file."""

    target = tmp_path / "protocol.json"
    written = writers.safe_write_json({"status": "ok", "value": 1}, target)

    assert written == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "ok", "value": 1}


def test_write_csv_uses_given_field_order(tmp_path: Path) -> None:
    """CSV writer should respect explicit field order."""

    target = tmp_path / "trials.csv"
    writers.write_csv(
        [{"trial": 1, "value": "a", "ignored": True}],
        target,
        fieldnames=["trial", "value"],
    )

    with target.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    assert rows == [{"trial": "1", "value": "a"}]


def test_append_csv_rows_creates_header_once(tmp_path: Path) -> None:
    """Appending rows should create one header and keep all rows."""

    target = tmp_path / "events.csv"
    writers.append_csv_rows(target, ["event", "value"], [{"event": "start", "value": 1}])
    writers.append_csv_rows(target, ["event", "value"], [{"event": "stop", "value": 0}])

    lines = target.read_text(encoding="utf-8-sig").splitlines()

    assert lines[0] == "event,value"
    assert lines.count("event,value") == 1
    assert len(lines) == 3


def test_save_session_outputs_writes_standard_files(tmp_path: Path) -> None:
    """Standard session output helper should write protocol and data files."""

    paths = {
        "protocol": tmp_path / "protocol.json",
        "trials": tmp_path / "trials.csv",
        "events": tmp_path / "events.csv",
        "ratings": tmp_path / "ratings.csv",
        "markerlog": tmp_path / "markerlog.csv",
    }
    protocol = {"status": "completed"}

    writers.save_session_outputs(
        protocol=protocol,
        trial_rows=[{"trial": 1}],
        event_rows=[{"event": "done"}],
        rating_rows=[{"rating": 6}],
        paths=paths,
        trial_fields=["trial"],
        rating_fields=["rating"],
    )

    assert paths["protocol"].is_file()
    assert paths["trials"].is_file()
    assert paths["events"].is_file()
    assert paths["ratings"].is_file()

from __future__ import annotations

"""
Session naming and output file helpers.

This module builds session IDs and output file names. It does not load .env
files and does not define global path settings.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class SessionPaths:
    """Output paths for one experiment session."""

    proband_folder: Path
    folder: Path
    stem: str
    log: Path
    protocol: Path
    trials: Path
    events: Path
    ratings: Path
    markerlog: Path

    def as_dict(self) -> dict[str, str]:
        """Return string paths for compatibility with older code."""

        return {
            "proband_folder": str(self.proband_folder),
            "folder": str(self.folder),
            "stem": self.stem,
            "log": str(self.log),
            "protocol": str(self.protocol),
            "trials": str(self.trials),
            "events": str(self.events),
            "ratings": str(self.ratings),
            "markerlog": str(self.markerlog),
        }


def build_run_timestamp() -> str:
    """Return the current local time as YYYYMMDD_HHMMSS."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    """Return the current local time as ISO-8601 with seconds precision."""

    return datetime.now().isoformat(timespec="seconds")


def determine_next_session_id(
    proband_id: str,
    phase_short: str,
    out_dir: str | Path,
    folder_name: str | None = None,
) -> tuple[str, Path]:
    """Return the next SNN session ID and the proband phase folder."""

    out_root = Path(out_dir)
    proband_folder_name = folder_name or f"{proband_id}_{phase_short}"
    proband_folder = out_root / proband_folder_name

    if not proband_folder.is_dir():
        return "S01", proband_folder

    pattern = re.compile(
        rf"^{re.escape(proband_id)}_S(\d+)_{re.escape(phase_short)}_\d{{8}}_\d{{6}}$"
    )
    max_num = 0
    for entry in proband_folder.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match:
            max_num = max(max_num, int(match.group(1)))

    return f"S{max_num + 1:02d}", proband_folder


def build_output_paths(
    proband_id: str,
    session_id: str,
    phase_short: str,
    run_timestamp: str,
    out_dir: str | Path,
    folder_name: str | None = None,
) -> SessionPaths:
    """Create and return output paths for one experiment session."""

    out_root = Path(out_dir)
    proband_folder_name = folder_name or f"{proband_id}_{phase_short}"
    proband_folder = out_root / proband_folder_name

    stem = f"{proband_id}_{session_id}_{phase_short}_{run_timestamp}"
    session_folder = proband_folder / stem
    session_folder.mkdir(parents=True, exist_ok=True)

    return SessionPaths(
        proband_folder=proband_folder,
        folder=session_folder,
        stem=stem,
        log=session_folder / f"{stem}.log",
        protocol=session_folder / f"{stem}_protocol.json",
        trials=session_folder / f"{stem}_trials.csv",
        events=session_folder / f"{stem}_events.csv",
        ratings=session_folder / f"{stem}_ratings.csv",
        markerlog=session_folder / f"{stem}_markerlog.csv",
    )


def build_cortex_record_title(
    proband_id: str,
    phase_short: str,
    session_id: str,
    run_timestamp: str,
) -> str:
    """Build the Cortex record title for one session."""

    return f"{proband_id}_{session_id}_{phase_short}_{run_timestamp}"


if __name__ == "__main__":
    demo_timestamp = "20260502_120000"
    session_id, proband_folder = determine_next_session_id(
        "P01",
        "demo",
        Path("data") / "_demo_sessions",
    )
    paths = build_output_paths(
        "P01",
        session_id,
        "demo",
        demo_timestamp,
        Path("data") / "_demo_sessions",
    )

    print("Session module demo")
    print("This demo creates one local demo folder only.")
    print(f"Next session: {session_id}")
    print(f"Participant folder: {proband_folder}")
    print(f"Session stem: {paths.stem}")
    print(f"Record title: {build_cortex_record_title('P01', 'demo', session_id, demo_timestamp)}")


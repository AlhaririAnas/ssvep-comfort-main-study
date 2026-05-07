from __future__ import annotations

from pathlib import Path

from core import session


def test_determine_next_session_id_returns_s01_for_new_folder(tmp_path: Path) -> None:
    """A new proband phase folder should start at S01."""

    session_id, proband_folder = session.determine_next_session_id("P01", "freqtest", tmp_path)

    assert session_id == "S01"
    assert proband_folder == tmp_path / "P01_freqtest"


def test_determine_next_session_id_increments_existing_sessions(tmp_path: Path) -> None:
    """Existing session folders should increment the next session number."""

    proband_folder = tmp_path / "P01_freqtest"
    (proband_folder / "P01_S01_freqtest_20260502_120000").mkdir(parents=True)
    (proband_folder / "P01_S03_freqtest_20260502_130000").mkdir()
    (proband_folder / "ignore_me").mkdir()

    session_id, _ = session.determine_next_session_id("P01", "freqtest", tmp_path)

    assert session_id == "S04"


def test_build_output_paths_creates_expected_files(tmp_path: Path) -> None:
    """Output paths should use the standard proband/session naming scheme."""

    paths = session.build_output_paths(
        proband_id="P02",
        session_id="S01",
        phase_short="stimscr",
        run_timestamp="20260502_123456",
        out_dir=tmp_path,
    )

    assert paths.folder.is_dir()
    assert paths.stem == "P02_S01_stimscr_20260502_123456"
    assert paths.protocol.name.endswith("_protocol.json")
    assert paths.trials.name.endswith("_trials.csv")
    assert paths.events.name.endswith("_events.csv")
    assert paths.ratings.name.endswith("_ratings.csv")
    assert paths.markerlog.name.endswith("_markerlog.csv")


def test_build_output_paths_supports_main_study_root_folder(tmp_path: Path) -> None:
    """Main-study outputs can be grouped as data/main_study/<proband>/<session>."""

    paths = session.build_output_paths(
        proband_id="P03",
        session_id="S01",
        phase_short="mainstudy",
        run_timestamp="20260507_120000",
        out_dir=tmp_path,
        folder_name=str(Path("main_study") / "P03"),
    )

    assert paths.proband_folder == tmp_path / "main_study" / "P03"
    assert paths.folder == tmp_path / "main_study" / "P03" / "P03_S01_mainstudy_20260507_120000"
    assert paths.folder.is_dir()


def test_session_paths_as_dict_returns_strings(tmp_path: Path) -> None:
    """Compatibility dict should contain string paths."""

    paths = session.build_output_paths("P01", "S01", "demo", "20260502_120000", tmp_path)
    as_dict = paths.as_dict()

    assert as_dict["stem"] == paths.stem
    assert isinstance(as_dict["folder"], str)
    assert isinstance(as_dict["protocol"], str)


def test_build_cortex_record_title_matches_session_stem() -> None:
    """Cortex record title should match the session stem format."""

    title = session.build_cortex_record_title("P01", "mainstudy", "S02", "20260502_120000")

    assert title == "P01_S02_mainstudy_20260502_120000"

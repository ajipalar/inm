"""Tests for build_spec_table: spectral count matrix and composite table."""

from pathlib import Path

import pandas as pd
import pytest

from inm.preprocess.build_spec_table import get_spec_table_and_composites, run

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Real cullin AP-MS xlsx (mmc2.xlsx from Wan et al. 2019, Cell Host & Microbe)
_CANONICAL = (
    Path(__file__).parent.parent.parent
    / "data"
    / "cullin"
    / "1-s2.0-S1931312819302537-mmc2.xlsx"
)
_CULLIN_SHEETS = [0, 1, 2]  # sheets containing Bait/Prey/SaintScore/Spec/ctrlCounts


def _write_apms_excel(path: Path, data: dict) -> None:
    pd.DataFrame(data).to_excel(path, index=False)


# ---------------------------------------------------------------------------
# get_spec_table_and_composites
# ---------------------------------------------------------------------------


def test_spec_table_basic_shape(tmp_path):
    """spec_table rows = unique preys; composites rows = total bait-prey rows."""
    xlsx = tmp_path / "apms.xlsx"
    _write_apms_excel(
        xlsx,
        {
            "Bait": ["CUL5", "CUL5"],
            "Prey": ["P00001", "P00002"],
            "SaintScore": [0.9, 0.7],
            "Spec": ["5|3|4", "2|1|0"],
            "ctrlCounts": ["0|0|0", "0|0|0"],
        },
    )
    spec_table, composites, prey_gene_map = get_spec_table_and_composites(
        xlsx, sheet_nums=[0]
    )

    assert set(spec_table.index) == {"P00001", "P00002"}
    assert len(composites) == 2
    assert list(composites.columns) == ["SID", "Bait", "Prey", "MSscore"]
    assert prey_gene_map == {}


def test_spec_table_pipe_delimited_counts(tmp_path):
    """Pipe-delimited Spec values are unpacked into separate columns."""
    xlsx = tmp_path / "apms.xlsx"
    _write_apms_excel(
        xlsx,
        {
            "Bait": ["CUL5"],
            "Prey": ["P00001"],
            "SaintScore": [0.9],
            "Spec": ["5|3|4"],
            "ctrlCounts": ["0|0|0"],
        },
    )
    spec_table, _, _map = get_spec_table_and_composites(xlsx, sheet_nums=[0])

    row = spec_table.loc["P00001"].values
    # 3 spec + 3 ctrl = first 6 values should be [5, 3, 4, 0, 0, 0]
    assert list(row[:6]) == [5, 3, 4, 0, 0, 0]


def test_spec_table_saint_score_out_of_range(tmp_path):
    """SaintScore outside [0, 1] raises AssertionError."""
    xlsx = tmp_path / "apms.xlsx"
    _write_apms_excel(
        xlsx,
        {
            "Bait": ["CUL5"],
            "Prey": ["P00001"],
            "SaintScore": [1.5],  # invalid
            "Spec": ["1|1|1"],
            "ctrlCounts": ["0|0|0"],
        },
    )
    with pytest.raises(AssertionError):
        get_spec_table_and_composites(xlsx, sheet_nums=[0])


def test_spec_table_composites_bait_truncated(tmp_path):
    """Bait name is truncated to first 4 characters in composites."""
    xlsx = tmp_path / "apms.xlsx"
    _write_apms_excel(
        xlsx,
        {
            "Bait": ["CUL5_FLAG"],
            "Prey": ["P00001"],
            "SaintScore": [0.8],
            "Spec": ["2|1|3"],
            "ctrlCounts": ["0|0|0"],
        },
    )
    _, composites, _ = get_spec_table_and_composites(xlsx, sheet_nums=[0])
    assert composites.iloc[0]["Bait"] == "CUL5"


def test_spec_table_control_deduplication(tmp_path):
    """Duplicate control replicates across two baits sharing the same ctrl vector
    are only included once per prey."""
    xlsx = tmp_path / "apms.xlsx"
    _write_apms_excel(
        xlsx,
        {
            "Bait": ["BAIT1", "BAIT2"],
            "Prey": ["P00001", "P00001"],  # same prey, two baits
            "SaintScore": [0.9, 0.8],
            "Spec": ["5|3|4", "2|1|0"],
            "ctrlCounts": ["0|0|0", "1|2|1"],  # different ctrl vectors
        },
    )
    spec_table, composites, _ = get_spec_table_and_composites(xlsx, sheet_nums=[0])
    assert "P00001" in spec_table.index
    assert len(composites) == 2


def test_prey_gene_map_extraction(tmp_path):
    """When prey_gene_col is given, prey_gene_map maps UniProt IDs to gene names."""
    xlsx = tmp_path / "apms.xlsx"
    _write_apms_excel(
        xlsx,
        {
            "Bait": ["CUL5", "CUL5"],
            "Prey": ["P00001", "P00002"],
            "PreyGene": ["CUL1", "RBX1"],
            "SaintScore": [0.9, 0.7],
            "Spec": ["5|3|4", "2|1|0"],
            "ctrlCounts": ["0|0|0", "0|0|0"],
        },
    )
    _, _, prey_gene_map = get_spec_table_and_composites(
        xlsx, sheet_nums=[0], prey_gene_col="PreyGene"
    )
    assert prey_gene_map == {"P00001": "CUL1", "P00002": "RBX1"}


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_writes_output_files(tmp_path):
    """run() writes spec_table.tsv and composite_table.tsv."""
    xlsx = tmp_path / "apms.xlsx"
    _write_apms_excel(
        xlsx,
        {
            "Bait": ["CUL5", "CUL5"],
            "Prey": ["P00001", "P00002"],
            "SaintScore": [0.9, 0.7],
            "Spec": ["5|3|4", "2|1|0"],
            "ctrlCounts": ["0|0|0", "0|0|0"],
        },
    )
    cfg = {
        "data": {"xlsx_path": str(xlsx)},
        "preprocess": {
            "sheet_nums": [0],
            "bait_col": "Bait",
            "prey_col": "Prey",
            "ms_score_col": "SaintScore",
        },
    }
    run(cfg, output_dir=tmp_path)

    assert (tmp_path / "spec_table.tsv").exists()
    assert (tmp_path / "composite_table.tsv").exists()
    assert not (tmp_path / "prey_names.tsv").exists()

    spec = pd.read_csv(tmp_path / "spec_table.tsv", sep="\t", header=None, index_col=0)
    assert set(spec.index) == {"P00001", "P00002"}

    comp = pd.read_csv(tmp_path / "composite_table.tsv", sep="\t")
    assert len(comp) == 2


def test_run_uses_fixture_apms(tmp_path, ensure_dummy_apms):
    """run() produces valid output on the shared dummy_apms fixture."""
    cfg = {
        "data": {"xlsx_path": str(FIXTURE_DIR / "dummy_apms.xlsx")},
        "preprocess": {
            "sheet_nums": [0],
            "bait_col": "Bait",
            "prey_col": "Prey",
            "ms_score_col": "SaintScore",
        },
    }
    run(cfg, output_dir=tmp_path)

    spec = pd.read_csv(tmp_path / "spec_table.tsv", sep="\t", header=None, index_col=0)
    # All 4 preys in the fixture should appear
    assert len(spec) == 4


def test_run_writes_prey_names_tsv(tmp_path, ensure_dummy_apms):
    """When prey_gene_col is configured, run() writes prey_names.tsv with
    UniProt and GeneName columns."""
    cfg = {
        "data": {"xlsx_path": str(FIXTURE_DIR / "dummy_apms.xlsx")},
        "preprocess": {
            "sheet_nums": [0],
            "bait_col": "Bait",
            "prey_col": "Prey",
            "ms_score_col": "SaintScore",
            "prey_gene_col": "PreyGene",
        },
    }
    run(cfg, output_dir=tmp_path)

    assert (tmp_path / "prey_names.tsv").exists()
    names = pd.read_csv(tmp_path / "prey_names.tsv", sep="\t")
    assert list(names.columns) == ["UniProt", "GeneName"]
    assert set(names["UniProt"]) == {"P00001", "Q99999", "OLDID", "P00002"}
    gene_map = dict(zip(names["UniProt"], names["GeneName"]))
    assert gene_map["P00001"] == "PROT_A"
    assert gene_map["P00002"] == "PROT_B"


# ---------------------------------------------------------------------------
# Integration tests against the real cullin AP-MS xlsx (mmc2.xlsx)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CANONICAL.exists(), reason="canonical cullin xlsx not present")
def test_composites_match_xlsx_row_by_row():
    """composite_table is a faithful transform of the xlsx: row count, bait
    truncation to 4 chars, prey identity, and SAINT score all match exactly."""
    _, composites, _ = get_spec_table_and_composites(
        _CANONICAL, sheet_nums=_CULLIN_SHEETS
    )

    frames = [
        pd.read_excel(_CANONICAL, sheet_name=i)[["Bait", "Prey", "SaintScore"]]
        for i in _CULLIN_SHEETS
    ]
    xlsx = pd.concat(frames, ignore_index=True)

    assert len(composites) == len(xlsx)
    assert (composites["Bait"].values == [b[:4] for b in xlsx["Bait"].values]).all()
    assert (composites["Prey"].values == xlsx["Prey"].values).all()
    assert (composites["MSscore"].values == xlsx["SaintScore"].values).all()


@pytest.mark.skipif(not _CANONICAL.exists(), reason="canonical cullin xlsx not present")
def test_spec_table_counts_match_xlsx_for_single_condition_prey():
    """For a prey that appears with exactly one bait condition, the first N
    entries of its spec_table row equal the pipe-delimited Spec values from
    the xlsx (spectral counts are round-tripped without alteration)."""
    spec_table, _, _ = get_spec_table_and_composites(
        _CANONICAL, sheet_nums=_CULLIN_SHEETS
    )

    # P31749 appears only once: CBFBwt_MG132, Spec="0|15|0|0"
    expected_spec = [0, 15, 0, 0]
    row = spec_table.loc["P31749"].values
    assert list(row[: len(expected_spec)]) == expected_spec

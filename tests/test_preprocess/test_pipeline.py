"""End-to-end test: parse_apms through filter_hits on real HHblits fixtures."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from inm.preprocess import add_sequences, filter_hits, parse_apms, parse_hhr
from inm.preprocess.config import load_config

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg():
    c = load_config(FIXTURE_DIR / "test_config.yaml")
    c["xlsx_path"] = str(FIXTURE_DIR / "dummy_apms.xlsx")
    c["exclude_prey_genes"] = ["IGHG1_MOUSE", "vifprotein"]
    return c


def test_pipeline_parse_and_add_sequences(tmp_path, cfg):
    """Step 1-3: parse AP-MS data and attach sequences."""
    parse_apms.run(cfg, output_dir=tmp_path)
    prey_table_path = tmp_path / "prey_table.csv"
    assert prey_table_path.exists()

    prey_table = pd.read_csv(prey_table_path)
    assert len(prey_table) == 2
    assert set(prey_table["UniprotId"]) == {"P00001", "P00002"}

    seq_dir = tmp_path / "input_sequences"
    seq_dir.mkdir(exist_ok=True)
    for fasta in FIXTURE_DIR.glob("*.fasta"):
        shutil.copy(fasta, seq_dir)

    add_sequences.run(cfg, output_dir=tmp_path)
    prey_table = pd.read_csv(prey_table_path)
    assert "UniprotSeq" in prey_table.columns
    assert all(prey_table["UniprotSeqLen"] >= 100)


def test_pipeline_parse_hhr_and_filter(tmp_path, cfg):
    """Steps 4-6: parse HHR fixtures and filter to significant hits."""
    # Set up hhblits_out with real HHR fixtures
    hhr_dir = tmp_path / "hhblits_out"
    hhr_dir.mkdir()
    for hhr in FIXTURE_DIR.glob("*.hhr"):
        shutil.copy(hhr, hhr_dir)

    # parse_hhr: produces hhr_alns.json + PreyPDB70PairAlign.csv
    parse_hhr.run(cfg, output_dir=tmp_path)
    assert (hhr_dir / "hhr_alns.json").exists()

    csv_path = hhr_dir / "PreyPDB70PairAlign.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "QueryUID" in df.columns
    assert "PDB70ID" in df.columns
    assert "Evalue" in df.columns
    assert "Aligned_cols" in df.columns
    assert "Identities" in df.columns
    total_hits = len(df)
    assert total_hits > 0

    # filter_hits: produces SignificantPreyPDB70PairAlign.csv
    filter_hits.run(cfg, output_dir=tmp_path)
    sig_path = hhr_dir / "SignificantPreyPDB70PairAlign.csv"
    assert sig_path.exists()
    sig_df = pd.read_csv(sig_path)
    assert len(sig_df) < total_hits  # filtering should reduce rows
    assert len(sig_df) > 0  # but real hits should survive


def test_pipeline_full_prestructure(tmp_path, cfg):
    """Steps 1-6 end-to-end: AP-MS parsing through significant hit filtering."""
    # Steps 1-3
    parse_apms.run(cfg, output_dir=tmp_path)
    seq_dir = tmp_path / "input_sequences"
    seq_dir.mkdir()
    for fasta in FIXTURE_DIR.glob("*.fasta"):
        shutil.copy(fasta, seq_dir)
    add_sequences.run(cfg, output_dir=tmp_path)

    # Steps 4-6: only copy .hhr files for proteins in the prey table
    prey_table = pd.read_csv(tmp_path / "prey_table.csv")
    hhr_dir = tmp_path / "hhblits_out"
    hhr_dir.mkdir()
    for uid in prey_table["UniprotId"]:
        hhr_file = FIXTURE_DIR / f"{uid}.hhr"
        if hhr_file.exists():
            shutil.copy(hhr_file, hhr_dir)
    parse_hhr.run(cfg, output_dir=tmp_path)
    filter_hits.run(cfg, output_dir=tmp_path)

    # Verify the chain of outputs exists
    assert (tmp_path / "prey_table.csv").exists()
    assert (hhr_dir / "PreyPDB70PairAlign.csv").exists()
    assert (hhr_dir / "SignificantPreyPDB70PairAlign.csv").exists()

    sig_hits = pd.read_csv(hhr_dir / "SignificantPreyPDB70PairAlign.csv")

    # All significant hits should map back to proteins in the prey table
    assert set(sig_hits["QueryUID"]).issubset(set(prey_table["UniprotId"]))
    assert len(sig_hits) > 0

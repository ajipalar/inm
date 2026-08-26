from unittest.mock import patch

import pandas as pd

from inm.preprocess.fetch_sequences import run


def test_fetch_sequences_skips_existing(tmp_path):
    # Create prey_table.csv with 2 UniProt IDs
    prey_df = pd.DataFrame(
        {
            "PreyGene": ["PROT_A", "PROT_B"],
            "UniprotId": ["P00001", "P00002"],
        }
    )
    prey_df.to_csv(tmp_path / "prey_table.csv", index=False)

    # Pre-populate FASTA dir with existing files
    seq_dir = tmp_path / "input_sequences"
    seq_dir.mkdir()
    (seq_dir / "P00001.fasta").write_text(">sp|P00001|PROT_A\nMKTAYIAKQ\n")
    (seq_dir / "P00002.fasta").write_text(">sp|P00002|PROT_B\nMGSSHHHH\n")

    cfg = {"dataset_name": "test"}
    run(cfg, output_dir=tmp_path)

    # Both files should still exist, no network calls made
    assert (seq_dir / "P00001.fasta").exists()
    assert (seq_dir / "P00002.fasta").exists()


def test_fetch_sequences_creates_dir(tmp_path):
    """When input_sequences does not exist, run() creates it and fetches missing files."""
    prey_df = pd.DataFrame(
        {
            "PreyGene": ["PROT_A"],
            "UniprotId": ["P00001"],
        }
    )
    prey_df.to_csv(tmp_path / "prey_table.csv", index=False)

    # input_sequences dir does NOT exist at this point
    assert not (tmp_path / "input_sequences").exists()

    with patch("biotite.database.uniprot.fetch") as mock_fetch:
        cfg = {"dataset_name": "test"}
        run(cfg, output_dir=tmp_path)

    assert (tmp_path / "input_sequences").is_dir()
    mock_fetch.assert_called_once()

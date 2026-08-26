import pandas as pd
from inm.preprocess.add_sequences import run


def test_add_sequences_adds_columns(tmp_path):
    prey_df = pd.DataFrame(
        {
            "PreyGene": ["PROT_A", "PROT_B"],
            "UniprotId": ["P00001", "P00002"],
        }
    )
    prey_df.to_csv(tmp_path / "prey_table.csv", index=False)

    seq_dir = tmp_path / "input_sequences"
    seq_dir.mkdir()
    (seq_dir / "P00001.fasta").write_text(">sp|P00001|PROT_A\nMKTAYIAKQ\n")
    (seq_dir / "P00002.fasta").write_text(">sp|P00002|PROT_B\nMGSSHHHH\n")

    cfg = {"dataset_name": "test"}
    run(cfg, output_dir=tmp_path)

    result = pd.read_csv(tmp_path / "prey_table.csv")
    assert "UniprotSeq" in result.columns
    assert "UniprotSeqLen" in result.columns
    assert result.loc[0, "UniprotSeq"] == "MKTAYIAKQ"
    assert result.loc[0, "UniprotSeqLen"] == 9
    assert result.loc[1, "UniprotSeq"] == "MGSSHHHH"
    assert result.loc[1, "UniprotSeqLen"] == 8

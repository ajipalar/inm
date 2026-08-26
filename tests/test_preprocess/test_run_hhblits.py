from pathlib import Path
import pandas as pd
from inm.preprocess.run_hhblits import build_hhblits_command, run


def test_build_command():
    cmd = build_hhblits_command(
        fasta_path=Path("input_sequences/P00001.fasta"),
        output_path=Path("hhblits_out/P00001.hhr"),
        db_path="/databases/pdb70/pdb70",
        cpu=2,
    )
    assert cmd[0] == "hhblits"
    assert "-i" in cmd
    assert "input_sequences/P00001.fasta" in cmd
    assert "-o" in cmd
    assert "hhblits_out/P00001.hhr" in cmd
    assert "-cpu" in cmd
    assert "2" in cmd


def test_run_skips_existing_hhr(tmp_path):
    prey_df = pd.DataFrame({"UniprotId": ["P00001"]})
    prey_df.to_csv(tmp_path / "prey_table.csv", index=False)

    seq_dir = tmp_path / "input_sequences"
    seq_dir.mkdir()
    (seq_dir / "P00001.fasta").write_text(">P00001\nMKTAYIAKQ\n")

    hhr_dir = tmp_path / "hhblits_out"
    hhr_dir.mkdir()
    (hhr_dir / "P00001.hhr").write_text("dummy hhr content")

    cfg = {"hhblits": {"db_path": "/dev/null", "cpu": 1}}
    run(cfg, output_dir=tmp_path)

    # Should not crash — skipped because .hhr exists
    assert (hhr_dir / "P00001.hhr").read_text() == "dummy hhr content"

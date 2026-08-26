from pathlib import Path

import pandas as pd

from inm.preprocess.config import load_config
from inm.preprocess.parse_apms import run

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_apms_produces_prey_table(tmp_path):
    cfg = load_config(FIXTURE_DIR / "test_config.yaml")
    cfg["xlsx_path"] = str(FIXTURE_DIR / "dummy_apms.xlsx")
    cfg["sheet_nums"] = [0]
    cfg["viral_remapping"] = {"vifprotein": "P69723"}
    cfg["exclude_prey_genes"] = ["IGHG1_MOUSE"]

    run(cfg, output_dir=tmp_path)

    prey_table = pd.read_csv(tmp_path / "prey_table.csv")
    assert "PreyGene" in prey_table.columns
    assert "UniprotId" in prey_table.columns
    # Excluded gene should be gone
    assert "IGHG1_MOUSE" not in prey_table["PreyGene"].values
    # Viral remapping should apply
    row = prey_table[prey_table["PreyGene"] == "vifprotein"]
    assert len(row) == 1
    assert row["UniprotId"].values[0] == "P69723"
    # Normal proteins should be present
    assert "PROT_A" in prey_table["PreyGene"].values
    assert "PROT_B" in prey_table["PreyGene"].values
    # Should have exactly 3 rows (IGHG1_MOUSE excluded)
    assert len(prey_table) == 3

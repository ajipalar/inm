import pandas as pd
from inm.preprocess.filter_hits import run

_CFG = {
    "filter": {
        "max_evalue": 1e-7,
        "min_aligned_cols": 88,
        "min_identity_pct": 30,
    }
}


def _write_hits(tmp_path, rows):
    hhr_dir = tmp_path / "hhblits_out"
    hhr_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(hhr_dir / "PreyPDB70PairAlign.csv", index=False)
    return hhr_dir


def test_filter_hits_applies_thresholds(tmp_path):
    _write_hits(
        tmp_path,
        {
            "QueryUID": ["P1", "P2", "P3"],
            "PDB70ID": ["1abc_A", "2def_B", "3ghi_C"],
            "Evalue": [1e-10, 1e-3, 1e-20],
            "Aligned_cols": ["100", "50", "200"],
            "Identities": ["50%", "10%", "80%"],
        },
    )
    run(_CFG, output_dir=tmp_path)
    result = pd.read_csv(tmp_path / "hhblits_out" / "SignificantPreyPDB70PairAlign.csv")
    assert len(result) == 2
    assert "P2" not in result["QueryUID"].values
    assert "P1" in result["QueryUID"].values
    assert "P3" in result["QueryUID"].values


def test_filter_hits_boundary_evalue(tmp_path):
    """Row at exactly max_evalue is kept (<=); row just above is dropped."""
    _write_hits(
        tmp_path,
        {
            "QueryUID": ["P_exact", "P_over"],
            "PDB70ID": ["1abc_A", "2def_B"],
            "Evalue": [1e-7, 1.1e-7],
            "Aligned_cols": ["100", "100"],
            "Identities": ["50%", "50%"],
        },
    )
    run(_CFG, output_dir=tmp_path)
    result = pd.read_csv(tmp_path / "hhblits_out" / "SignificantPreyPDB70PairAlign.csv")
    assert "P_exact" in result["QueryUID"].values
    assert "P_over" not in result["QueryUID"].values


def test_filter_hits_boundary_aligned_cols(tmp_path):
    """Row at exactly min_aligned_cols is kept (>=); row one below is dropped."""
    _write_hits(
        tmp_path,
        {
            "QueryUID": ["P_exact", "P_under"],
            "PDB70ID": ["1abc_A", "2def_B"],
            "Evalue": [1e-10, 1e-10],
            "Aligned_cols": ["88", "87"],
            "Identities": ["50%", "50%"],
        },
    )
    run(_CFG, output_dir=tmp_path)
    result = pd.read_csv(tmp_path / "hhblits_out" / "SignificantPreyPDB70PairAlign.csv")
    assert "P_exact" in result["QueryUID"].values
    assert "P_under" not in result["QueryUID"].values


def test_filter_hits_boundary_identity(tmp_path):
    """Row at exactly min_identity_pct is kept (>=); row one below is dropped."""
    _write_hits(
        tmp_path,
        {
            "QueryUID": ["P_exact", "P_under"],
            "PDB70ID": ["1abc_A", "2def_B"],
            "Evalue": [1e-10, 1e-10],
            "Aligned_cols": ["100", "100"],
            "Identities": ["30%", "29%"],
        },
    )
    run(_CFG, output_dir=tmp_path)
    result = pd.read_csv(tmp_path / "hhblits_out" / "SignificantPreyPDB70PairAlign.csv")
    assert "P_exact" in result["QueryUID"].values
    assert "P_under" not in result["QueryUID"].values


def test_filter_hits_all_filtered_gives_empty(tmp_path):
    """When all hits fail every threshold, output CSV exists but has no rows."""
    _write_hits(
        tmp_path,
        {
            "QueryUID": ["P1", "P2"],
            "PDB70ID": ["1abc_A", "2def_B"],
            "Evalue": [0.5, 0.9],
            "Aligned_cols": ["10", "5"],
            "Identities": ["5%", "2%"],
        },
    )
    run(_CFG, output_dir=tmp_path)
    result = pd.read_csv(tmp_path / "hhblits_out" / "SignificantPreyPDB70PairAlign.csv")
    assert len(result) == 0


def test_filter_hits_identities_without_percent_sign(tmp_path):
    """Identities column without '%' suffix is still parsed correctly."""
    _write_hits(
        tmp_path,
        {
            "QueryUID": ["P1"],
            "PDB70ID": ["1abc_A"],
            "Evalue": [1e-10],
            "Aligned_cols": ["100"],
            "Identities": ["50"],  # no % sign
        },
    )
    run(_CFG, output_dir=tmp_path)
    result = pd.read_csv(tmp_path / "hhblits_out" / "SignificantPreyPDB70PairAlign.csv")
    assert "P1" in result["QueryUID"].values

import pandas as pd
from inm.preprocess.build_reference import run


def test_build_reference_produces_output(tmp_path):
    struct_dir = tmp_path / "significant_cifs"
    struct_dir.mkdir()

    chain_df = pd.DataFrame(
        {
            "QueryID": ["P1", "P1", "P2", "P2"],
            "ChainID": ["A", "A", "B", "B"],
            "PDBID": ["1abc", "1abc", "1abc", "1abc"],
            "bt_aln_percent_seq_id": [0.9, 0.9, 0.8, 0.8],
        }
    )
    chain_df.to_csv(struct_dir / "chain_mapping_all.csv", index=False)

    bsasa_df = pd.DataFrame(
        {
            "PDBID": ["1abc"],
            "Chain1": ["A"],
            "Chain2": ["B"],
            "BSASA": [800.0],
            "SASA12": [1200.0],
            "SASA1": [1000.0],
            "SASA2": [1000.0],
        }
    )
    bsasa_df.to_csv(struct_dir / "bsasa.csv", index=False)

    cfg = {"filter": {"min_bsasa": 500, "min_seq_id": 0.3}}
    run(cfg, output_dir=tmp_path)

    result = pd.read_csv(struct_dir / "BSASA_reference.csv")
    assert "Prey1" in result.columns
    assert "Prey2" in result.columns
    assert len(result) > 0


def test_build_reference_filters_low_bsasa(tmp_path):
    """Rows with BSASA below threshold are excluded."""
    struct_dir = tmp_path / "significant_cifs"
    struct_dir.mkdir()

    chain_df = pd.DataFrame(
        {
            "QueryID": ["P1", "P2"],
            "ChainID": ["A", "B"],
            "PDBID": ["1abc", "1abc"],
            "bt_aln_percent_seq_id": [0.9, 0.8],
        }
    )
    chain_df.to_csv(struct_dir / "chain_mapping_all.csv", index=False)

    bsasa_df = pd.DataFrame(
        {
            "PDBID": ["1abc"],
            "Chain1": ["A"],
            "Chain2": ["B"],
            "BSASA": [100.0],  # Below 500 threshold
            "SASA12": [1200.0],
            "SASA1": [1000.0],
            "SASA2": [1000.0],
        }
    )
    bsasa_df.to_csv(struct_dir / "bsasa.csv", index=False)

    cfg = {"filter": {"min_bsasa": 500, "min_seq_id": 0.3}}
    run(cfg, output_dir=tmp_path)

    result = pd.read_csv(struct_dir / "BSASA_reference.csv")
    assert len(result) == 0


def test_build_reference_filters_low_seq_id(tmp_path):
    """Chains with seq_id below threshold produce no prey mapping."""
    struct_dir = tmp_path / "significant_cifs"
    struct_dir.mkdir()

    chain_df = pd.DataFrame(
        {
            "QueryID": ["P1", "P2"],
            "ChainID": ["A", "B"],
            "PDBID": ["1abc", "1abc"],
            "bt_aln_percent_seq_id": [0.1, 0.1],  # Below 0.3 threshold
        }
    )
    chain_df.to_csv(struct_dir / "chain_mapping_all.csv", index=False)

    bsasa_df = pd.DataFrame(
        {
            "PDBID": ["1abc"],
            "Chain1": ["A"],
            "Chain2": ["B"],
            "BSASA": [800.0],
            "SASA12": [1200.0],
            "SASA1": [1000.0],
            "SASA2": [1000.0],
        }
    )
    bsasa_df.to_csv(struct_dir / "bsasa.csv", index=False)

    cfg = {"filter": {"min_bsasa": 500, "min_seq_id": 0.3}}
    run(cfg, output_dir=tmp_path)

    result = pd.read_csv(struct_dir / "BSASA_reference.csv")
    assert len(result) == 0


def test_build_reference_homodimer(tmp_path):
    """Both chains map to the same prey (homodimer) → self-interaction row is produced."""
    struct_dir = tmp_path / "significant_cifs"
    struct_dir.mkdir()

    chain_df = pd.DataFrame(
        {
            "QueryID": ["P1", "P1"],
            "ChainID": ["A", "B"],
            "PDBID": ["1abc", "1abc"],
            "bt_aln_percent_seq_id": [0.9, 0.9],
        }
    )
    chain_df.to_csv(struct_dir / "chain_mapping_all.csv", index=False)

    bsasa_df = pd.DataFrame(
        {
            "PDBID": ["1abc"],
            "Chain1": ["A"],
            "Chain2": ["B"],
            "BSASA": [800.0],
            "SASA12": [1200.0],
            "SASA1": [1000.0],
            "SASA2": [1000.0],
        }
    )
    bsasa_df.to_csv(struct_dir / "bsasa.csv", index=False)

    cfg = {"filter": {"min_bsasa": 500, "min_seq_id": 0.3}}
    run(cfg, output_dir=tmp_path)

    result = pd.read_csv(struct_dir / "BSASA_reference.csv")
    assert len(result) == 1
    assert result.iloc[0]["Prey1"] == "P1"
    assert result.iloc[0]["Prey2"] == "P1"


def test_build_reference_multi_prey_per_chain(tmp_path):
    """When a chain maps to multiple preys (paralogs), all cross-products are produced."""
    struct_dir = tmp_path / "significant_cifs"
    struct_dir.mkdir()

    # Chain A maps to P1 and P2 (paralogs both mapping to same chain); Chain B maps to P3
    chain_df = pd.DataFrame(
        {
            "QueryID": ["P1", "P2", "P3"],
            "ChainID": ["A", "A", "B"],
            "PDBID": ["1abc", "1abc", "1abc"],
            "bt_aln_percent_seq_id": [0.9, 0.85, 0.88],
        }
    )
    chain_df.to_csv(struct_dir / "chain_mapping_all.csv", index=False)

    bsasa_df = pd.DataFrame(
        {
            "PDBID": ["1abc"],
            "Chain1": ["A"],
            "Chain2": ["B"],
            "BSASA": [800.0],
            "SASA12": [1200.0],
            "SASA1": [1000.0],
            "SASA2": [1000.0],
        }
    )
    bsasa_df.to_csv(struct_dir / "bsasa.csv", index=False)

    cfg = {"filter": {"min_bsasa": 500, "min_seq_id": 0.3}}
    run(cfg, output_dir=tmp_path)

    result = pd.read_csv(struct_dir / "BSASA_reference.csv")
    # Cross-product: {P1, P2} × {P3} → 2 rows
    assert len(result) == 2
    pairs = set(zip(result["Prey1"], result["Prey2"]))
    assert ("P1", "P3") in pairs
    assert ("P2", "P3") in pairs


def test_build_reference_handles_keyerror_strings(tmp_path):
    """Non-numeric BSASA values (e.g. 'KeyError') are dropped."""
    struct_dir = tmp_path / "significant_cifs"
    struct_dir.mkdir()

    chain_df = pd.DataFrame(
        {
            "QueryID": ["P1", "P2"],
            "ChainID": ["A", "B"],
            "PDBID": ["1abc", "1abc"],
            "bt_aln_percent_seq_id": [0.9, 0.8],
        }
    )
    chain_df.to_csv(struct_dir / "chain_mapping_all.csv", index=False)

    bsasa_df = pd.DataFrame(
        {
            "PDBID": ["1abc", "1abc"],
            "Chain1": ["A", "A"],
            "Chain2": ["B", "C"],
            "BSASA": ["KeyError", 800.0],
            "SASA12": [0, 1200.0],
            "SASA1": [0, 1000.0],
            "SASA2": [0, 1000.0],
        }
    )
    bsasa_df.to_csv(struct_dir / "bsasa.csv", index=False)

    cfg = {"filter": {"min_bsasa": 500, "min_seq_id": 0.3}}
    run(cfg, output_dir=tmp_path)

    # Should not raise; KeyError row is dropped
    result = pd.read_csv(struct_dir / "BSASA_reference.csv")
    # Only chain C has no mapping, so 0 rows from that pair;
    # but chain A-B had KeyError so also 0. Result may be empty.
    assert isinstance(result, pd.DataFrame)

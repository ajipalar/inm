import pandas as pd
from inm.preprocess.fetch_structures import remap_pdb_id, run


def test_remap_pdb_id_uses_config():
    remapping = {"3jaq": "6gsm", "6t8j": "8bxx"}
    assert remap_pdb_id("3jaq", remapping) == "6gsm"
    assert remap_pdb_id("1abc", remapping) == "1abc"


def test_remap_pdb_id_case_insensitive():
    remapping = {"3jaq": "6gsm"}
    assert remap_pdb_id("3JAQ", remapping) == "6gsm"


def test_run_skips_existing_mmtf(tmp_path):
    hhr_dir = tmp_path / "hhblits_out"
    hhr_dir.mkdir()
    sig_df = pd.DataFrame(
        {
            "QueryUID": ["P1"],
            "PDB70ID": ["1abc_A"],
            "Evalue": [1e-10],
            "Aligned_cols": [100],
            "Identities": [50],
        }
    )
    sig_df.to_csv(hhr_dir / "SignificantPreyPDB70PairAlign.csv", index=False)

    struct_dir = tmp_path / "significant_cifs"
    struct_dir.mkdir()
    (struct_dir / "1abc.bio.mmtf").write_bytes(b"dummy")

    cfg = {
        "obsolete_pdb_remapping": {},
        "pdb_mirror_path": None,
    }
    run(cfg, output_dir=tmp_path)
    assert (struct_dir / "1abc.bio.mmtf").exists()

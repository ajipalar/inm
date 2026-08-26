"""End-to-end test: chain_mapping, compute_bsasa, build_reference, build_benchmark
(steps 8-11).

Uses the CRL5-Vif complex (PDB: 4N9F) as a biologically meaningful fixture, with
no HHblits/PDB70 dependency: the significant-hits table is written by hand and
the query sequences are either real UniProt FASTAs (Vif, CBF-beta) or extracted
directly from the fixture structure itself (Cul5), so the whole chain runs
offline.

Positive case: Vif (P69723) and CBF-beta (Q13951) directly contact in the
crystal (~4556 Å² BSASA) and should end up in both the direct and cocomplex
benchmarks.

Negative case: Cul5 (Q93034) is resolved in the same 4N9F assembly but its
BSASA with every prey-mapped chain falls below the 500 Å² cutoff (~385 Å²
with Vif, ~0 Å² with CBF-beta; its one >=cutoff contact is with the
EloB/EloC chain, which has no prey mapped to it). Real structural data, but
no qualifying evidence of interaction with anything -- Q93034 should be
absent from BSASA_reference.csv and from both benchmark outputs.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from inm.preprocess import (
    build_benchmark,
    build_reference,
    chain_mapping,
    compute_bsasa,
)
from inm.preprocess.config import load_config

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg():
    return load_config(FIXTURE_DIR / "test_config.yaml")


def _setup_intermediate_state(tmp_path: Path) -> None:
    """Set up the intermediate pipeline state needed for steps 8-11."""
    # SignificantPreyPDB70PairAlign.csv: Vif → 4n9f_E, CBF-beta → 4n9f_D,
    # Cul5 → 4n9f_A. The PDB70ID chain suffix is only used to locate the PDB
    # file; chain_mapping.run aligns each prey sequence against every chain
    # in the structure and lets sequence identity decide the real mapping.
    hhr_dir = tmp_path / "hhblits_out"
    hhr_dir.mkdir()
    sig_df = pd.DataFrame(
        {
            "QueryUID": ["P69723", "Q13951", "Q93034"],
            "PDB70ID": ["4n9f_E", "4n9f_D", "4n9f_A"],
            "Probab": [100.0, 100.0, 100.0],
            "Evalue": [1e-100, 1e-100, 1e-100],
            "Score": [999.9, 999.9, 999.9],
            "Aligned_cols": [172, 146, 310],
            "Identities": ["100%", "100%", "100%"],
            "Similarity": [1.5, 1.5, 1.5],
            "Sum_probs": [172.0, 146.0, 310.0],
            "Template_Neff": [1.0, 1.0, 1.0],
            "Q": ["", "", ""],
            "T": ["", "", ""],
        }
    )
    sig_df.to_csv(hhr_dir / "SignificantPreyPDB70PairAlign.csv", index=False)

    # FASTA sequences for Vif, CBF-beta, and Cul5
    seq_dir = tmp_path / "input_sequences"
    seq_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "P69723.fasta", seq_dir / "P69723.fasta")
    shutil.copy(FIXTURE_DIR / "Q13951.fasta", seq_dir / "Q13951.fasta")
    shutil.copy(FIXTURE_DIR / "Q93034.fasta", seq_dir / "Q93034.fasta")

    # 4N9F biological assembly
    cif_dir = tmp_path / "significant_cifs"
    cif_dir.mkdir()
    shutil.copy(FIXTURE_DIR / "4n9f.bio.mmtf", cif_dir / "4n9f.bio.mmtf")


def test_chain_mapping(tmp_path, cfg):
    """Step 8: prey sequences are mapped to PDB chain sequences."""
    _setup_intermediate_state(tmp_path)
    chain_mapping.run(cfg, output_dir=tmp_path)

    cm_path = tmp_path / "significant_cifs" / "chain_mapping_all.csv"
    assert cm_path.exists()
    cm_df = pd.read_csv(cm_path)
    assert len(cm_df) > 0
    assert {"P69723", "Q13951"}.issubset(set(cm_df["QueryID"]))
    assert (cm_df["PDBID"] == "4n9f").all()


def test_compute_bsasa(tmp_path, cfg):
    """Step 9: BSASA is computed for all chain pairs in 4N9F."""
    _setup_intermediate_state(tmp_path)
    compute_bsasa.run(cfg, output_dir=tmp_path)

    bsasa_path = tmp_path / "significant_cifs" / "bsasa.csv"
    assert bsasa_path.exists()
    bsasa_df = pd.read_csv(bsasa_path)
    assert len(bsasa_df) > 0
    assert (bsasa_df["PDBID"] == "4n9f").all()
    # At least one pair should have substantial BSASA (Vif/CBF-beta interface ~4556 Å²)
    assert bsasa_df["BSASA"].max() > 1000


def test_build_reference(tmp_path, cfg):
    """Step 10: BSASA reference is built from chain mapping and BSASA data."""
    _setup_intermediate_state(tmp_path)
    chain_mapping.run(cfg, output_dir=tmp_path)
    compute_bsasa.run(cfg, output_dir=tmp_path)
    build_reference.run(cfg, output_dir=tmp_path)

    ref_path = tmp_path / "significant_cifs" / "BSASA_reference.csv"
    assert ref_path.exists()
    ref_df = pd.read_csv(ref_path)
    assert len(ref_df) > 0

    # Both Vif and CBF-beta should appear
    all_prey = set(ref_df["Prey1"]) | set(ref_df["Prey2"])
    assert "P69723" in all_prey
    assert "Q13951" in all_prey

    # The Vif/CBF-beta pair must appear (they form a direct interface in CRL5)
    pairs = set(zip(ref_df["Prey1"], ref_df["Prey2"])) | set(
        zip(ref_df["Prey2"], ref_df["Prey1"])
    )
    assert ("P69723", "Q13951") in pairs

    # BSASA for Vif/CBF-beta should be substantial
    mask = ((ref_df["Prey1"] == "P69723") & (ref_df["Prey2"] == "Q13951")) | (
        (ref_df["Prey1"] == "Q13951") & (ref_df["Prey2"] == "P69723")
    )
    assert ref_df.loc[mask, "bsasa_lst"].max() > 1000


def test_build_benchmark_end_to_end(tmp_path, cfg):
    """Steps 8-11 end-to-end, from a real structure through the benchmark TSVs.

    Positive: Vif/CBF-beta (P69723/Q13951) directly contact and appear in
    both benchmarks. Negative: Cul5 (Q93034) has no >=cutoff contact with any
    prey-mapped chain and must not appear in the reference table or either
    benchmark, despite being resolved in the same validated PDB.
    """
    _setup_intermediate_state(tmp_path)
    chain_mapping.run(cfg, output_dir=tmp_path)
    compute_bsasa.run(cfg, output_dir=tmp_path)
    build_reference.run(cfg, output_dir=tmp_path)
    build_benchmark.run(cfg, output_dir=tmp_path)

    ref_df = pd.read_csv(tmp_path / "significant_cifs" / "BSASA_reference.csv")
    all_prey = set(ref_df["Prey1"]) | set(ref_df["Prey2"])
    assert "Q93034" not in all_prey

    direct = pd.read_csv(tmp_path / "direct_benchmark.tsv", sep="\t", names=["a", "b"])
    cocomplex = pd.read_csv(
        tmp_path / "cocomplex_benchmark.tsv", sep="\t", names=["a", "b"]
    )
    direct_pairs = set(zip(direct["a"], direct["b"])) | set(
        zip(direct["b"], direct["a"])
    )
    cocomplex_pairs = set(zip(cocomplex["a"], cocomplex["b"])) | set(
        zip(cocomplex["b"], cocomplex["a"])
    )

    # Positive: the direct Vif/CBF-beta interface reaches both benchmarks.
    assert ("P69723", "Q13951") in direct_pairs
    assert ("P69723", "Q13951") in cocomplex_pairs

    # Negative: Cul5 has no qualifying evidence, so it reaches neither.
    assert not any("Q93034" in pair for pair in direct_pairs)
    assert not any("Q93034" in pair for pair in cocomplex_pairs)

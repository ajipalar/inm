import pandas as pd
from inm.preprocess.build_benchmark import run


def _write_ref(struct_dir, rows):
    struct_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows, columns=["Prey1", "Prey2", "PDBID", "Chain1", "Chain2", "bsasa_lst"]
    ).to_csv(struct_dir / "BSASA_reference.csv", index=False)


def test_direct_benchmark_uses_own_pair_bsasa(tmp_path):
    struct_dir = tmp_path / "significant_cifs"
    _write_ref(
        struct_dir,
        [
            ["P1", "P2", "1abc", "A", "B", 800.0],
            ["P1", "P3", "1abc", "A", "C", 100.0],  # below cutoff
        ],
    )

    cfg = {"filter": {"min_bsasa": 500}}
    run(cfg, output_dir=tmp_path)

    direct = pd.read_csv(tmp_path / "direct_benchmark.tsv", sep="\t", names=["a", "b"])
    pairs = set(zip(direct["a"], direct["b"])) | set(zip(direct["b"], direct["a"]))
    assert ("P1", "P2") in pairs
    assert ("P1", "P3") not in pairs
    assert len(direct) == 1


def test_cocomplex_benchmark_includes_indirect_pairs_in_validated_pdb(tmp_path):
    """P2-P3 never has its own >=500 row but co-occurs in a PDB validated by P1-P2."""
    struct_dir = tmp_path / "significant_cifs"
    _write_ref(
        struct_dir,
        [
            ["P1", "P2", "1abc", "A", "B", 800.0],
            ["P2", "P3", "1abc", "B", "C", 200.0],
        ],
    )

    cfg = {"filter": {"min_bsasa": 500}}
    run(cfg, output_dir=tmp_path)

    cocomplex = pd.read_csv(
        tmp_path / "cocomplex_benchmark.tsv", sep="\t", names=["a", "b"]
    )
    pairs = set(zip(cocomplex["a"], cocomplex["b"])) | set(
        zip(cocomplex["b"], cocomplex["a"])
    )
    assert ("P1", "P2") in pairs
    assert ("P2", "P3") in pairs
    assert ("P1", "P3") in pairs  # transitively co-present via chain-to-prey union


def test_cocomplex_benchmark_excludes_unvalidated_pdb(tmp_path):
    """A PDB with no contact >=500 anywhere contributes no cocomplex pairs."""
    struct_dir = tmp_path / "significant_cifs"
    _write_ref(
        struct_dir,
        [
            ["P1", "P2", "1abc", "A", "B", 100.0],
        ],
    )

    cfg = {"filter": {"min_bsasa": 500}}
    run(cfg, output_dir=tmp_path)

    cocomplex = pd.read_csv(
        tmp_path / "cocomplex_benchmark.tsv", sep="\t", names=["a", "b"]
    )
    assert len(cocomplex) == 0


def test_no_self_pairs(tmp_path):
    """Homodimer rows (Prey1 == Prey2) are excluded from both benchmarks."""
    struct_dir = tmp_path / "significant_cifs"
    _write_ref(
        struct_dir,
        [
            ["P1", "P1", "1abc", "A", "B", 800.0],
        ],
    )

    cfg = {"filter": {"min_bsasa": 500}}
    run(cfg, output_dir=tmp_path)

    direct = pd.read_csv(tmp_path / "direct_benchmark.tsv", sep="\t", names=["a", "b"])
    cocomplex = pd.read_csv(
        tmp_path / "cocomplex_benchmark.tsv", sep="\t", names=["a", "b"]
    )
    assert len(direct) == 0
    assert len(cocomplex) == 0

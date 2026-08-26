from inm.preprocess.chain_mapping import mutate_nonstandard_residues


def test_mutate_nonstandard_residues():
    seq = "MKUAJZBXO"
    result = mutate_nonstandard_residues(seq)
    assert "U" not in result
    assert "J" not in result
    assert "Z" not in result
    assert "B" not in result
    assert "O" not in result


def test_mutate_nonstandard_preserves_standard():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    result = mutate_nonstandard_residues(seq)
    assert result == seq

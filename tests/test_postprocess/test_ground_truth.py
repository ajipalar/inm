# tests/test_postprocess/test_ground_truth.py
import numpy as np
import pandas as pd
import pytest
from inm.postprocess.ground_truth import build_refij_from_tsv


@pytest.fixture
def ref_tsv(tmp_path):
    # Two edges: A-B (both in graph), A-C (C not in graph)
    df = pd.DataFrame({"a": ["Q93034", "Q93034"], "b": ["Q15370", "P99999"]})
    p = tmp_path / "ref.tsv"
    df.to_csv(p, sep="\t", index=False, header=False)
    return str(p)


def test_positive_pair(ref_tsv):
    node_idx2name = {0: "Q93034", 1: "Q15370", 2: "P78536"}
    mat = build_refij_from_tsv(ref_tsv, node_idx2name)
    assert mat[0, 1] == 1
    assert mat[1, 0] == 1


def test_out_of_graph_node_ignored(ref_tsv):
    # P99999 is not in node_idx2name — edge should not appear
    node_idx2name = {0: "Q93034", 1: "Q15370", 2: "P78536"}
    mat = build_refij_from_tsv(ref_tsv, node_idx2name)
    assert mat[0, 2] == 0
    assert mat[2, 0] == 0


def test_symmetry(ref_tsv):
    node_idx2name = {0: "Q93034", 1: "Q15370", 2: "P78536"}
    mat = build_refij_from_tsv(ref_tsv, node_idx2name)
    assert np.array_equal(mat, mat.T)


def test_zero_diagonal(ref_tsv):
    node_idx2name = {0: "Q93034", 1: "Q15370", 2: "P78536"}
    mat = build_refij_from_tsv(ref_tsv, node_idx2name)
    assert np.all(np.diag(mat) == 0)


def test_empty_tsv_returns_zero_matrix(tmp_path):
    p = tmp_path / "empty.tsv"
    p.write_text("")
    node_idx2name = {0: "Q93034", 1: "Q15370"}
    mat = build_refij_from_tsv(str(p), node_idx2name)
    assert mat.shape == (2, 2)
    assert np.all(mat == 0)


def test_self_loop_ignored(tmp_path):
    df = pd.DataFrame({"a": ["Q93034"], "b": ["Q93034"]})
    p = tmp_path / "self_loop.tsv"
    df.to_csv(p, sep="\t", index=False, header=False)
    mat = build_refij_from_tsv(str(p), {0: "Q93034", 1: "Q15370"})
    assert mat[0, 0] == 0

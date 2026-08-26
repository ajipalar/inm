# tests/test_postprocess/test_samples.py
import numpy as np
from inm.postprocess.samples import (
    edge_probabilities,
    merge_chains,
    save_average_edge_scores,
)


def test_merge_chains_shape():
    z = np.zeros((4, 10, 5, 5), dtype=np.float32)
    merged = merge_chains(z)
    assert merged.shape == (40, 5, 5)


def test_merge_chains_preserves_values():
    z = np.arange(4 * 10 * 5 * 5, dtype=np.float32).reshape(4, 10, 5, 5)
    merged = merge_chains(z)
    # First element of merged should match first element of chain 0
    assert merged[0, 0, 0] == z[0, 0, 0, 0]
    # Element at index 10 should be first element of chain 1
    assert merged[10, 0, 0] == z[1, 0, 0, 0]


def test_edge_probabilities_high_z():
    # z >> 0.5 → Z2A → ≈ 1.0
    z = np.full((40, 4, 4), 2.0, dtype=np.float32)
    probs = edge_probabilities(z)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (4, 4)
    assert np.allclose(probs, 1.0, atol=1e-3)


def test_edge_probabilities_low_z():
    # z << 0.5 → Z2A → ≈ 0.0
    z = np.full((40, 4, 4), -2.0, dtype=np.float32)
    probs = edge_probabilities(z)
    assert np.allclose(probs, 0.0, atol=1e-3)


def test_edge_probabilities_threshold():
    # z exactly at 0.5 → Z2A → 0.5
    z = np.full((40, 4, 4), 0.5, dtype=np.float32)
    probs = edge_probabilities(z)
    assert np.allclose(probs, 0.5, atol=1e-3)


def test_save_average_edge_scores(tmp_path):
    import pandas as pd

    mean_aij = np.array([[0.0, 0.8, 0.2], [0.8, 0.0, 0.5], [0.2, 0.5, 0.0]])
    node_idx2name = {0: "A00001", 1: "B00002", 2: "C00003"}
    out = tmp_path / "scores.tsv"
    save_average_edge_scores(mean_aij, node_idx2name, out)
    df = pd.read_csv(out, sep="\t")
    assert list(df.columns) == ["node_a", "node_b", "score"]
    # Upper triangle only: (0,1), (0,2), (1,2) → 3 rows for N=3
    assert len(df) == 3
    row = df[(df["node_a"] == "A00001") & (df["node_b"] == "B00002")]
    assert np.isclose(row["score"].values[0], 0.8)


def test_merge_chains_single_chain():
    z = np.zeros((1, 10, 5, 5), dtype=np.float32)
    merged = merge_chains(z)
    assert merged.shape == (10, 5, 5)

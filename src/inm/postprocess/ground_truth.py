"""Build binary ground-truth adjacency matrices from PDB edge list TSVs."""

import numpy as np
import pandas as pd


def build_refij_from_tsv(
    tsv_path: str,
    node_idx2name: dict,
) -> np.ndarray:
    """Build NxN binary ground-truth adjacency matrix from an edge list TSV.

    tsv_path: path to TSV with no header, columns: UniProt_A, UniProt_B
    node_idx2name: {int -> UniProt ID} mapping from corr.tsv row order
    Returns symmetric NxN int32 matrix; 1 where both nodes appear in the TSV.
    """
    try:
        df = pd.read_csv(tsv_path, sep="\t", names=["a", "b"])
    except (pd.errors.EmptyDataError, KeyError):
        N = len(node_idx2name)
        return np.zeros((N, N), dtype=np.int32)

    name2idx = {name: i for i, name in node_idx2name.items()}
    N = len(node_idx2name)
    mat = np.zeros((N, N), dtype=np.int32)
    df["i"] = df["a"].map(name2idx)
    df["j"] = df["b"].map(name2idx)
    valid = df.dropna(subset=["i", "j"])
    valid = valid[valid["i"] != valid["j"]]  # drop self-loops
    rows = valid["i"].astype(int).to_numpy()
    cols = valid["j"].astype(int).to_numpy()
    mat[rows, cols] = 1
    mat[cols, rows] = 1
    return mat

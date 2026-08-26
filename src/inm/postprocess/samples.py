"""Chain merging and edge probability extraction for MCMC samples."""

import numpy as np
import pandas as pd

from inm.model.utils import Z2A, Z2A_soft


def merge_chains(z: np.ndarray) -> np.ndarray:
    """Merge chain and sample axes. (C, S, ...) -> (C*S, ...)."""
    n_chains, n_samples = z.shape[:2]
    return z.reshape(n_chains * n_samples, *z.shape[2:])


def edge_probabilities(z: np.ndarray) -> np.ndarray:
    """Compute mean edge probability from merged z samples.

    z's lower triangle drives the degree/edge-count restraints in
    model23_n_degree_ (see inm.model.model23); its upper triangle is sampled
    independently and unconstrained by them. Average over the lower triangle
    only and mirror, so the reported network matches what those restraints
    actually shape (rather than mixing in the unconstrained upper triangle).

    z: (total_samples, N, N) float array of latent edge scores
    Returns (N, N) float array of mean edge probabilities, symmetric.
    """
    lower_mean = np.tril(np.array(Z2A(z)), k=-1).mean(axis=0)  # (N, N)
    return lower_mean + lower_mean.T


def per_sample_degree(z_merged: np.ndarray, weighted: bool = False) -> np.ndarray:
    """Compute per-sample, per-node degree from merged z samples.

    z_merged: (S, N, N) merged samples.
    weighted: if True use Z2A_soft (continuous in [0,1]); else Z2A thresholded
        at 0.5 (binary 0/1).
    Returns (S, N) ndarray of degrees per node per sample.
    """
    if weighted:
        A = np.array(Z2A_soft(z_merged), dtype=np.float32, copy=True)
    else:
        A = (np.array(Z2A(z_merged)) > 0.5).astype(np.int8)
    N = A.shape[-1]
    diag = np.arange(N)
    A[:, diag, diag] = 0
    A_lower = np.tril(A, k=-1)
    A_sym = A_lower + np.transpose(A_lower, (0, 2, 1))
    return A_sym.sum(axis=2)


def save_average_edge_scores(
    mean_aij: np.ndarray,
    node_idx2name: dict,
    output_path,
) -> None:
    """Save upper-triangle mean edge probabilities to a TSV.

    Columns: node_a, node_b, score
    """
    N = mean_aij.shape[0]
    rows = [
        (node_idx2name[i], node_idx2name[j], float(mean_aij[i, j]))
        for i in range(N)
        for j in range(i + 1, N)
    ]
    pd.DataFrame(rows, columns=["node_a", "node_b", "score"]).to_csv(
        output_path, sep="\t", index=False
    )

"""Diagnostics for how modeling results depend on the amount of MCMC sampling."""

import numpy as np
import sklearn.metrics


def sample_count_checkpoints(n_total: int, n_points: int = 10) -> list:
    """Log-spaced sample-count checkpoints from 1 to n_total, deduplicated."""
    if n_total <= 1:
        return [max(n_total, 1)]
    raw = np.unique(np.geomspace(1, n_total, num=n_points).astype(int))
    return raw.tolist()


def roc_auc_vs_sample_count(
    edge_scores: np.ndarray,
    refij_flat: np.ndarray,
    sample_counts: list,
    n_shuffles: int = 5,
    n_bootstraps: int = 3,
    seed: int = 0,
) -> dict:
    """AUC of the mean predicted edge score vs. how many posterior draws it is pooled over.

    edge_scores: (total_draws, M) continuous per-edge scores (e.g. Z2A(z) on the upper
        triangle), pooled across chains.
    refij_flat: (M,) binary ground-truth labels.
    At each checkpoint, bootstraps a random subset of that many draws, averages their
    edge scores, and scores the real reference against a fixed set of shuffled-reference
    nulls (reused at every checkpoint) so the real AUC can be read against chance.
    Returns dict with sample_counts, mean/std auc, mean/std null_auc.
    """
    rng = np.random.default_rng(seed)
    total_draws = edge_scores.shape[0]
    shuffled_refs = [rng.permutation(refij_flat) for _ in range(n_shuffles)]

    mean_auc, std_auc, mean_null_auc, std_null_auc = [], [], [], []
    for k in sample_counts:
        aucs = []
        null_aucs = []
        for _ in range(n_bootstraps):
            idx = rng.choice(total_draws, size=min(k, total_draws), replace=False)
            pred = edge_scores[idx].mean(axis=0)
            aucs.append(sklearn.metrics.roc_auc_score(refij_flat, pred))
            null_aucs.extend(
                sklearn.metrics.roc_auc_score(shuffled, pred)
                for shuffled in shuffled_refs
            )
        mean_auc.append(float(np.mean(aucs)))
        std_auc.append(float(np.std(aucs)))
        mean_null_auc.append(float(np.mean(null_aucs)))
        std_null_auc.append(float(np.std(null_aucs)))

    return dict(
        sample_counts=list(sample_counts),
        mean_auc=mean_auc,
        std_auc=std_auc,
        mean_null_auc=mean_null_auc,
        std_null_auc=std_null_auc,
    )


def best_potential_energy_vs_sample_count(
    potential_energy: np.ndarray, sample_counts: list
) -> dict:
    """Per-chain minimum potential energy using only the first k samples, for k in sample_counts.

    potential_energy: (n_chains, n_samples) float array.
    Returns dict with sample_counts, mean_best_energy, std_best_energy (mean/std across chains).
    """
    mean_best, std_best = [], []
    for k in sample_counts:
        k = min(k, potential_energy.shape[1])
        best_per_chain = potential_energy[:, :k].min(axis=1)
        mean_best.append(float(best_per_chain.mean()))
        std_best.append(float(best_per_chain.std()))
    return dict(
        sample_counts=list(sample_counts),
        mean_best_energy=mean_best,
        std_best_energy=std_best,
    )


def split_half_edge_scores(edge_scores: np.ndarray, seed: int = 0) -> dict:
    """Split pooled draws into two random halves; return each half's + the full pool's mean.

    edge_scores: (total_draws, M) continuous per-edge scores, pooled across chains.
    Used as a mixing check: independent halves of the posterior giving consistent ROC
    curves against ground truth supports (but does not prove) convergence.
    Returns dict with keys 'a', 'b', 'ab', each an (M,) mean edge-score array.
    """
    rng = np.random.default_rng(seed)
    total_draws = edge_scores.shape[0]
    idx = rng.permutation(total_draws)
    midpoint = total_draws // 2
    return dict(
        a=edge_scores[idx[:midpoint]].mean(axis=0),
        b=edge_scores[idx[midpoint:]].mean(axis=0),
        ab=edge_scores.mean(axis=0),
    )

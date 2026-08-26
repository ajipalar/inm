# tests/test_postprocess/test_sampling_diagnostics.py
import numpy as np
from inm.postprocess.sampling_diagnostics import (
    best_potential_energy_vs_sample_count,
    roc_auc_vs_sample_count,
    sample_count_checkpoints,
    split_half_edge_scores,
)


def test_sample_count_checkpoints_bounds():
    checkpoints = sample_count_checkpoints(1000, n_points=10)
    assert checkpoints[0] == 1
    assert checkpoints[-1] == 1000
    assert checkpoints == sorted(checkpoints)


def test_sample_count_checkpoints_deduplicated():
    checkpoints = sample_count_checkpoints(3, n_points=10)
    assert checkpoints == sorted(set(checkpoints))


def test_sample_count_checkpoints_single_draw():
    assert sample_count_checkpoints(1, n_points=10) == [1]


def test_roc_auc_vs_sample_count_returns_expected_keys():
    rng = np.random.default_rng(0)
    edge_scores = rng.uniform(size=(200, 50))
    refij_flat = rng.integers(0, 2, size=50)
    result = roc_auc_vs_sample_count(edge_scores, refij_flat, [1, 10, 100])
    assert set(result.keys()) == {
        "sample_counts",
        "mean_auc",
        "std_auc",
        "mean_null_auc",
        "std_null_auc",
    }
    assert len(result["mean_auc"]) == 3


def test_roc_auc_vs_sample_count_recovers_signal():
    # edge_scores correlated with refij_flat → real AUC should beat the shuffled null
    rng = np.random.default_rng(0)
    refij_flat = rng.integers(0, 2, size=200)
    edge_scores = refij_flat[None, :] + rng.normal(scale=0.3, size=(500, 200))
    result = roc_auc_vs_sample_count(edge_scores, refij_flat, [500], n_bootstraps=5)
    assert result["mean_auc"][0] > result["mean_null_auc"][0]


def test_best_potential_energy_vs_sample_count_monotonic_nonincreasing():
    # minimum over a growing window can only go down (or stay flat)
    rng = np.random.default_rng(0)
    potential_energy = rng.normal(size=(4, 1000))
    result = best_potential_energy_vs_sample_count(potential_energy, [10, 100, 1000])
    values = result["mean_best_energy"]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_best_potential_energy_vs_sample_count_clips_to_available_samples():
    potential_energy = np.zeros((4, 5))
    result = best_potential_energy_vs_sample_count(potential_energy, [1000])
    assert result["mean_best_energy"] == [0.0]


def test_split_half_edge_scores_shapes_and_disjoint_halves_cover_pool():
    rng = np.random.default_rng(0)
    edge_scores = rng.uniform(size=(101, 20))
    result = split_half_edge_scores(edge_scores)
    assert set(result.keys()) == {"a", "b", "ab"}
    for v in result.values():
        assert v.shape == (20,)


def test_split_half_edge_scores_ab_matches_full_mean():
    rng = np.random.default_rng(0)
    edge_scores = rng.uniform(size=(100, 20))
    result = split_half_edge_scores(edge_scores)
    np.testing.assert_allclose(result["ab"], edge_scores.mean(axis=0))

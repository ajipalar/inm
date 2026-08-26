# tests/test_postprocess/test_convergence.py
import numpy as np
from inm.postprocess.convergence import (
    ess_ranked,
    ess_upper_triangle,
    rhat_ranked,
    rhat_upper_triangle,
)


def test_rhat_upper_triangle_shape():
    # (C, S, N, N) → N*(N-1)//2 values
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 100, 5, 5)).astype(np.float32)
    vals = rhat_upper_triangle(z)
    assert vals.shape == (5 * 4 // 2,)


def test_rhat_upper_triangle_well_mixed():
    # IID samples across chains → R-hat ≈ 1.0
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 500, 5, 5)).astype(np.float32)
    vals = rhat_upper_triangle(z)
    assert np.all(vals < 1.2)


def test_rhat_ranked_length():
    rng = np.random.default_rng(0)
    samples = {"z": rng.normal(size=(4, 100, 5, 5)).astype(np.float32)}
    result = rhat_ranked(samples, top_n=5)
    assert len(result) == 5


def test_rhat_ranked_sorted_descending():
    rng = np.random.default_rng(0)
    samples = {"z": rng.normal(size=(4, 100, 5, 5)).astype(np.float32)}
    result = rhat_ranked(samples, top_n=10)
    values = [v for _, v in result]
    assert values == sorted(values, reverse=True)


def test_rhat_ranked_labels_contain_param_name():
    rng = np.random.default_rng(0)
    samples = {"my_param": rng.normal(size=(4, 100, 3, 3)).astype(np.float32)}
    result = rhat_ranked(samples, top_n=5)
    assert all("my_param" in label for label, _ in result)


def test_ess_upper_triangle_shape():
    # (C, S, N, N) → N*(N-1)//2 values
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 100, 5, 5)).astype(np.float32)
    vals = ess_upper_triangle(z)
    assert vals.shape == (5 * 4 // 2,)


def test_ess_upper_triangle_well_mixed_close_to_total_draws():
    # IID samples across chains → ESS should be close to total draw count
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 1000, 5, 5)).astype(np.float32)
    vals = ess_upper_triangle(z)
    assert np.all(vals > 1000)  # total draws = 4*1000


def test_ess_ranked_length():
    rng = np.random.default_rng(0)
    samples = {"z": rng.normal(size=(4, 100, 5, 5)).astype(np.float32)}
    result = ess_ranked(samples, bottom_n=5)
    assert len(result) == 5


def test_ess_ranked_sorted_ascending():
    rng = np.random.default_rng(0)
    samples = {"z": rng.normal(size=(4, 100, 5, 5)).astype(np.float32)}
    result = ess_ranked(samples, bottom_n=10)
    values = [v for _, v in result]
    assert values == sorted(values)


def test_ess_ranked_labels_contain_param_name():
    rng = np.random.default_rng(0)
    samples = {"my_param": rng.normal(size=(4, 100, 3, 3)).astype(np.float32)}
    result = ess_ranked(samples, bottom_n=5)
    assert all("my_param" in label for label, _ in result)

"""Tests for inm.model.utils, including a vectorized-vs-loop equivalence check
for flat2matrix/matrix2flat (KNOWN_ISSUES.md Engineering item 4)."""

import jax.numpy as jnp
import numpy as np
import pytest

from inm.model.utils import flat2matrix, matrix2flat


def _loop_matrix2flat(M, row_major=True):
    """Original pre-vectorization implementation, kept only as a correctness
    oracle for the vectorized flat2matrix/matrix2flat now in inm.model.utils."""
    n, m = M.shape
    a = jnp.zeros(n * (n - 1) // 2, dtype=M.dtype)
    k = 0
    if row_major:
        for row in range(n):
            length = m - row - 1
            a = a.at[k : k + length].set(M[row, row + 1 : m])
            k += length
    else:
        for col in range(m):
            length = n - col - 1
            a = a.at[k : k + length].set(M[col + 1 : n, col])
            k += length
    return a


def _loop_flat2matrix(a, n):
    """Original pre-vectorization implementation, kept only as a correctness
    oracle for the vectorized flat2matrix/matrix2flat now in inm.model.utils."""
    M = jnp.zeros((n, n))
    k = 0
    for row in range(n):
        length = n - row - 1
        M = M.at[row, row + 1 : n].set(a[k : k + length])
        k += length
    return M + M.T


def _random_symmetric_matrix(n, seed, zero_diag=True):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(-2, 2, size=(n, n)).astype(np.float32)
    M = (vals + vals.T) / 2
    if zero_diag:
        np.fill_diagonal(M, 0)
    return jnp.array(M)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 7, 10, 25])
def test_matrix2flat_matches_loop_reference(n):
    M = _random_symmetric_matrix(n, seed=n)
    got = matrix2flat(M)
    expected = _loop_matrix2flat(M)
    np.testing.assert_array_equal(np.array(got), np.array(expected))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 7, 10, 25])
def test_flat2matrix_matches_loop_reference(n):
    M_len = n * (n - 1) // 2
    rng = np.random.default_rng(100 + n)
    a = jnp.array(rng.uniform(-2, 2, size=M_len).astype(np.float32))
    got = flat2matrix(a, n)
    expected = _loop_flat2matrix(a, n)
    np.testing.assert_array_equal(np.array(got), np.array(expected))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7, 10, 25])
def test_matrix2flat_flat2matrix_roundtrip(n):
    M = _random_symmetric_matrix(n, seed=1000 + n)
    roundtripped = flat2matrix(matrix2flat(M), n)
    np.testing.assert_array_equal(np.array(roundtripped), np.array(M))


def test_matrix2flat_preserves_dtype():
    M = _random_symmetric_matrix(5, seed=0)
    assert matrix2flat(M).dtype == M.dtype

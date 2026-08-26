"""Pure math utilities for the inm.model package."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist


class Histogram(dist.Distribution):
    """Piecewise-constant histogram distribution for the null model."""

    support = dist.constraints.real

    def __init__(self, a, bins, density=True, validate_args=None):
        def histogram_piecewise_constant_pdf(x, bin_edges, bin_heights, n_bins):
            bin_indices = jnp.digitize(x, bin_edges) - 1
            bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)
            NUMERICAL_CONSTANT = 1e-9
            return bin_heights[bin_indices] + NUMERICAL_CONSTANT

        bin_heights, bin_edges = jnp.histogram(a, bins=bins, density=density)
        bin_heights = jnp.array(bin_heights)
        bin_edges = jnp.array(bin_edges)
        self.probs = bin_heights
        self.n_bins = len(bin_heights)
        self.bin_edges = bin_edges
        super(Histogram, self).__init__(
            batch_shape=(), event_shape=(), validate_args=validate_args
        )
        pdf = jax.tree_util.Partial(
            histogram_piecewise_constant_pdf,
            bin_edges=self.bin_edges,
            bin_heights=self.probs,
            n_bins=self.n_bins,
        )
        self.pdf = pdf

    def sample(self, key, sample_shape=()):
        bin_indices = jax.random.categorical(
            key, jnp.log(self.probs), shape=sample_shape
        )
        return self.bin_edges[bin_indices]

    def log_prob(self, value):
        return jnp.log(self.pdf(value))


def Z2A(z: "jnp.ndarray") -> "jnp.ndarray":
    """Map latent z to adjacency probability via hard sigmoid."""
    return jax.nn.sigmoid((z - 0.5) * 1_000)


def Z2A_soft(z: "jnp.ndarray") -> "jnp.ndarray":
    """Soft Z2A: 100x less steep than Z2A (k=10 vs k=1000) so NUTS sees gradient."""
    return jax.nn.sigmoid((z - 0.5) * 10)


def matrix2flat(M: "jnp.ndarray") -> "jnp.ndarray":
    """Flatten upper triangle of a symmetric NxN matrix to 1D array."""
    n, _ = M.shape
    i, j = jnp.triu_indices(n, k=1)
    return M[i, j]


def flat2matrix(a: "jnp.ndarray", n: int) -> "jnp.ndarray":
    """Reconstruct symmetric NxN matrix from upper-triangle flat array."""
    i, j = jnp.triu_indices(n, k=1)
    M = jnp.zeros((n, n)).at[i, j].set(a)
    return M + M.T

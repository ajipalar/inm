"""Smoke tests: run each MODEL_REGISTRY entry with tiny synthetic data."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from inm.model.model23 import MODEL_REGISTRY
from inm.model.utils import matrix2flat


def make_minimal_model_data(N: int = 4) -> dict:
    """Build a minimal model_data dict for smoke testing with N proteins."""
    M = N * (N - 1) // 2
    rng = np.random.default_rng(0)

    # Fake correlation matrices (NxN symmetric, diagonal = 1)
    corr_vals = rng.uniform(0, 1, size=(N, N)).astype(np.float32)
    np.fill_diagonal(corr_vals, 1.0)
    corr_vals = (corr_vals + corr_vals.T) / 2

    # Flat upper-triangle arrays
    corr_flat = matrix2flat(jnp.array(corr_vals))
    shuff_corr_flat = matrix2flat(jnp.array(corr_vals * 0.5))

    # SAINT max-pair score (one value per edge, all-positive for log stability)
    saint_flat = jnp.ones(M, dtype=jnp.float32) * 0.5

    # Fake composite dicts (not used by any current model variant)
    empty_composite: dict = {}

    return {
        "N": N,
        "alpha": 0.02,
        "beta": 0.1,
        "apms_corr_flat": corr_flat,
        "apms_shuff_corr_flat": shuff_corr_flat,
        "apms_shuff_corr_all_flat": shuff_corr_flat,  # global null used by unpack
        "saint_max_pair_score_edgelist": saint_flat,
        "new_composite_dict_norm_approx": empty_composite,
        "new_composite_dict_p_is_1": empty_composite,
        "disconectivity_distance": 4,
        "max_distance": 10,
        "n_null_bins": 10,
    }


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
def test_smoke(model_name):
    model_data = make_minimal_model_data(N=4)
    model_fn = MODEL_REGISTRY[model_name]
    rng_key = jax.random.PRNGKey(0)
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=2, num_samples=2, num_chains=1, progress_bar=False)
    mcmc.run(rng_key, model_data)
    samples = mcmc.get_samples()
    assert "z" in samples

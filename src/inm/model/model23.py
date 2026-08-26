"""NumPyro model functions for the model23_n_p_* family."""

from functools import partial

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from inm.model.utils import Z2A, Z2A_soft, flat2matrix


def model23_n_prior_(model_data):
    """The prior term of model23_n"""
    N = model_data["N"]
    R = jnp.clip(model_data["apms_corr_flat"], 0, 1.0)
    saint = model_data["saint_max_pair_score_edgelist"]

    diag_indices = jnp.diag_indices(N)
    flat2matrix_f = partial(flat2matrix, n=N)

    saint_pairwise = flat2matrix_f(saint).at[diag_indices].set(0)
    R_pairwise = flat2matrix_f(R).at[diag_indices].set(0)

    z_prior_mean = model_data.get("z_prior_mean", 0.1)
    z_prior_scale = model_data.get("z_prior_scale", 1.0)
    z = numpyro.sample("z", dist.Normal(z_prior_mean, z_prior_scale).expand([N, N]))
    return saint_pairwise, R_pairwise, z


def model23_n_saint_(saint_max_pair_score_pairwise_matrix, z):
    """The saint term of model23_n"""
    s_restraint = dist.Normal(
        saint_max_pair_score_pairwise_matrix - 0.5,
        saint_max_pair_score_pairwise_matrix**2 + 1e-2,
    )
    s_score = jnp.sum(s_restraint.log_prob(z))
    numpyro.factor("s_score_", s_score)
    numpyro.deterministic("s_score", -s_score)


def model23_n_r_(R_pairwise_matrix, z):
    """The R term of model23_n"""
    r_z_restraint = dist.Normal(R_pairwise_matrix, 2)
    r_z_score = jnp.sum(r_z_restraint.log_prob(z))
    numpyro.factor("r_z_score_", r_z_score)
    numpyro.deterministic("r_z_score", -r_z_score)

    r_restraint = dist.Normal(R_pairwise_matrix - 0.5, R_pairwise_matrix**2 + 1e-2)
    r_score = jnp.sum(r_restraint.log_prob(z))
    numpyro.factor("r_score_", r_score)
    numpyro.deterministic("r_score", -r_score)


def model23_n_degree_(z, diag_indices, N, model_data):
    """The degree term of model23_n"""
    aij = Z2A(z)
    aij = aij.at[diag_indices].set(0)
    aij = jnp.tril(aij, k=-1)
    aij = aij + aij.T
    degree = jnp.sum(aij, axis=1)

    degree_prior_mean = model_data.get("degree_prior_mean", 3.0)
    degree_prior_scale = model_data.get("degree_prior_scale", 3.0)
    degree_expected = jnp.ones(N) * degree_prior_mean
    degree_restraint = dist.Normal(degree_expected, degree_prior_scale)
    degree_score = jnp.sum(degree_restraint.log_prob(degree))
    numpyro.factor("degree_score_", degree_score)
    numpyro.deterministic("degree_score", -degree_score)


def model23_n_degree_soft_(z, diag_indices, N, model_data):
    """Soft variant of model23_n_degree_ using Z2A_soft (k=10) for differentiability."""
    aij = Z2A_soft(z)
    aij = aij.at[diag_indices].set(0)
    aij = jnp.tril(aij, k=-1)
    aij = aij + aij.T
    degree = jnp.sum(aij, axis=1)

    degree_prior_mean = model_data.get("degree_prior_mean", 3.0)
    degree_prior_scale = model_data.get("degree_prior_scale", 3.0)
    degree_expected = jnp.ones(N) * degree_prior_mean
    degree_restraint = dist.Normal(degree_expected, degree_prior_scale)
    degree_score = jnp.sum(degree_restraint.log_prob(degree))
    numpyro.factor("degree_score_", degree_score)
    numpyro.deterministic("degree_score", -degree_score)


def model23_n_nedges_(degree, model_data):
    """The nedges term of model23_n"""
    n_edges = jnp.sum(degree) / 2
    n_edges_prior_mean = model_data.get("n_edges_prior_mean", 300.0)
    n_edges_prior_scale = model_data.get("n_edges_prior_scale", 100.0)
    n_edges_restraint = dist.Normal(n_edges_prior_mean, n_edges_prior_scale)
    n_edges_score = n_edges_restraint.log_prob(n_edges)
    numpyro.factor("n_edges_score_", n_edges_score)
    numpyro.deterministic("n_edges_score", -n_edges_score)


def model23_n_p(model_data):
    """S(M) = SP(M)"""
    model23_n_prior_(model_data)


def model23_n_p_s(model_data):
    """S(M) = SP(M) + SS(M)"""
    saint_pairwise, _, z = model23_n_prior_(model_data)
    model23_n_saint_(saint_pairwise, z)


def model23_n_p_r(model_data):
    """S(M) = SP(M) + SR(M)"""
    _, R_pairwise, z = model23_n_prior_(model_data)
    model23_n_r_(R_pairwise, z)


def model23_n_p_d(model_data):
    """S(M) = SP(M) + SD(M)"""
    _, _, z = model23_n_prior_(model_data)
    N = model_data["N"]
    diag_indices = jnp.diag_indices(N)
    model23_n_degree_(z, diag_indices, N, model_data)


def model23_n_p_ne(model_data):
    """S(M) = SP(M) + SE(M)"""
    _, _, z = model23_n_prior_(model_data)
    aij = Z2A(z)
    diag_indices = jnp.diag_indices(model_data["N"])
    aij = aij.at[diag_indices].set(0)
    aij = jnp.tril(aij, k=-1)
    aij = aij + aij.T
    degree = jnp.sum(aij, axis=1)
    model23_n_nedges_(degree, model_data)


def model23_n_p_s_r(model_data):
    """S(M) = SP(M) + SS(M) + SR(M)"""
    saint_pairwise, R_pairwise, z = model23_n_prior_(model_data)
    model23_n_saint_(saint_pairwise, z)
    model23_n_r_(R_pairwise, z)


def model23_n_p_s_d(model_data):
    """S(M) = SP(M) + SS(M) + SD(M)"""
    saint_pairwise, _, z = model23_n_prior_(model_data)
    model23_n_saint_(saint_pairwise, z)
    N = model_data["N"]
    diag_indices = jnp.diag_indices(N)
    model23_n_degree_(z, diag_indices, N, model_data)


def model23_n_p_s_ne(model_data):
    """S(M) = SP(M) + SS(M) + SE(M)"""
    saint_pairwise, _, z = model23_n_prior_(model_data)
    model23_n_saint_(saint_pairwise, z)
    aij = Z2A(z)
    diag_indices = jnp.diag_indices(model_data["N"])
    aij = aij.at[diag_indices].set(0)
    aij = jnp.tril(aij, k=-1)
    aij = aij + aij.T
    degree = jnp.sum(aij, axis=1)
    model23_n_nedges_(degree, model_data)


def model23_n_p_r_d(model_data):
    """S(M) = SP(M) + SR(M) + SD(M)"""
    _, R_pairwise, z = model23_n_prior_(model_data)
    N = model_data["N"]
    diag_indices = jnp.diag_indices(N)
    model23_n_degree_(z, diag_indices, N, model_data)
    model23_n_r_(R_pairwise, z)


def model23_n_p_s_r_d(model_data):
    """S(M) = SP(M) + SS(M) + SR(M) + SD(M)"""
    saint_pairwise, R_pairwise, z = model23_n_prior_(model_data)
    model23_n_saint_(saint_pairwise, z)
    model23_n_r_(R_pairwise, z)
    N = model_data["N"]
    diag_indices = jnp.diag_indices(N)
    model23_n_degree_(z, diag_indices, N, model_data)


def model23_n_p_s_r_d_soft(model_data):
    """S(M) = SP(M) + SS(M) + SR(M) + SD(M), with soft Z2A (k=10) in the degree term."""
    saint_pairwise, R_pairwise, z = model23_n_prior_(model_data)
    model23_n_saint_(saint_pairwise, z)
    model23_n_r_(R_pairwise, z)
    N = model_data["N"]
    diag_indices = jnp.diag_indices(N)
    model23_n_degree_soft_(z, diag_indices, N, model_data)


MODEL_REGISTRY: dict = {
    "model23_n_p": model23_n_p,
    "model23_n_p_s": model23_n_p_s,
    "model23_n_p_r": model23_n_p_r,
    "model23_n_p_d": model23_n_p_d,
    "model23_n_p_ne": model23_n_p_ne,
    "model23_n_p_s_r": model23_n_p_s_r,
    "model23_n_p_s_d": model23_n_p_s_d,
    "model23_n_p_s_ne": model23_n_p_s_ne,
    "model23_n_p_r_d": model23_n_p_r_d,
    "model23_n_p_s_r_d": model23_n_p_s_r_d,
    "model23_n_p_s_r_d_soft": model23_n_p_s_r_d_soft,
}

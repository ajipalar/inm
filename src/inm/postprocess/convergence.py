"""MCMC convergence diagnostics."""

import numpy as np


def rhat_lower_triangle(z: np.ndarray) -> np.ndarray:
    """R-hat per edge from (C, S, N, N) z samples.

    z's lower triangle drives the degree/edge-count restraints in
    model23_n_degree_ (see inm.model.model23); its upper triangle is sampled
    independently and unconstrained by them, so the lower triangle is the
    canonical set of edges to report diagnostics for.

    Returns 1D array of R-hat values for lower-triangle edges only (length N*(N-1)//2).
    """
    import numpyro.diagnostics

    rhat_matrix = np.array(numpyro.diagnostics.gelman_rubin(z))  # (N, N)
    N = rhat_matrix.shape[0]
    i, j = np.tril_indices(N, k=-1)
    return rhat_matrix[i, j]


def rhat_ranked(samples: dict, top_n: int = 10) -> list:
    """Compute R-hat for all parameters in samples dict; return top_n highest.

    samples: dict of {param_name: array of shape (C, S, ...)}
    Returns list of (label, rhat_value) tuples sorted descending by rhat_value.
    Label format: 'param' for scalar, 'param[i]' for 1D, 'param[i,j]' for 2D, etc.
    """
    import numpyro.diagnostics

    all_vals = []
    for param_name, param_samples in samples.items():
        rhat = np.array(numpyro.diagnostics.gelman_rubin(param_samples))
        for idx in np.ndindex(rhat.shape):
            if len(idx) == 0:
                label = param_name
            else:
                label = f"{param_name}[{','.join(map(str, idx))}]"
            all_vals.append((label, float(rhat[idx])))
    all_vals.sort(key=lambda x: x[1], reverse=True)
    return all_vals[:top_n]


def ess_lower_triangle(z: np.ndarray) -> np.ndarray:
    """Effective sample size per edge from (C, S, N, N) z samples.

    z's lower triangle drives the degree/edge-count restraints in
    model23_n_degree_ (see inm.model.model23); its upper triangle is sampled
    independently and unconstrained by them, so the lower triangle is the
    canonical set of edges to report diagnostics for.

    Returns 1D array of ESS values for lower-triangle edges only (length N*(N-1)//2).

    Slices to the lower triangle *before* calling effective_sample_size, not
    after: its autocorrelation step is FFT-based and memory-heavy, so running
    it over the full symmetric N*N matrix instead of the N*(N-1)/2 lower
    triangle roughly doubles peak memory for redundant (duplicate/diagonal)
    entries that are discarded anyway.
    """
    import numpyro.diagnostics

    N = z.shape[-1]
    i, j = np.tril_indices(N, k=-1)
    z_lower = z[..., i, j]  # (C, S, N*(N-1)//2)
    return np.array(numpyro.diagnostics.effective_sample_size(z_lower))


def ess_ranked(samples: dict, bottom_n: int = 10) -> list:
    """Compute ESS for all parameters in samples dict; return bottom_n lowest.

    Low ESS (relative to the number of draws) indicates poor mixing, so the
    lowest values are the ones worth surfacing — the mirror image of
    rhat_ranked, which surfaces the highest (worst) R-hat values.

    samples: dict of {param_name: array of shape (C, S, ...)}
    Returns list of (label, ess_value) tuples sorted ascending by ess_value.
    Label format: 'param' for scalar, 'param[i]' for 1D, 'param[i,j]' for 2D, etc.
    """
    import numpyro.diagnostics

    all_vals = []
    for param_name, param_samples in samples.items():
        ess = np.array(numpyro.diagnostics.effective_sample_size(param_samples))
        for idx in np.ndindex(ess.shape):
            if len(idx) == 0:
                label = param_name
            else:
                label = f"{param_name}[{','.join(map(str, idx))}]"
            all_vals.append((label, float(ess[idx])))
    all_vals.sort(key=lambda x: x[1])
    return all_vals[:bottom_n]

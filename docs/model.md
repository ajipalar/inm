# The Scoring Function `S(M)`

This document derives and motivates each term of the Bayesian network scoring
function implemented in `inm.model.model23`. It closes out KNOWN_ISSUES.md
Scientific items 3 and 10: the prior hyperparameters already live in
`config.yaml` (see below); this file explains *why* those values, and why
each restraint term takes the algebraic form it does.

## 1. Representation: a network as a latent adjacency matrix

A candidate interaction network `M` over `N` proteins is represented as a
symmetric `N x N` adjacency matrix `A`, `A[i,j] in {0,1}`, stored internally
as its upper-triangle flat vector of length `M_pairs = N(N-1)/2`
(`matrix2flat`/`flat2matrix`, `inm/src/inm/model/utils.py`).

Rather than sampling `A` directly (a discrete space HMC/NUTS cannot move
through), the model samples a continuous latent score `z` per pair and maps
it to an edge through a steep sigmoid centered at `0.5`:

```python
def Z2A(z):
    return jax.nn.sigmoid((z - 0.5) * 1_000)   # hard: edge iff z > 0.5
def Z2A_soft(z):
    return jax.nn.sigmoid((z - 0.5) * 10)      # soft: continuous edge weight
```

`Z2A` (k=1000) is close enough to a step function that `z = 0.5` is, for all
practical purposes, the decision boundary: `z` above it is an edge, below it
is not. `Z2A_soft` (k=10, used by the `*_soft` model variants) trades that
sharp boundary for a gradient NUTS can actually follow — see the README's
"Sampling near the Z2A boundary" section for the sampling-efficiency
consequence of that choice.

Every restraint below is therefore, in effect, a statement about where `z`
should sit relative to `0.5`.

The overall score is a sum of independent NumPyro `factor` log-densities
(`s_score_`, `r_z_score_`, `r_score_`, `degree_score_`, `n_edges_score_`),
not a properly normalized joint likelihood:

```
S(M) = SP(M) + SS(M) + SR(M) + SD(M) + SE(M)
```

| Term | Function | Evidence | In production model? |
|---|---|---|---|
| `SP` | `model23_n_prior_` | none (structural prior on `z`) | yes |
| `SS` | `model23_n_saint_` | SAINT max pairwise score | yes |
| `SR` | `model23_n_r_` | AP-MS abundance correlation `R` | yes |
| `SD` | `model23_n_degree_` / `_soft` | none (sparsity prior on degree) | yes |
| `SE` | `model23_n_nedges_` | none (sparsity prior on edge count) | only in `*_ne`/`*_p_s_ne` variants |

The production default is `model.name: model23_n_p_s_r_d` in
`examples/cullin_example/config.yaml`, i.e. `S(M) = SP + SS + SR + SD` — `SE`
exists as an alternative/additional sparsity term but is not in the default
composition (see §5 for why `SD` alone is arguably sufficient).

## 2. `SP(M)` — structural prior on `z`

```python
z ~ Normal(z_prior_mean=0.1, z_prior_scale=1.0)   # per pair, iid
```

With the hard `Z2A`, `P(edge) = P(z > 0.5)` under this prior alone is
`1 - Phi((0.5 - 0.1)/1.0) ≈ 0.345` per pair *before* any evidence — that
sounds permissive, but multiplied across `N(N-1)/2` pairs it still implies a
dense random graph unless the evidence terms actively suppress most pairs.
The real work of enforcing sparsity is not done here; it's done by `SS`/`SR`
pulling most pairs' `z` well below `0.5` (§4) and by `SD`/`SE` constraining
the aggregate (§3). `SP` itself is deliberately weak and symmetric around a
point slightly below the threshold — a mild, non-committal prior that leaves
the decision to the data.

## 3. `SD(M)` and `SE(M)` — sparsity restraints

```python
degree_i = sum_j Aij            # Aij = Z2A(z), symmetrized, diagonal zeroed
degree_i ~ Normal(degree_prior_mean=3.0, degree_prior_scale=3.0)   # per node

n_edges = sum(degree) / 2
n_edges ~ Normal(n_edges_prior_mean=300.0, n_edges_prior_scale=100.0)
```

Both encode the same underlying assumption — that a cullin-RING ligase
interactome is sparse, not hub-dominated — at two different resolutions:
`SD` constrains it node-by-node (mean degree 3, generous scale 3 so any
individual node can range roughly 0-12 without much penalty), `SE`
constrains it in aggregate (mean 300 edges, scale 100), independent of how
degree is distributed across nodes.

These two priors are internally consistent, which is a useful sanity check
on the calibration: for the `N = 236`-node minimal cullin set (`saint_cutoff
= 0.5`, see README), the handshake lemma gives an expected edge count from
`SD` alone of `degree_prior_mean * N / 2 = 3 * 236 / 2 = 354`, which is
within one `n_edges_prior_scale` (100) of `SE`'s own mean of 300. The two
terms are not fighting each other. Because of that redundancy, and because
`SE` is a single scalar restraint (one data point) against `SD`'s `N`
restraints, `SE` adds little independent constraint on top of `SD` at this
`N` — plausibly why the production model (`model23_n_p_s_r_d`) omits it and
only the `*_ne` variants (`SP` + `SE` with no `SD`) use `SE` on its own.

## 4. `SS(M)` and `SR(M)` — evidence restraints, and their shared design pattern

Both the SAINT term and one half of the correlation term follow the same
algebraic shape:

```python
restraint = Normal(score - 0.5, score**2 + 1e-2)
factor    = sum(restraint.log_prob(z))
```

for `score in {saint_max_pairwise, R_pairwise}` (`R` is the AP-MS
correlation matrix, clipped to `[0, 1]` in `model23_n_prior_`). The `1e-2`
term is a numerical floor preventing a zero-variance (infinite log-density)
Gaussian when `score = 0`.

This form has two effects that only make sense together, and are worth
naming explicitly since they are not obvious from the code alone:

- **Mean, `score - 0.5`.** Shifting by `0.5` re-expresses the raw `[0, 1]`
  score in `z`'s coordinate system, where `0.5` is the `Z2A` edge threshold.
  `score = 1` centers the restraint exactly *at* the threshold; `score = 0`
  centers it at `-0.5`, well inside the "no edge" plateau.
- **Variance, `score**2 + 1e-2`.** This is the load-bearing, non-obvious
  choice: variance *grows* with the score. A low score produces a narrow
  Gaussian pinned near `-0.5` — a strong, confident push toward "no edge". A
  high score produces a wide, comparatively uninformative Gaussian centered
  right at the decision boundary — a weak, permissive nudge that says "this
  pair is *allowed* to be an edge" rather than "this pair *is* an edge".

The asymmetry is a negative-design restraint: absence of SAINT/correlation
support is treated as strong evidence against an interaction, while its
presence is treated as necessary-but-not-sufficient, requiring corroboration
from the other restraint terms (`SR`/`SS`, `SD`, `SE`) before `z` actually
crosses `0.5`. `SS` alone cannot place an edge with high confidence; it can
only decline to rule one out. The same pattern is proposed for a future
Predictomes/SPOC restraint in `docs/external_priors.md` ("Centering at
`target - 0.5` mirrors the SAINT term"), so this is the established
house style for evidence restraints in this codebase, not a one-off.

`SR(M)` additionally carries a second, unrelated factor:

```python
r_z_restraint = Normal(R_pairwise, 2)          # note: R, not R - 0.5
r_z_score = sum(r_z_restraint.log_prob(z))
```

This treats `R` as if it were directly a noisy observation of `z` itself (no
`-0.5` shift, fixed variance 2, `std ≈ 1.41` — wide relative to the unit
interval `R` lives in). Read generously, it is a second, independent
evidence channel: the score-scaled term above says "how confidently does `R`
argue for/against crossing the edge threshold", while this term says
"roughly where does `R` suggest `z` should sit on its own numeric scale".

**This is the weakest-motivated piece of `S(M)`, and the audit should say so
plainly rather than force a post-hoc justification**: two different
functional forms of the same evidence (`R`) are summed into `z`'s posterior
with no documented derivation for why both are needed, what `Normal(R, 2)`'s
prior predictive implies on top of `Normal(R-0.5, R^2+1e-2)`'s, or whether
`std = sqrt(2)` was tuned or a placeholder. It is exactly the "grew by
accretion" pattern flagged in KNOWN_ISSUES.md. Recommended follow-up (not
done here, to avoid changing model behavior/posteriors as a side effect of a
documentation pass): either (a) run `model23_n_p_r` with and without the
`r_z_restraint` factor and compare posterior edge probabilities against the
PDB benchmark ROC to see whether it does anything the score-scaled term
doesn't, or (b) if it demonstrably changes results, document the intended
role explicitly instead of leaving it implicit.

## 5. What this does and does not establish

This document motivates each term's *functional form* from the model code
and config, and cross-checks `SD`/`SE` for mutual consistency. It does not
constitute a prior predictive check — i.e. it does not show what network
distribution these priors actually imply once all terms interact (KNOWN_ISSUES.md
Scientific item 12, still open). The handshake-lemma check in §3 is a
first-order consistency argument, not a substitute for sampling from the
prior and inspecting the resulting degree/edge-count distribution directly.

## References

- `inm/src/inm/model/model23.py` — restraint implementations.
- `inm/src/inm/model/utils.py` — `Z2A`, `Z2A_soft`, `flat2matrix`/`matrix2flat`.
- `inm/examples/cullin_example/config.yaml` (`model.priors` block) — prior
  hyperparameters and their one-line rationale.
- `inm/docs/external_priors.md` — the SAINT-restraint pattern applied
  prospectively to a new evidence channel (Predictomes/SPOC).
- `inm/README.md` — "Sampling near the Z2A boundary" (hard vs. soft `Z2A`
  trade-off).

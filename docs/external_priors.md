# External PPI Priors

External protein–protein-interaction (PPI) prediction sources that can supply prior information to the inm scoring function. Each source has its own coverage, calibration, and missingness semantics that must be respected when wiring it into `S(M)`.

## Predictomes (Schmidt et al., *Mol Cell* 2025)

### What it is

Predictomes is a curated database of AlphaFold-Multimer-modeled human PPIs with classifier-derived confidence scores. The full pair-score dump used here is `data/predictome/20251110_hs_predictome_pair_scores.csv` (1,614,047 rows). Each row is one unordered UniProt pair plus eight score columns:

| Column | Range | Source | Notes |
|---|---|---|---|
| `kirc_score` | [0, 1] | KIRC classifier | Knowledge-informed prior used for *selecting* pairs into AF-M |
| `spoc_score` | [0, 1] | SPOC classifier | Structure-and-omics classifier; the headline confidence score |
| `num_unique_contacts` | int | AF-M structure | Inter-chain residue contacts in the predicted complex |
| `string_db_score` | [0, 1000] | STRING DB | Combined evidence score (any channel) |
| `string_db_physical_score` | [0, 1000] | STRING DB | Physical-binding subset of the score |
| `biogrid_detect_count` | int | BioGRID | Number of times the pair has been detected experimentally |
| `intact_db_evidence_count` | int | IntAct | Number of evidence entries |
| `in_pdb` | {0, 1} | PDB | Whether a co-crystal structure exists |

Distributions of all eight columns over the full 1.6 M pairs:
[`examples/cullin_example/results/predictomes_histograms.png`](../examples/cullin_example/results/predictomes_histograms.png)

### Selection pipeline (critical for missingness)

1. ~200 M possible human pairs (≈20 k proteins, all unordered pairs).
2. KIRC ranks all 200 M by knowledge-based prior evidence.
3. The **top 1.6 M by KIRC** are submitted to AlphaFold-Multimer. This is the file we have.
4. Pairs with combined length > ~3600 aa are dropped before AF-M (compute constraint).
5. SPOC scores the 1.6 M AF-modeled pairs.
6. The "high-confidence" curated set is ~16 k pairs at SPOC ≥ 0.86 (≈90% precision, ≈27% recall at that threshold).

So the 1.6 M file is a **prefiltered candidate set, not a thresholded result set**. A pair absent from the file means KIRC ranked it outside the top 1.6 M (or it was too long), *not* that AF-M judged it negative. Reported KIRC recall at the cutoff used is ~68%, so roughly one third of true interactions are absent from the file because of the prefilter — they were never folded.

### Coverage against the cullin protein sets

Computed by `snakemake predictomes_all` (rules in `examples/cullin_example/rules/predictomes.smk`):

| Set | N nodes | N pairs (`N·(N−1)/2`) | Predictomes-covered pairs | Fraction covered |
|---|---:|---:|---:|---:|
| Cullin **full** (all formatted prey) | 3,063 | 4,689,453 | 154,270 | **3.3%** |
| Cullin **minimal** (SAINT ≥ 0.5, used in NUTS) | 236 | 27,730 | 1,677 | **6.0%** |

The minimal set covers a higher fraction because SAINT pre-filtering enriches for biologically credible prey, which KIRC also tends to rank highly.

### Score-distribution observations

From the histograms over all 1.6 M pairs:

- **`spoc_score`**: strongly bimodal with most mass near 0 and a sparse tail toward 1. Behaves like a calibrated classifier — only a small fraction of pairs receive any meaningful positive score. This is the cleanest signal in the table.
- **`kirc_score`**: smoother distribution with most mass in [0.1, 0.4]. Less discriminating because it is a *prior* over biological features, not a posterior given a structure.
- **`string_db_score`** / **`string_db_physical_score`**: spike at 0 (no STRING evidence) and a second bump near 1000 (high-confidence STRING entries) with sparse middle. Useful as a binary-ish secondary channel.
- **`num_unique_contacts`**, **`biogrid_detect_count`**, **`intact_db_evidence_count`**: heavy long tails; most pairs have zero or one piece of evidence. Treat as auxiliary metadata.
- **`in_pdb`**: ~1% of the 1.6 M pairs have a co-crystal structure. Sparse but extremely strong positive evidence when present.

### Interpreting SPOC vs. KIRC

These are *not interchangeable*:

- **KIRC** is a knowledge prior. It says "given the orthogonal biological evidence (co-expression, GO, BioGRID, etc.), is this pair likely to interact?" It was trained without seeing AF-M output. Using KIRC as a prior in our model would partially double-count whatever evidence channels overlap with our own data (especially BioGRID/IntAct if we were to add them).
- **SPOC** is a posterior given AF-M structure plus omics. It says "given that AF-M predicted *this* complex and the omics support it, is this pair likely to be a true interaction?" SPOC adds the *structural* evidence channel that our AP-MS-based model does not have.

For our purposes, **SPOC is the load-bearing column**. KIRC, STRING, BioGRID, and IntAct columns largely encode what either our SAINT term already covers or what would create dependence between "prior" and "data" channels.

### How to wire this into `S(M)`

The current model has `S(M) = SP + SS + SR + SD + SE`. Adding a Predictomes channel `S_PRED(M)` — sketch:

**Three-state encoding** (the central design choice):

- *Present + SPOC ≥ τ_high* → strong positive evidence: contributes a Gaussian factor centered on a high `aij` value.
- *Present + SPOC < τ_low* → mild negative evidence: AF-M was run and did not find a confident interface, so this is *some* evidence the pair does not interact (but not strong evidence — SPOC's recall is ~27% at the high-confidence threshold).
- *Absent from file* → **no factor contributed**. This is essential. Encoding absence as "score = 0" treats the KIRC pre-filter as ground truth, which it is not.

A reasonable concrete form, matching the existing pattern in `model23_n_saint_`:

```python
# pred_score: shape (M,), filled with SPOC for covered pairs, NaN for absent pairs
# Skip the NaN entries via a mask.
mask = ~jnp.isnan(pred_score)
target = jnp.where(mask, pred_score, 0.0)
sigma = jnp.where(mask, pred_sigma, jnp.inf)  # inf σ → zero log-prob contribution
spoc_restraint = dist.Normal(target - 0.5, sigma)
spoc_log_like = jnp.sum(spoc_restraint.log_prob(z) * mask)
numpyro.factor("spoc_score_", spoc_log_like)
```

Notes on the form above:

- Centering at `target - 0.5` mirrors the SAINT term so the restraint is "high SPOC → push z above 0.5", consistent with `Z2A`'s edge interpretation.
- `pred_sigma` should be looser than the SAINT σ — SPOC is calibrated as a binary classifier, not as a regression target on `z`. A reasonable starting choice: `sigma = 1.0` constant, or `sigma = (1 - spoc_score) + 0.1` to weight high-SPOC pairs more strongly.
- The `mask`-out-then-multiply pattern keeps the JAX trace static while contributing zero log-likelihood for absent pairs.

**What not to do:**

- Do not add a KIRC-derived factor in addition to SPOC — KIRC is upstream of SPOC and overlaps with priors we already get from BioGRID/IntAct, risking double-counting.
- Do not include `string_db_score` and SPOC in the same restraint without thinking about correlation — STRING is an input channel to KIRC, which is the gate to SPOC.
- Do not encode `in_pdb = 0` as negative evidence. PDB absence reflects experimental tractability, not absence of interaction.

**A defensible composition:**

- Primary: SPOC (3-state encoding above).
- Optional secondary, orthogonal channel: `in_pdb = 1` as a hard-positive prior on the affected pairs only (still no negative contribution from `in_pdb = 0`).
- All other Predictomes columns: ignore for the prior; keep them as auxiliary metadata for downstream interpretation.

### References

- Paper: Schmidt et al. (2025). *Mol Cell.* https://doi.org/10.1016/j.molcel.2025.01.034
- Database: https://predictomes.org
- Snakemake rules in this repo: `examples/cullin_example/rules/predictomes.smk`
- Coverage outputs: `examples/cullin_example/results/predictomes_overlap_{full,minimal}_summary.txt`

---

## (stub) Burke et al. 2023 — AlphaFold-Multimer human PPI screen

Reserved. See `examples/cullin_example/rules/burke.smk` for fetch + coverage rules.

## (stub) Sequence-based predictors (D-SCRIPT, Topsy-Turvy)

Reserved. Candidate gap-filling tools when neither Predictomes nor Burke covers a pair of interest.

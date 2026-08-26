# Integrative Network Modeling

Integrative network modeling (INM) is a method and framework (library) for modeling biological interaction networks. The INM library supports the modeling of protein interactions networks based on AP-MS data.
INM can:
- Predict direct protein-protein interaction (PPI) networks for all pairs of protein types (matrix model) on the order of hundreds or perhaps thousands of nodes
- Predict condition specific PPI networks (e.g., gene deletion, chemical perturbation, cell-type)
- Predict a distribution over networks to estimate both the average network and ascociated uncertainty
- Has a scoring funciton that may be extended to other types of information that may inform PPI networks (e.g, deep learning based PPI prediction, proximity labeling)

Modeling proceeds in 5 stages: (i) gathering AP-MS protein spectral counts under multiple conditions, (ii) representing a model where nodes represent protein types
and edges represent a protein interaction between at least one pair of molecules of the corresponding types, (iii) a scoring function
that scores the agreement between AP-MS spectral counts and a network model, (iv) sampling alternative configurations of edges using an MCMC scheme,
(v) analysis of the output sample of network models.

inm includes an example system and benchmarks against Protein Data Bank and Humap 2.0 derived interactions. INM is a part of the Integrative Modeling Platform ([IMP](https://github.com/salilab/imp/)) module.

## Installation

To clone the repository use the following command
```bash
git clone https://github.com/ajipalar/inm.git
```

Create and activate the conda environment
```bash
cd inm
conda env create -f environment.yml
conda activate py39
```

## Run tests
```bash
python3 -m unittest discover -s tests
```

## Usage

Pipeline stages (preprocess, analyze, filter, model, postprocess) are run through Snakemake, not by calling `scripts/*.py` directly — Snakemake's DAG ensures downstream stages re-run when their inputs change and outputs land in the right `results/{model.name}/` directories.

## Example: Cullin AP-MS

A minimal reproducible example of the INM workflow applied to the cullin AP-MS dataset (Wan et al., *Cell Host & Microbe*, 2019), in [`examples/cullin_example/`](examples/cullin_example/).

This example runs the full INM pipeline end-to-end:

1. **Preprocess** — load raw AP-MS spectral counts from Excel, format into spectral count and composite tables
2. **Analyze** — exploratory figures (spectral count distributions, SAINT score summaries)
3. **Filter** — retain prey with max SAINT score ≥ 0.5; compute pairwise Pearson correlations
4. **Model** — run HMC/NUTS sampling via NumPyro/JAX to infer a posterior over protein interaction networks
5. **Postprocess** — sampling diagnostics (R-hat, effective sample size), edge probability estimates, ROC curves against PDB references

The model used is `model23_n_p_s_r_d`: a Bayesian network scoring function combining a Gaussian prior on the latent interaction variable *z*, a SAINT score restraint, a pairwise correlation restraint, and a per-node degree restraint. Networks are represented as symmetric adjacency matrices; edges are inferred via a hard sigmoid transformation of *z*. Each term is derived and motivated in [`docs/model.md`](docs/model.md).

Evaluated against PDB-derived interaction references, the model achieves:
- **Direct interactions AUROC: 0.756**
- **Co-complex interactions AUROC: 0.712**

using 4 chains × 1000 sampling steps (500 warmup).

### Prerequisites

- `inm` package installed (`conda activate py39`)
- Snakemake (`conda install -c bioconda snakemake`)

### Data Setup

The raw AP-MS Excel file is not included in the repo. The `fetch_cullin` Snakemake rule
attempts to download it automatically; the publisher (Cell Host & Microbe) blocks
automated requests (Cloudflare), so this will usually fail with instructions to
download it manually from [Wan et al. 2019](https://doi.org/10.1016/j.chom.2019.05.008)
(Table S2 / mmc2.xlsx) and place it at:

```
data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx
```

Re-run `snakemake` once the file is in place; the checksum is verified automatically.

### Running

```bash
cd examples/cullin_example
snakemake postprocess_all --cores 1
```

Outputs are written to `results/model23_n_p_s_r/`. The model stage (~20 min on CPU) dominates runtime.

### Running with Docker (CPU-only)

A CPU-only image is available as an alternative to the conda environment. It covers the
core pipeline (preprocess → model → postprocess → analyze).

```bash
docker build -t inm:cpu .
```

The raw AP-MS xlsx (see Data Setup above) and the `results/` directory should live on the
host, not baked into the image, so bind-mount them:

```bash
cd examples/cullin_example
docker run --rm \
  -v "$(pwd)/data:/app/examples/cullin_example/data" \
  -v "$(pwd)/results:/app/examples/cullin_example/results" \
  inm:cpu all
```

The image's entrypoint is `snakemake --cores all`; the trailing argument (`all`,
`postprocess_all`, `-n` for a dry run, ...) is passed straight to it.

### Configuration

Key parameters in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `model23_n_p_s_r` | Model variant (see `inm.model.model23.MODEL_REGISTRY`) |
| `model.num_warmup` | 500 | HMC warmup steps per chain |
| `model.num_samples` | 1000 | Posterior samples per chain |
| `model.num_chains` | 4 | Independent Markov chains |
| `filter.saint_cutoff` | 0.5 | Minimum SAINT score to retain a prey |

Outputs are keyed by `model.name`, so changing the model name and re-running preserves prior results. To run a different model variant, set `model.name` in `examples/cullin_example/config.yaml` (or pass `--config model.name=...` on the command line).

### Outputs

```
results/{model_name}/
├── model/
│   ├── samples.npz              # Posterior samples (chains × steps × N × N)
│   ├── extra_fields.npz         # Potential energy (sampling phase)
│   └── warmup_extra_fields.npz  # Potential energy (warmup + sampling)
└── postprocess/
    ├── 10_roc_curve.png                     # ROC curves vs. PDB references
    ├── 11_edge_prob_hist.png                # Edge probability distribution
    ├── 12_degree_distribution.png           # Node degree distribution (threshold 0.5)
    ├── 13_rhat_histogram.png                # R-hat convergence diagnostic
    ├── 14_rhat_rank_plot.png                # Top 10 highest R-hat values
    ├── 15_energy_trace.png                  # Potential energy (sampling only)
    ├── 16_energy_trace_with_warmup.png      # Potential energy (warmup + sampling)
    ├── 17_ground_truth_matrices.png         # Direct and co-complex reference matrices
    ├── 18_example_sampled_matrices.png      # Four random posterior adjacency matrices
    ├── 19_average_edge_probability_matrix.png  # Mean edge probability over all samples
    ├── 20_edge_variance_matrix.png          # Edge variance p*(1-p)
    ├── 21_pooled_binary_degree_posterior.png   # Pooled posterior binary degree distribution
    ├── 22_pooled_weighted_degree_posterior.png # Pooled posterior weighted degree distribution
    ├── 23-25_node_{low,mid,high}_binary_degree_posterior.png    # Per-node binary degree posteriors
    ├── 26-28_node_{low,mid,high}_weighted_degree_posterior.png  # Per-node weighted degree posteriors
    ├── 30_ess_histogram.png                 # Effective sample size distribution across edges
    ├── 31_ess_rank_plot.png                 # Bottom 10 lowest ESS values
    └── average_edge_scores.tsv             # Per-pair mean edge probabilities
```

### Sampling diagnostics

Convergence and mixing are assessed two ways, both in `inm.postprocess.convergence`:

- **R-hat** (`rhat_lower_triangle`, `rhat_ranked`) — Gelman-Rubin statistic per edge and top-10 highest across all sampled parameters. Values near 1.0 indicate the chains agree; `> 1.1` is the conventional cutoff.
- **Effective sample size** (`ess_lower_triangle`, `ess_ranked`) — how many independent draws the autocorrelated chain is worth, per edge and bottom-10 lowest across all parameters. Figures flag edges below 10% of total draws.

Both operate on the `(num_chains, num_samples, ...)` arrays NumPyro returns from a single multi-chain `MCMC.run` call — see `config.yaml`'s `model.num_chains`.

### Structure

```
examples/cullin_example/
├── Snakefile
├── config.yaml
├── rules/
│   ├── preprocess.smk
│   ├── analyze.smk
│   ├── model.smk
│   └── postprocess.smk
└── scripts/
    ├── fetch_cullin.py
    ├── preprocess.py
    ├── analyze.py
    ├── filter.py
    ├── correlations.py
    ├── model.py
    └── postprocess.py
```

### Data Attribution

Cullin AP-MS data: Wan et al. (2019) [https://doi.org/10.1016/j.chom.2019.05.008](https://doi.org/10.1016/j.chom.2019.05.008)

## External priors

External PPI prediction sources (Predictomes, Burke et al., sequence-based predictors) and how they should be wired into the scoring function `S(M)` — including coverage analyses against the cullin example and the missingness semantics that determine the right encoding — are documented in [`docs/external_priors.md`](docs/external_priors.md).

## Limitations

### Sampling near the Z2A boundary

The default `Z2A` map (`sigmoid((z - 0.5) * 1000)`) is effectively a step function: NUTS gets gradient signal only in a band of width ~0.004 around `z = 0.5`, so HMC trajectories typically leap across the boundary instead of integrating through it. A softened variant (`Z2A_soft`, k=10, available in the `*_soft` model variants such as `model23_n_p_s_r_d_soft`) speeds up sampling by **at least ~4x** because gradients flow continuously through the degree and edge-count restraints. The trade-off is interpretive: with the soft sigmoid, an edge is no longer a discrete on/off variable but a **continuous edge weight** in `(0, 1)`, and downstream summaries (degree, total edges, posterior edge probabilities) need to be read accordingly.

### Cullin example: under-sampling and scale

Sampling is likely not exhaustive. Runs of 20,000 steps over fewer than 40 chains produced a higher direct-interactions AUROC (~0.82) compared to the cullin example's AUROC of 0.76, suggesting the 4-chain × 1000-step configuration used in the cullin example undersamples the posterior and may underestimate edge probabilities for weaker interactions. Results from the cullin example should be treated as illustrative rather than definitive.

The full cullin dataset contains 236 prey after filtering. The N×N adjacency matrix grows quadratically, and runtime scales accordingly. The example is intended for demonstration on a single CPU; production runs used HPC resources.

## Citation

INM is based on the dissertation:

> Palar, A. (2024). *Modeling direct protein interaction networks from mass spectrometry data*. UCSF. ProQuest ID: Palar_ucsf_0034D_12855. Merritt ID: ark:/13030/m5k75z67. Retrieved from https://escholarship.org/uc/item/4t37f9dn

BibTeX:
```bibtex
@phdthesis{palar2024modeling,
  author = {Palar, Aji},
  title  = {Modeling direct protein interaction networks from mass spectrometry data},
  school = {University of California, San Francisco},
  year   = {2024},
  url    = {https://escholarship.org/uc/item/4t37f9dn},
  note   = {ProQuest ID: Palar\_ucsf\_0034D\_12855; Merritt ID: ark:/13030/m5k75z67}
}
```

## AI Usage
AI was not used for hypothesis generation, project design, manuscript preperation, initial implementation, analysis, or figure generation. AI was used for refactoring and testing. Beyond refactoring and testing, AI tools were not used for programming, with one exception: the Predictomes comparison analysis (`examples/cullin_example/scripts/predictomes_*.py` and the accompanying Snakemake rules) was implemented with AI assistance.

# PDB PPI Benchmark Pipeline

Regenerates the two ground-truth benchmark edge lists used by
`examples/cullin_example`'s ROC evaluation:

- `direct_benchmark.tsv` — prey pairs with a direct chain-pair contact
  (buried SASA >= `filter.min_bsasa`, 500 Å² by default) in a shared PDB
  structure.
- `cocomplex_benchmark.tsv` — prey pairs co-present in a PDB structure that
  has at least one contact >= the cutoff (includes prey pairs whose own
  chains never directly touched, but that were resolved in the same
  validated complex).

Both are derived from `BSASA_reference.csv`, a cross of PDB structural data
against the prey list from the same cullin AP-MS dataset used by
`cullin_example`. See `inm/src/inm/preprocess/build_benchmark.py` for the
exact derivation.

## Pipeline stages

```
fetch_cullin -> parse_apms -> fetch_sequences -> add_sequences -> run_hhblits
  -> parse_hhr -> filter_hits -> fetch_structures -> {chain_mapping, compute_bsasa}
  -> build_reference -> build_benchmark -> [publish_benchmark]
```

Each stage is a thin Snakemake wrapper around an `inm.preprocess` CLI step
(`python -m inm.preprocess <step> --config config.yaml --output-dir data`).

## Prerequisites

This pipeline touches external structural-biology infrastructure that isn't
installed or downloaded by Snakemake itself:

- **HHblits** (HH-suite) on `PATH`, plus a local PDB70 HHblits database
  (`config.yaml`: `hhblits.db_path`).
- **PDB structure files.** `fetch_structures` only *converts* CIFs already
  present under `data/significant_cifs/` (or reachable via
  `config.yaml`: `pdb_mirror_path`, a local rsync PDB mirror) — it does not
  fetch from RCSB itself. You need one of those two in place.
- **Network access** to UniProt (`fetch_sequences`) and to the publisher for
  the cullin AP-MS xlsx (`fetch_cullin`; the publisher blocks automated
  downloads via Cloudflare — on failure it prints manual-download
  instructions).

Given these, this pipeline is not meant to run start-to-finish in CI or on a
fresh clone in minutes — it's for regenerating the benchmark when the
upstream PDB data or the BSASA cutoff changes, on a machine with the above
already set up.

## Usage

```bash
cd examples/pdb_benchmark
snakemake -j4
```

This regenerates `data/direct_benchmark.tsv` and `data/cocomplex_benchmark.tsv`
in this example's own sandbox — it does **not** touch the tracked repo copy.

To overwrite the tracked benchmark TSVs consumed by `cullin_example`
(`inm/data/processed/references/pdb_ppi_prediction/*.tsv`), run the publish
step explicitly:

```bash
snakemake publish_benchmark
```

This is a separate step on purpose: it overwrites tracked, shared files that
downstream ROC figures depend on, so it shouldn't happen silently as part of
a default `snakemake` run.

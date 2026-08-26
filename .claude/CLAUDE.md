# inm Project Instructions

## Running Jobs

- **Always prefer Snakemake to invoke pipeline stages** (preprocess, model, postprocess) rather than calling the underlying `scripts/*.py` directly with `python`.
- Snakemake's DAG ensures dependent rules (e.g. postprocess after model) are triggered automatically and outputs are placed in the right `results/{MODEL_NAME}/...` directories.
- Running scripts directly bypasses the DAG, so downstream rules won't fire and outputs may be missed or stale.
- To run a different model variant, change `model.name` in `config.yaml` (or use `snakemake --config model.name=...`) and run `snakemake all` from `examples/cullin_example/`.

## Pipeline Organization

- **One DAG, not parallel pipelines.** When integrating an external tool or comparison source (e.g. D-SCRIPT), add its rules to the same Snakemake workflow rather than maintaining a separate Snakefile. This is what lets Snakemake propagate changes through comparison plots and any model that consumes the external tool's output.
- Split rules across files by concern (e.g. `rules/inm.smk`, `rules/dscript.smk`, `rules/compare.smk`) and pull them in via `include:` directives — one logical DAG, separate namespaces.
- Place external inputs/outputs under `data/external/<tool>/` so provenance is obvious.
- Only use a separate Snakefile when the pipelines are genuinely independent: incompatible envs, very different scheduling needs, or vendoring someone else's repo. The cost is losing automatic re-runs across the boundary.

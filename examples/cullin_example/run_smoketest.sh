#!/usr/bin/env bash
# Smoke-test run of the cullin minimal pipeline: exercises every stage
# (preprocess -> model -> postprocess -> analyze) with the cheapest registered
# model (model23_n_p) and a drastically reduced sample budget (2 chains x 10
# samples) so the full DAG runs in seconds instead of ~20 min. This is for
# verifying output *format* (files, shapes, figure layout) only — the
# posterior estimates are not statistically meaningful.
#
# Writes to results/model23_n_p/ (separate from a real model23_n_p_s_r_d run's
# results/model23_n_p_s_r_d/, so it never clobbers real output).
#
# Uses --cores 1 (serial rule execution) rather than --cores all: on a
# memory-constrained runner (e.g. a 2 GB Colima VM) letting analyze_data and
# run_model execute concurrently risks an OOM kill of one of them. The DAG is
# small enough that serializing costs only a few seconds.
#
# Snakemake's --config/--configfile overrides only affect path construction
# (results/{model.name}/...) — every rule's shell command passes the literal
# "config.yaml" to scripts/*.py, which reload it straight from disk, bypassing
# Snakemake's merged config entirely. So the override in config.smoketest.yaml
# is applied by patching config.yaml in place for the run, then restoring the
# original from a backup on exit (even on failure/interrupt).
set -euo pipefail
cd "$(dirname "$0")"

cp config.yaml config.yaml.smoketest_backup
trap 'mv config.yaml.smoketest_backup config.yaml' EXIT

python3 - <<'PY'
import yaml

def deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base

with open("config.yaml") as f:
    base = yaml.safe_load(f)
with open("config.smoketest.yaml") as f:
    override = yaml.safe_load(f)
with open("config.yaml", "w") as f:
    yaml.safe_dump(deep_merge(base, override), f, sort_keys=False)
PY

snakemake all --cores 1 "$@"

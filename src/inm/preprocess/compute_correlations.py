"""Compute Pearson correlations and shuffled-null correlations from a spectral count table."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pandas as pd


def calc_shuff_correlations(
    spec_table: pd.DataFrame, rseed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return observed corr, shuffled corr, and the shuffled table used as null."""
    null_sc = spec_table.copy()
    nrows, ncolumns = null_sc.shape
    temp = np.zeros(null_sc.shape)
    key = jax.random.PRNGKey(rseed)
    for rid in range(nrows):
        key, k1 = jax.random.split(key)
        shuffled_row = jax.random.permutation(k1, null_sc.iloc[rid, :].values)
        temp[rid] = shuffled_row
    null_sc.loc[:, :] = temp
    shuff_corr = np.corrcoef(null_sc.values, rowvar=True)
    corr = np.corrcoef(spec_table.values, rowvar=True)
    shuff_corr = pd.DataFrame(
        shuff_corr, columns=spec_table.index, index=spec_table.index
    )
    corr = pd.DataFrame(corr, columns=spec_table.index, index=spec_table.index)
    return corr, shuff_corr, null_sc


def run(cfg: dict, output_dir: str | Path) -> None:
    """Compute and write correlation matrices for the spectral count table."""
    output_dir = Path(output_dir)

    spec_table = pd.read_csv(
        output_dir / "spec_table.tsv", sep="\t", index_col=0, header=None
    )

    model_cfg = cfg.get("model", {})
    rseed = model_cfg.get("shuff_rseed", 0)
    rseed_all = model_cfg.get("shuff_all_rseed", 1)

    corr, shuff_corr, _null_sc = calc_shuff_correlations(spec_table, rseed)

    # Global null: shuffle with a different seed
    _corr_all, shuff_corr_all, _null_all = calc_shuff_correlations(
        spec_table, rseed_all
    )

    corr.to_csv(output_dir / "corr.tsv", sep="\t")
    shuff_corr.to_csv(output_dir / "shuff_corr.tsv", sep="\t")
    shuff_corr_all.to_csv(output_dir / "shuff_corr_all.tsv", sep="\t")

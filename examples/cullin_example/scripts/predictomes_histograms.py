"""
Plot histograms of every Predictomes score column into a single PNG figure.

Score columns are taken from config["predictomes"]["score_cols"]. Bounded
[0, 1] columns get linear y-axis; integer count-style columns get log y-axis
because they are heavy-tailed.

Data: Schmidt et al. (2025). Mol Cell. https://doi.org/10.1016/j.molcel.2025.01.034
https://predictomes.org
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


_LOG_Y_COLUMNS = {
    "num_unique_contacts",
    "string_db_score",
    "string_db_physical_score",
    "biogrid_detect_count",
    "intact_db_evidence_count",
    "in_pdb",
}


def main(config_path: str, predictomes_path: str, output_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    score_cols = cfg["predictomes"]["score_cols"]

    df = pd.read_csv(predictomes_path, usecols=score_cols)
    n = len(df)

    n_cols = 4
    n_rows = math.ceil(len(score_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, score_cols):
        s = df[col].dropna()
        ax.hist(s, bins=60, color="steelblue", edgecolor="black", linewidth=0.3)
        ax.set_title(f"{col}  (n={len(s)})")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        if col in _LOG_Y_COLUMNS:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    for ax in axes[len(score_cols) :]:
        ax.axis("off")

    fig.suptitle(f"Predictomes score distributions  (N={n} pairs)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python predictomes_histograms.py <config.yaml> <predictomes_csv> <output_png>"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

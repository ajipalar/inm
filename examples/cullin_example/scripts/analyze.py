"""
Generate exploratory figures for cullin AP-MS dataset.
Figures: spectral count distribution, prey coverage, ground truth network.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def plot_spectral_count_distribution(
    spec_table: pd.DataFrame, output_path: Path, dpi: int = 300
) -> None:
    """
    Plot distribution of spectral counts across prey.
    Shows: histogram of total spec counts per prey.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sum spectral counts across conditions for each prey
    total_specs = spec_table.sum(axis=1)

    ax.hist(total_specs, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Total Spectral Counts (per prey)")
    ax.set_ylabel("Number of Prey")
    ax.set_title("Cullin AP-MS: Spectral Count Distribution")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Saved spectral count distribution to {output_path}")
    plt.close(fig)


def plot_spec_table_heatmap(
    spec_table: pd.DataFrame, output_path: Path, dpi: int = 300
) -> None:
    """
    Plot heatmap of spectral count matrix (first 50 prey x all conditions).
    Shows: which conditions detected which prey.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot top 50 prey by total spec count
    top_prey = spec_table.sum(axis=1).nlargest(50).index
    subset = spec_table.loc[top_prey]

    im = ax.imshow(subset.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xlabel("Condition / Replicate")
    ax.set_ylabel("Prey (top 50)")
    ax.set_title("Cullin AP-MS: Spectral Count Heatmap (top 50 prey)")

    fig.colorbar(im, ax=ax, label="Spectral Counts")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Saved heatmap to {output_path}")
    plt.close(fig)


def plot_bait_prey_network(
    composites: pd.DataFrame, ground_truth: list, output_path: Path, dpi: int = 300
) -> None:
    """
    Plot bipartite network of baits vs. prey, highlighting ground truth members.
    Shows: which proteins interact with which baits, colored by ground truth.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Count how many times each prey appears
    prey_counts = composites["Prey"].value_counts().head(20)

    # Color by whether prey is in ground truth
    colors = [
        "red" if prey in ground_truth else "lightblue" for prey in prey_counts.index
    ]

    ax.barh(
        range(len(prey_counts)), prey_counts.values, color=colors, edgecolor="black"
    )
    ax.set_yticks(range(len(prey_counts)))
    ax.set_yticklabels(prey_counts.index, fontsize=9)
    ax.set_xlabel("Number of Bait Interactions")
    ax.set_title(
        "Cullin AP-MS: Top Prey by Interaction Count\n(Red = Ground Truth Cullin Members)"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Saved network plot to {output_path}")
    plt.close(fig)


def plot_ground_truth_comparison(
    composites: pd.DataFrame, ground_truth: list, output_path: Path, dpi: int = 300
) -> None:
    """
    Plot SAINT scores for ground truth vs. non-ground truth interactions.
    Shows: whether ground truth interactions have high confidence.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    gt_scores = composites[composites["Prey"].isin(ground_truth)]["MSscore"]
    other_scores = composites[~composites["Prey"].isin(ground_truth)]["MSscore"]

    ax.hist(
        [other_scores, gt_scores],
        bins=20,
        label=["Other Prey", "Ground Truth"],
        color=["lightblue", "red"],
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xlabel("SAINT Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Cullin AP-MS: SAINT Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Saved ground truth comparison to {output_path}")
    plt.close(fig)


def plot_pairwise_corr_distribution(
    spec_table: pd.DataFrame, output_path: Path, dpi: int = 300
) -> None:
    """
    Plot distribution of pairwise Pearson correlations across all prey.
    Shows: histogram of Pearson r values for all prey pairs.
    """
    logger.info(
        "Computing pairwise correlations for %d prey (may take ~30s)...",
        len(spec_table),
    )
    corr = spec_table.T.corr()
    n = len(corr)
    upper = corr.values[np.triu_indices(n, k=1)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(upper, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.set_ylabel("Count")
    ax.set_title("Cullin AP-MS: Pairwise Correlation Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Saved pairwise correlation distribution to {output_path}")
    plt.close(fig)


def plot_shuffled_corr_distribution(
    spec_table: pd.DataFrame, output_path: Path, dpi: int = 300, seed: int = 0
) -> None:
    """
    Plot distribution of pairwise Pearson correlations after row-shuffling (null distribution).
    Shows: null histogram of Pearson r values after per-row shuffling.
    """
    rng = np.random.default_rng(seed)
    arr = spec_table.values.copy().astype(float)
    for row in arr:
        rng.shuffle(row)
    shuffled = pd.DataFrame(arr, index=spec_table.index, columns=spec_table.columns)
    logger.info(
        "Computing shuffled pairwise correlations for %d prey (may take ~30s)...",
        len(spec_table),
    )
    corr = shuffled.T.corr()
    n = len(corr)
    upper = corr.values[np.triu_indices(n, k=1)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(upper, bins=50, color="orange", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.set_ylabel("Count")
    ax.set_title("Cullin AP-MS: Shuffled Pairwise Correlation Distribution (Null)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Saved shuffled correlation distribution to {output_path}")
    plt.close(fig)


def plot_topn_corr_matrix(
    spec_table: pd.DataFrame,
    composites: pd.DataFrame,
    prey_gene_map: Dict[str, str],
    output_path: Path,
    dpi: int = 300,
    top_n: int = 50,
) -> None:
    """
    All-pairs Pearson correlation heatmap for top N prey by max SAINT score.
    Shows: correlation structure among the highest-confidence interactors.
    """
    max_saint = composites.groupby("Prey")["MSscore"].max()
    top_prey = [p for p in max_saint.nlargest(top_n).index if p in spec_table.index][
        :top_n
    ]
    if not top_prey:
        logger.warning(
            "No prey found in spec_table matching top-%d by SAINT score; skipping %s",
            top_n,
            output_path,
        )
        return
    labels = [prey_gene_map.get(p, p) for p in top_prey]
    sub = spec_table.loc[top_prey]
    corr = sub.T.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Top {top_n} Prey by SAINT Score: Pairwise Correlation Matrix")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Saved top-{top_n} correlation matrix to {output_path}")
    plt.close(fig)


def plot_topn_shuffled_corr_matrix(
    spec_table: pd.DataFrame,
    composites: pd.DataFrame,
    prey_gene_map: Dict[str, str],
    output_path: Path,
    dpi: int = 300,
    top_n: int = 50,
    seed: int = 0,
) -> None:
    """
    Shuffled all-pairs Pearson correlation heatmap for top N prey by max SAINT score (null).
    Shows: null baseline — same prey subset as the real matrix but with per-row shuffling.
    """
    max_saint = composites.groupby("Prey")["MSscore"].max()
    top_prey = [p for p in max_saint.nlargest(top_n).index if p in spec_table.index][
        :top_n
    ]
    if not top_prey:
        logger.warning(
            "No prey found in spec_table matching top-%d by SAINT score; skipping %s",
            top_n,
            output_path,
        )
        return
    labels = [prey_gene_map.get(p, p) for p in top_prey]
    sub = spec_table.loc[top_prey]
    rng = np.random.default_rng(seed)
    arr = sub.values.copy().astype(float)
    for row in arr:
        rng.shuffle(row)
    shuffled = pd.DataFrame(arr, index=sub.index, columns=sub.columns)
    corr = shuffled.T.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(
        f"Top {top_n} Prey by SAINT Score: Shuffled Pairwise Correlation Matrix (Null)"
    )
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Saved top-{top_n} shuffled correlation matrix to {output_path}")
    plt.close(fig)


def plot_top50_ground_truth_matrix(
    spec_table: pd.DataFrame,
    composites: pd.DataFrame,
    prey_gene_map: Dict[str, str],
    ground_truth: list,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    Binary adjacency matrix: (i,j)=1 when both prey_i and prey_j are ground truth members.
    Shows: which top-50 prey pairs are both known cullin complex members.
    """
    max_saint = composites.groupby("Prey")["MSscore"].max()
    top50 = [p for p in max_saint.nlargest(50).index if p in spec_table.index][:50]
    if not top50:
        logger.warning(
            "No prey found in spec_table matching top-50 by SAINT score; skipping %s",
            output_path,
        )
        return
    labels = [prey_gene_map.get(p, p) for p in top50]
    is_gt = [prey_gene_map.get(p, p) in ground_truth for p in top50]
    n = len(top50)
    gt_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if is_gt[i] and is_gt[j]:
                gt_matrix[i, j] = 1.0
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(gt_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Top 50 Prey by SAINT Score: Ground Truth Adjacency Matrix")
    ax.set_xlabel("Prey")
    ax.set_ylabel("Prey")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    logger.info(f"Saved top-50 ground truth matrix to {output_path}")
    plt.close(fig)


def main(
    config_path: str,
    spec_table_path: str,
    composite_table_path: str,
    prey_names_path: str,
    output_dir: str,
) -> None:
    """
    Generate exploratory figures.

    Args:
        config_path: Path to config.yaml
        spec_table_path: Path to spec_table.tsv
        composite_table_path: Path to composite_table.tsv
        prey_names_path: Path to prey_names.tsv
        output_dir: Directory to save figures
    """
    # Convert to Path objects
    config_path = Path(config_path)
    spec_table_path = Path(spec_table_path)
    composite_table_path = Path(composite_table_path)
    prey_names_path = Path(prey_names_path)
    output_dir = Path(output_dir)

    # Validate input files exist
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    if not spec_table_path.exists():
        logger.error(f"Spec table file not found: {spec_table_path}")
        sys.exit(1)
    if not composite_table_path.exists():
        logger.error(f"Composite table file not found: {composite_table_path}")
        sys.exit(1)
    if not prey_names_path.exists():
        logger.error(f"Prey names file not found: {prey_names_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration with error handling
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {config_path}: {e}")
        sys.exit(1)

    # Validate required config keys
    try:
        ground_truth = config["data"]["ground_truth_members"]
        dpi = config["analysis"]["figure_dpi"]
        top_n = config["analysis"].get("top_n", 50)
    except (KeyError, TypeError) as e:
        logger.error(f"Missing required config key: {e}")
        sys.exit(1)

    # Load TSV files
    try:
        spec_table = pd.read_csv(spec_table_path, sep="\t", index_col=0, header=0)
        composites = pd.read_csv(composite_table_path, sep="\t")
        prey_names_df = pd.read_csv(prey_names_path, sep="\t")
    except Exception as e:
        logger.error(f"Error loading data files: {e}")
        sys.exit(1)

    prey_gene_map: Dict[str, str] = dict(
        zip(prey_names_df["UniProt"], prey_names_df["GeneName"])
    )

    # Normalize ground_truth to match prey gene name format (e.g. CUL1 -> CUL1_HUMAN if that form exists)
    gene_names_set = set(prey_gene_map.values())
    ground_truth = [
        g + "_HUMAN" if g + "_HUMAN" in gene_names_set else g for g in ground_truth
    ]

    # Convert ground_truth gene names to UniProt IDs for legacy functions
    ground_truth_uniprots = [
        k for k, v in prey_gene_map.items() if v in set(ground_truth)
    ]

    logger.info(
        f"Loaded spec_table ({len(spec_table)} prey) and {len(composites)} composite entries"
    )

    # Generate figures
    plot_spectral_count_distribution(
        spec_table, output_dir / "01_spec_count_distribution.png", dpi=dpi
    )
    plot_spec_table_heatmap(
        spec_table, output_dir / "02_spec_table_heatmap.png", dpi=dpi
    )
    plot_bait_prey_network(
        composites,
        ground_truth_uniprots,
        output_dir / "03_prey_interaction_count.png",
        dpi=dpi,
    )
    plot_ground_truth_comparison(
        composites,
        ground_truth_uniprots,
        output_dir / "04_saint_score_distribution.png",
        dpi=dpi,
    )
    plot_pairwise_corr_distribution(
        spec_table, output_dir / "05_pairwise_corr_distribution.png", dpi=dpi
    )
    plot_shuffled_corr_distribution(
        spec_table, output_dir / "06_shuffled_corr_distribution.png", dpi=dpi
    )
    plot_topn_corr_matrix(
        spec_table,
        composites,
        prey_gene_map,
        output_dir / "07_topn_corr_matrix.png",
        dpi=dpi,
        top_n=top_n,
    )
    plot_top50_ground_truth_matrix(
        spec_table,
        composites,
        prey_gene_map,
        ground_truth,
        output_dir / "08_top50_ground_truth_matrix.png",
        dpi=dpi,
    )
    plot_topn_shuffled_corr_matrix(
        spec_table,
        composites,
        prey_gene_map,
        output_dir / "09_topn_shuffled_corr_matrix.png",
        dpi=dpi,
        top_n=top_n,
    )

    logger.info(f"All exploratory figures saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 6:
        print(
            "Usage: python analyze.py <config.yaml> <spec_table.tsv> <composite_table.tsv> <prey_names.tsv> <output_dir>"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

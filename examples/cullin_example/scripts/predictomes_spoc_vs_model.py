"""Compare the AP-MS model's average edge score against thresholded spoc_score.

This flips the roles of the earlier direct/cocomplex PDB benchmark: instead of
treating a PDB-structural edge list as ground truth (which AlphaFold-Multimer,
and so spoc_score, may have partly been trained on), spoc_score itself becomes
the reference by thresholding it at several cutoffs, and the model's average
edge probability (from postprocess.py's average_edge_scores.tsv) is the
predictor. This measures agreement between two independent evidence channels
(AP-MS spectral counts vs. an AF-M + omics classifier), not accuracy against
biological truth — a pair counted "positive" at threshold tau just means
spoc_score >= tau, restricted to pairs both Predictomes and the model score
(the minimal 236-node cullin set).

Predictomes data: Schmidt et al. (2025). Mol Cell.
https://doi.org/10.1016/j.molcel.2025.01.034 / https://predictomes.org
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import yaml

MIN_POSITIVES = 10


def _load_node_set(
    prey_list_path: str, prey_col: str, prey_has_header: bool
) -> set[str]:
    if prey_has_header:
        df = pd.read_csv(prey_list_path, sep="\t")
        col = df[prey_col]
    else:
        col_idx = int(prey_col)
        df = pd.read_csv(prey_list_path, sep="\t", header=None)
        col = df[col_idx]
    return set(col.astype(str).str.strip())


def _load_predictomes_pairs(
    predictomes_path: str, ids_col: str, sep: str, score_col: str
) -> pd.DataFrame:
    """Load normalized (a <= b) unordered Predictomes pairs with their score."""
    df = pd.read_csv(predictomes_path, usecols=[ids_col, score_col])
    parts = df[ids_col].astype(str).str.split(sep, n=1, expand=True)
    parts.columns = ["a_raw", "b_raw"]
    parts["score"] = df[score_col]
    parts = parts.dropna()
    parts["a_raw"] = parts["a_raw"].str.strip()
    parts["b_raw"] = parts["b_raw"].str.strip()
    parts = parts[parts["a_raw"] != parts["b_raw"]]

    a = parts["a_raw"].to_numpy()
    b = parts["b_raw"].to_numpy()
    pair_a = pd.Series([min(x, y) for x, y in zip(a, b)], name="a")
    pair_b = pd.Series([max(x, y) for x, y in zip(a, b)], name="b")
    out = pd.concat(
        [pair_a, pair_b, parts["score"].rename(score_col).reset_index(drop=True)],
        axis=1,
    )
    out = out.drop_duplicates(subset=["a", "b"]).reset_index(drop=True)
    return out


def _load_average_edge_scores(path: str) -> pd.DataFrame:
    """Load node_a, node_b, score TSV and normalize to (a <= b)."""
    df = pd.read_csv(path, sep="\t")
    a = df["node_a"].astype(str).str.strip().to_numpy()
    b = df["node_b"].astype(str).str.strip().to_numpy()
    pair_a = pd.Series([min(x, y) for x, y in zip(a, b)], name="a")
    pair_b = pd.Series([max(x, y) for x, y in zip(a, b)], name="b")
    out = pd.concat([pair_a, pair_b, df["score"].rename("model_score")], axis=1)
    out = out.drop_duplicates(subset=["a", "b"]).reset_index(drop=True)
    return out


def main(
    config_path: str,
    predictomes_path: str,
    average_edge_scores_path: str,
    prey_list_path: str,
    prey_col: str,
    prey_has_header_str: str,
    thresholds_str: str,
    output_dir: str,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pcfg = cfg["predictomes"]
    ids_col = pcfg["uniprot_ids_col"]
    sep = pcfg["uniprot_ids_sep"]
    dpi = cfg.get("analysis", {}).get("figure_dpi", 150)

    prey_has_header = prey_has_header_str.lower() == "true"
    node_set = _load_node_set(prey_list_path, prey_col, prey_has_header)
    thresholds = [float(t) for t in thresholds_str.split(",")]

    spoc = _load_predictomes_pairs(predictomes_path, ids_col, sep, "spoc_score")
    spoc = spoc[spoc["a"].isin(node_set) & spoc["b"].isin(node_set)]

    model = _load_average_edge_scores(average_edge_scores_path)

    merged = spoc.merge(model, on=["a", "b"], how="inner")
    print(f"Pairs with both spoc_score and model average edge score: {len(merged)}")

    y_score = merged["model_score"].to_numpy()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    fig_roc, ax_roc = plt.subplots()
    fig_prc, ax_prc = plt.subplots()
    cmap = plt.get_cmap("viridis")

    for i, tau in enumerate(thresholds):
        y_true = (merged["spoc_score"].to_numpy() >= tau).astype(int)
        n_pos = int(y_true.sum())
        color = cmap(i / max(1, len(thresholds) - 1))
        if n_pos < MIN_POSITIVES or n_pos > len(y_true) - MIN_POSITIVES:
            print(f"tau={tau}: n_pos={n_pos}/{len(y_true)}, skipping (too few pos/neg)")
            continue

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        ax_roc.plot(
            fpr, tpr, color=color, label=f"τ={tau:g} (AUC={roc_auc:.3f}, n_pos={n_pos})"
        )

        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_score)
        ap = sklearn.metrics.average_precision_score(y_true, y_score)
        ax_prc.plot(
            recall,
            precision,
            color=color,
            label=f"τ={tau:g} (AP={ap:.3f}, n_pos={n_pos})",
        )

        summary_rows.append(
            {
                "spoc_threshold": tau,
                "n_pairs": len(y_true),
                "n_positive": n_pos,
                "roc_auc": roc_auc,
                "average_precision": ap,
            }
        )
        print(
            f"tau={tau}: n_pos={n_pos}/{len(y_true)}, ROC AUC={roc_auc:.3f}, AP={ap:.3f}"
        )

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(
        "ROC: model avg edge score vs thresholded spoc_score (minimal set)"
    )
    ax_roc.legend(fontsize=8)
    fig_roc.tight_layout()
    fig_roc.savefig(out_dir / "spoc_vs_model_roc.png", dpi=dpi)
    plt.close(fig_roc)

    ax_prc.set_xlabel("Recall")
    ax_prc.set_ylabel("Precision")
    ax_prc.set_title(
        "PRC: model avg edge score vs thresholded spoc_score (minimal set)"
    )
    ax_prc.legend(fontsize=8)
    fig_prc.tight_layout()
    fig_prc.savefig(out_dir / "spoc_vs_model_prc.png", dpi=dpi)
    plt.close(fig_prc)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "spoc_vs_model_summary.tsv", sep="\t", index=False)

    # --- Compact AUC(tau) / AP(tau) trend figure, with n_positive annotated ---
    fig, ax1 = plt.subplots()
    ax1.plot(
        summary["spoc_threshold"],
        summary["roc_auc"],
        "o-",
        color="tab:blue",
        label="ROC AUC",
    )
    ax1.plot(
        summary["spoc_threshold"],
        summary["average_precision"],
        "s-",
        color="tab:orange",
        label="Avg. Precision",
    )
    ax1.set_xlabel("spoc_score threshold (τ)")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    for _, row in summary.iterrows():
        ax1.annotate(
            f"n={int(row['n_positive'])}",
            (row["spoc_threshold"], row["roc_auc"]),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7,
            ha="center",
        )
    ax1.set_title("Model-vs-SPOC agreement across spoc_score thresholds (minimal set)")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spoc_vs_model_threshold_trend.png", dpi=dpi)
    plt.close(fig)

    print(f"Wrote ROC/PRC/trend figures and summary to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print(
            "Usage: python predictomes_spoc_vs_model.py <config.yaml> <predictomes_csv> "
            "<average_edge_scores.tsv> <prey_list> <prey_col> <prey_has_header:true|false> "
            "<comma_separated_thresholds> <output_dir>"
        )
        sys.exit(1)
    main(*sys.argv[1:])

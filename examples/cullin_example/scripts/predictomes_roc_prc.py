"""ROC and PRC curves using the Predictomes spoc_score as a predictor.

Reference (positive-label) sets are the PDB-structural direct and cocomplex
interaction benchmarks (config["data"]["direct_reference"] /
["cocomplex_reference"]). The evaluation universe is every unordered
Predictomes pair with both endpoints in a cullin node set (label "full" or
"minimal", same node sets used by predictomes_overlap.py); a pair counts as
a reference positive if it appears in the benchmark TSV, regardless of
whether either endpoint is in the node set.

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


def _load_node_set(
    prey_list_path: str, prey_col: str, prey_has_header: bool
) -> set[str]:
    if prey_has_header:
        df = pd.read_csv(prey_list_path, sep="\t")
        if prey_col not in df.columns:
            raise KeyError(
                f"Column '{prey_col}' not found in {prey_list_path}. "
                f"Found: {list(df.columns)}."
            )
        col = df[prey_col]
    else:
        # No header: prey_col is a 0-indexed column number as string.
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
    out = pd.concat([pair_a, pair_b, parts["score"].reset_index(drop=True)], axis=1)
    out = out.drop_duplicates(subset=["a", "b"]).reset_index(drop=True)
    return out


def _load_reference_pairs(ref_path: str) -> set[tuple[str, str]]:
    """Load a headerless two-column benchmark TSV as a set of normalized pairs."""
    try:
        df = pd.read_csv(ref_path, sep="\t", header=None, names=["a", "b"], dtype=str)
    except pd.errors.EmptyDataError:
        return set()
    df["a"] = df["a"].str.strip()
    df["b"] = df["b"].str.strip()
    pairs = set()
    for x, y in zip(df["a"], df["b"]):
        if x == y:
            continue
        pairs.add((min(x, y), max(x, y)))
    return pairs


def main(
    config_path: str,
    predictomes_path: str,
    prey_list_path: str,
    prey_col: str,
    prey_has_header_str: str,
    label: str,
    output_dir: str,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pcfg = cfg["predictomes"]
    ids_col = pcfg["uniprot_ids_col"]
    sep = pcfg["uniprot_ids_sep"]
    score_col = "spoc_score"
    dpi = cfg.get("analysis", {}).get("figure_dpi", 150)

    prey_has_header = prey_has_header_str.lower() == "true"
    node_set = _load_node_set(prey_list_path, prey_col, prey_has_header)

    pairs = _load_predictomes_pairs(predictomes_path, ids_col, sep, score_col)
    in_set = pairs["a"].isin(node_set) & pairs["b"].isin(node_set)
    pairs = pairs[in_set].reset_index(drop=True)
    print(f"[{label}] node set size={len(node_set)}, scored pairs in set={len(pairs)}")

    references = {
        "direct": cfg["data"]["direct_reference"],
        "cocomplex": cfg["data"]["cocomplex_reference"],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_score = pairs["score"].to_numpy()
    pair_keys = list(zip(pairs["a"], pairs["b"]))

    summary_rows = []
    fig_roc, ax_roc = plt.subplots()
    fig_prc, ax_prc = plt.subplots()

    for ref_name, ref_path in references.items():
        ref_pairs = _load_reference_pairs(ref_path)
        y_true = pd.Series([1 if k in ref_pairs else 0 for k in pair_keys]).to_numpy()
        n_pos = int(y_true.sum())
        print(
            f"[{label}] {ref_name}: {n_pos} positive pairs out of {len(y_true)} scored pairs"
        )
        if n_pos == 0 or n_pos == len(y_true):
            print(
                f"[{label}] {ref_name}: skipping (need both positive and negative pairs)"
            )
            continue

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        ax_roc.plot(fpr, tpr, label=f"{ref_name} (AUC={roc_auc:.3f})")

        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_score)
        ap = sklearn.metrics.average_precision_score(y_true, y_score)
        ax_prc.plot(recall, precision, label=f"{ref_name} (AP={ap:.3f})")

        summary_rows.append(
            {
                "reference": ref_name,
                "n_scored_pairs": len(y_true),
                "n_positive_pairs": n_pos,
                "roc_auc": roc_auc,
                "average_precision": ap,
            }
        )

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(
        f"ROC: Predictomes spoc_score vs PDB references ({label} node set)"
    )
    ax_roc.legend()
    fig_roc.savefig(out_dir / f"predictomes_roc_{label}.png", dpi=dpi)
    plt.close(fig_roc)

    ax_prc.set_xlabel("Recall")
    ax_prc.set_ylabel("Precision")
    ax_prc.set_title(
        f"PRC: Predictomes spoc_score vs PDB references ({label} node set)"
    )
    ax_prc.legend()
    fig_prc.savefig(out_dir / f"predictomes_prc_{label}.png", dpi=dpi)
    plt.close(fig_prc)

    pd.DataFrame(summary_rows).to_csv(
        out_dir / f"predictomes_roc_prc_{label}_summary.tsv", sep="\t", index=False
    )
    print(f"[{label}] wrote ROC/PRC figures and summary to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python predictomes_roc_prc.py <config.yaml> <predictomes_csv> "
            "<prey_list> <prey_col> <prey_has_header:true|false> <label> <output_dir>"
        )
        sys.exit(1)
    main(*sys.argv[1:])

"""
Compute overlap of Predictomes unordered UniProt pairs with a cullin node set.

Predictomes stores each pair as a colon-separated string (e.g. `P31946:P62258`)
in the column named in config["predictomes"]["uniprot_ids_col"]. A pair is
considered "covered" by the node set if BOTH endpoints are members of the set,
regardless of whether our data has any AP-MS signal at that pair. Pairs are
unordered, so (a, b) and (b, a) are collapsed by sorting before comparison.

Data: Schmidt et al. (2025). Mol Cell. https://doi.org/10.1016/j.molcel.2025.01.034
https://predictomes.org
"""

import sys
from pathlib import Path

import pandas as pd
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

    prey_has_header = prey_has_header_str.lower() == "true"
    node_set = _load_node_set(prey_list_path, prey_col, prey_has_header)
    n_nodes = len(node_set)
    n_our_pairs = n_nodes * (n_nodes - 1) // 2

    df = pd.read_csv(predictomes_path, usecols=[ids_col])
    parts = df[ids_col].astype(str).str.split(sep, n=1, expand=True)
    parts.columns = ["a_raw", "b_raw"]
    parts = parts.dropna()
    parts["a_raw"] = parts["a_raw"].str.strip()
    parts["b_raw"] = parts["b_raw"].str.strip()
    parts = parts[parts["a_raw"] != parts["b_raw"]]

    a = parts["a_raw"].to_numpy()
    b = parts["b_raw"].to_numpy()
    pair_a = pd.Series([min(x, y) for x, y in zip(a, b)], name="a")
    pair_b = pd.Series([max(x, y) for x, y in zip(a, b)], name="b")
    norm = pd.concat([pair_a, pair_b], axis=1).drop_duplicates().reset_index(drop=True)

    norm["covered"] = norm["a"].isin(node_set) & norm["b"].isin(node_set)
    n_predictomes_total = len(norm)
    n_covered = int(norm["covered"].sum())
    frac_predictomes_covered = (
        n_covered / n_predictomes_total if n_predictomes_total else 0.0
    )
    frac_our_with_predictomes = n_covered / n_our_pairs if n_our_pairs else 0.0

    summary = "\n".join(
        [
            f"Cullin node set:                                       {label}",
            f"N nodes in set:                                        {n_nodes}",
            f"N unordered pairs in set (N*(N-1)/2):                  {n_our_pairs}",
            f"N Predictomes unordered pairs (deduped):               {n_predictomes_total}",
            f"N Predictomes pairs with both endpoints in node set:   {n_covered}",
            f"Fraction of Predictomes pairs covered:                 {frac_predictomes_covered:.6f}",
            f"Fraction of our pairs with a Predictomes prediction:   {frac_our_with_predictomes:.6f}",
        ]
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"predictomes_overlap_{label}_summary.txt").write_text(summary + "\n")
    print(summary)

    norm[norm["covered"]].drop(columns=["covered"]).to_csv(
        out_dir / f"predictomes_overlap_{label}_covered_pairs.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python predictomes_overlap.py <config.yaml> <predictomes_csv> "
            "<prey_list> <prey_col> <prey_has_header:true|false> <label> <output_dir>"
        )
        sys.exit(1)
    main(*sys.argv[1:])

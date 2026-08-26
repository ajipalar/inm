"""
Compute coverage of Burke et al. 2023 predicted PPIs over our network's protein set.

Edges in Burke are unordered pairs of UniProt accessions. For each Burke edge we
ask: are *both* endpoints present as nodes in our network? (We do not care
whether the corresponding entry in our data has any AP-MS signal — the question
is purely set membership.)
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_burke(burke_path: Path, burke_cfg: dict) -> pd.DataFrame:
    """Read the Burke supplementary table (xlsx/tsv/csv) into a normalized DataFrame."""
    suffix = burke_path.suffix.lower()
    sheet = burke_cfg.get("sheet", 0)
    skiprows = burke_cfg.get("skiprows", 0)
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(burke_path, sheet_name=sheet, skiprows=skiprows)
    elif suffix == ".tsv":
        df = pd.read_csv(burke_path, sep="\t", skiprows=skiprows)
    else:
        df = pd.read_csv(burke_path, skiprows=skiprows)
    return df


def _normalize_pairs(
    df: pd.DataFrame, a_col: str, b_col: str, score_col: str | None
) -> pd.DataFrame:
    """Sort each pair so (a,b) and (b,a) collapse to one row; drop duplicates."""
    keep = [a_col, b_col] + ([score_col] if score_col else [])
    out = df[keep].copy()
    out.columns = ["a_raw", "b_raw"] + (["score"] if score_col else [])
    out = out.dropna(subset=["a_raw", "b_raw"])
    out["a_raw"] = out["a_raw"].astype(str).str.strip()
    out["b_raw"] = out["b_raw"].astype(str).str.strip()
    out = out[out["a_raw"] != out["b_raw"]]
    pair = out.apply(lambda r: tuple(sorted((r["a_raw"], r["b_raw"]))), axis=1)
    out["a"] = pair.str[0]
    out["b"] = pair.str[1]
    out = out.drop_duplicates(subset=["a", "b"])
    cols = ["a", "b"] + (["score"] if score_col else [])
    return out[cols].reset_index(drop=True)


def main(
    config_path: str, burke_path: str, prey_names_path: str, output_dir: str
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    burke_cfg = cfg["burke"]
    a_col = burke_cfg["protein_a_col"]
    b_col = burke_cfg["protein_b_col"]
    score_col = burke_cfg.get("score_col") or None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prey = pd.read_csv(prey_names_path, sep="\t")
    our_nodes = set(prey["UniProt"].astype(str).str.strip())
    n_nodes = len(our_nodes)
    n_our_pairs = n_nodes * (n_nodes - 1) // 2

    burke_raw = _load_burke(Path(burke_path), burke_cfg)
    missing = [c for c in (a_col, b_col) if c not in burke_raw.columns]
    if missing:
        raise KeyError(
            f"Burke table missing expected columns {missing}. "
            f"Found: {list(burke_raw.columns)}. "
            "Update burke.protein_a_col / burke.protein_b_col in config.yaml."
        )

    burke = _normalize_pairs(burke_raw, a_col, b_col, score_col)
    burke["covered"] = burke["a"].isin(our_nodes) & burke["b"].isin(our_nodes)

    n_burke_total = len(burke)
    n_burke_covered = int(burke["covered"].sum())
    frac_burke_covered = n_burke_covered / n_burke_total if n_burke_total else 0.0
    frac_our_with_burke = n_burke_covered / n_our_pairs if n_our_pairs else 0.0

    summary = "\n".join(
        [
            f"N nodes in our network:                              {n_nodes}",
            f"N unordered pairs in our network (N*(N-1)/2):        {n_our_pairs}",
            f"N Burke unordered pairs (deduped):                   {n_burke_total}",
            f"N Burke pairs with both endpoints in our network:    {n_burke_covered}",
            f"Fraction of Burke pairs covered by our nodes:        {frac_burke_covered:.4f}",
            f"Fraction of our pairs that have a Burke prediction:  {frac_our_with_burke:.4f}",
        ]
    )

    (out_dir / "burke_coverage_summary.txt").write_text(summary + "\n")
    print(summary)

    burke.to_csv(out_dir / "burke_coverage_report.tsv", sep="\t", index=False)
    burke[burke["covered"]].drop(columns=["covered"]).to_csv(
        out_dir / "burke_covered_pairs.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 5:
        print(
            "Usage: python burke_coverage.py <config.yaml> <burke_raw> <prey_names> <output_dir>"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

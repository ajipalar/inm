"""Filter prey by SAINT score cutoff and write model-ready tables to output_dir."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def main(config_path: str, output_dir: str) -> None:
    config_path = Path(config_path)
    output_dir = Path(output_dir)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    saint_cutoff = cfg.get("filter", {}).get("saint_cutoff", 0.5)
    spec_table_path = cfg["data"]["formatted_spec_table"]
    composite_table_path = cfg["data"]["formatted_composite_table"]
    prey_names_path = cfg["data"]["prey_names"]

    spec_table = pd.read_csv(spec_table_path, sep="\t", index_col=0, header=None)
    composite_table = pd.read_csv(composite_table_path, sep="\t")
    prey_names = pd.read_csv(prey_names_path, sep="\t")

    # Prey that pass SAINT cutoff
    max_scores = composite_table.groupby("Prey")["MSscore"].max()
    kept_prey = set(max_scores[max_scores >= saint_cutoff].index)

    # Baits are stored as 4-char gene name prefixes (e.g. "CUL5").
    # Resolve them to UniProt IDs via prey_names and always include them,
    # regardless of their SAINT score as prey.
    gene_prefix_to_uniprot = dict(
        zip(prey_names["GeneName"].str.split("_").str[0], prey_names["UniProt"])
    )
    bait_uniprots = {
        gene_prefix_to_uniprot[b]
        for b in composite_table["Bait"].unique()
        if b in gene_prefix_to_uniprot
    }
    kept_prey |= bait_uniprots
    logger.info(
        "Always keeping %d bait UniProt IDs: %s",
        len(bait_uniprots),
        sorted(bait_uniprots),
    )

    filtered_composite = composite_table[composite_table["Prey"].isin(kept_prey)]
    filtered_spec = spec_table[spec_table.index.isin(kept_prey)]

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_spec.to_csv(output_dir / "spec_table.tsv", sep="\t", header=False)
    filtered_composite.to_csv(output_dir / "composite_table.tsv", sep="\t", index=False)

    logger.info(
        "Filtered %d → %d prey (SAINT >= %s + %d forced baits); wrote to %s",
        len(spec_table),
        len(filtered_spec),
        saint_cutoff,
        len(bait_uniprots),
        output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Filter prey by SAINT cutoff.")
    parser.add_argument("config_path", help="Path to config.yaml")
    parser.add_argument("output_dir", help="Directory to write filtered tables")
    args = parser.parse_args()
    main(args.config_path, args.output_dir)

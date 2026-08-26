"""Compute Pearson correlations on the filtered spectral count table."""

import argparse
import logging

import yaml

from inm.preprocess.compute_correlations import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Compute pairwise correlations.")
    parser.add_argument("config_path", help="Path to config.yaml")
    parser.add_argument(
        "data_dir",
        help="Directory containing spec_table.tsv; correlations written here",
    )
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)

    run(cfg, args.data_dir)

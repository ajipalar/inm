"""
Download the Predictomes pair-score dump (Schmidt et al., Mol Cell 2025)
used by the predictomes_* analysis scripts.

Paper: https://doi.org/10.1016/j.molcel.2025.01.034
Database: https://predictomes.org
"""

import logging
import sys
from pathlib import Path

import yaml

from inm.preprocess import fetch_predictome

logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    """Download the Predictomes CSV to the path config["predictomes"]["raw_path"] resolves to."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    fetch_predictome.run(cfg, Path(config_path).parent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python fetch_predictome.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])

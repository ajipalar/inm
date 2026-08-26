"""
Load cullin AP-MS data and format using inm.preprocess module.
Outputs: spectral count table and composite table for model stage.
"""

import logging
import sys

import yaml

from inm.preprocess import build_spec_table

logger = logging.getLogger(__name__)


def main(config_path: str, output_dir: str) -> None:
    """Load cullin xlsx and build spec_table.tsv and composite_table.tsv."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    build_spec_table.run(cfg, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <config.yaml> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

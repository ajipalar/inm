"""
Download the cullin AP-MS xlsx (Wan et al. 2019) used as the prey universe
for the PDB PPI benchmark pipeline.

The publisher blocks automated downloads (Cloudflare); if fetch() raises
FileNotFoundError, it prints instructions for placing the file manually.
"""

import logging
import sys

import yaml

from inm.preprocess import fetch_cullin_apms

logger = logging.getLogger(__name__)


def main(config_path: str, output_dir: str) -> None:
    """Download the cullin AP-MS xlsx into output_dir, verifying its checksum."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    fetch_cullin_apms.run(cfg, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print("Usage: python fetch_cullin.py <config.yaml> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

"""Derive PDB PPI benchmark edge lists from the prey-prey BSASA reference.

Two benchmarks are produced from ``BSASA_reference.csv``:

- direct: prey pairs whose own chain-pair contact has BSASA >= the cutoff.
- cocomplex: prey pairs that co-occur in a PDB structure containing at least
  one chain-pair contact >= the cutoff (includes prey pairs whose own chains
  did not directly contact, but that were resolved in the same validated
  complex).
"""

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def build_direct_benchmark(ref: pd.DataFrame, min_bsasa: float) -> pd.DataFrame:
    """Unique prey pairs with a chain-pair contact BSASA >= min_bsasa."""
    hits = ref[ref["bsasa_lst"] >= min_bsasa]
    pairs = {
        tuple(sorted((p1, p2)))
        for p1, p2 in zip(hits["Prey1"], hits["Prey2"])
        if p1 != p2
    }
    return pd.DataFrame(sorted(pairs), columns=["auid", "buid"])


def build_cocomplex_benchmark(ref: pd.DataFrame, min_bsasa: float) -> pd.DataFrame:
    """Unique prey pairs co-present in a PDB with >=1 contact >= min_bsasa."""
    pairs = set()
    for pdb_id, group in ref.groupby("PDBID"):
        if not (group["bsasa_lst"] >= min_bsasa).any():
            continue
        preys = sorted(set(group["Prey1"]) | set(group["Prey2"]))
        pairs.update(combinations(preys, 2))
    return pd.DataFrame(sorted(pairs), columns=["auid", "buid"])


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Build direct and cocomplex benchmark TSVs from BSASA_reference.csv."""
    output_dir = Path(output_dir)
    cif_dir = output_dir / "significant_cifs"
    min_bsasa = cfg["filter"]["min_bsasa"]

    ref = pd.read_csv(cif_dir / "BSASA_reference.csv")
    n_before = len(ref)
    ref = ref.dropna(subset=["Prey1", "Prey2"])
    if len(ref) < n_before:
        logger.warning(
            "Dropped %d rows with unmapped (NaN) prey IDs", n_before - len(ref)
        )
    logger.info("Read BSASA_reference.csv: %d rows", len(ref))

    direct = build_direct_benchmark(ref, min_bsasa)
    direct_path = output_dir / "direct_benchmark.tsv"
    direct.to_csv(direct_path, sep="\t", header=False, index=False)
    logger.info("Wrote %d direct benchmark pairs to %s", len(direct), direct_path)

    cocomplex = build_cocomplex_benchmark(ref, min_bsasa)
    cocomplex_path = output_dir / "cocomplex_benchmark.tsv"
    cocomplex.to_csv(cocomplex_path, sep="\t", header=False, index=False)
    logger.info(
        "Wrote %d cocomplex benchmark pairs to %s", len(cocomplex), cocomplex_path
    )

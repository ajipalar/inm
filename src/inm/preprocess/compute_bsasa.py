"""Compute Buried Solvent Accessible Surface Area (BSASA) for structure chain pairs."""

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_bsasa_value(sasa1: float, sasa2: float, sasa12: float) -> float:
    """Return BSASA = SASA1 + SASA2 - SASA_complex."""
    return sasa1 + sasa2 - sasa12


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Compute BSASA for all chain pairs in each .bio.mmtf structure file."""
    import biotite.structure
    import biotite.structure.io.mmtf as mmtf

    output_dir = Path(output_dir)
    cif_dir = output_dir / "significant_cifs"

    bio_mmtfs = sorted(cif_dir.glob("*.bio.mmtf"))
    if not bio_mmtfs:
        logger.warning("No .bio.mmtf files found in %s", cif_dir)
        return

    logger.info("Computing BSASA for %d structure files", len(bio_mmtfs))

    row_lst = []
    for mmtf_file_path in bio_mmtfs:
        pdb_id = mmtf_file_path.name[:4]
        logger.info("Reading %s", mmtf_file_path.name)

        try:
            mmtf_file = mmtf.MMTFFile.read(str(mmtf_file_path))
            atom_array = mmtf.get_structure(mmtf_file, model=1)

            mask = biotite.structure.filter_canonical_amino_acids(atom_array) & (
                atom_array.element != "H"
            )

            chain_set = set(atom_array[mask].chain_id)
            chain_pairs = list(combinations(sorted(chain_set), 2))

            # Compute per-chain SASA
            monomer_sasa = {}
            for chain in chain_set:
                chain_mask = mask & (atom_array.chain_id == chain)
                sasa_array = biotite.structure.sasa(atom_array[chain_mask])
                if np.any(np.isnan(sasa_array)):
                    logger.warning("NaN in SASA for chain %s of %s", chain, pdb_id)
                monomer_sasa[chain] = float(np.sum(sasa_array))

            # Compute pairwise BSASA
            for chain1, chain2 in chain_pairs:
                pair_mask = mask & (
                    (atom_array.chain_id == chain1) | (atom_array.chain_id == chain2)
                )
                sasa_array = biotite.structure.sasa(atom_array[pair_mask])
                if np.any(np.isnan(sasa_array)):
                    logger.warning(
                        "NaN in SASA for %s-%s of %s", chain1, chain2, pdb_id
                    )
                sasa12 = float(np.sum(sasa_array))
                sasa1 = monomer_sasa[chain1]
                sasa2 = monomer_sasa[chain2]
                bsasa = compute_bsasa_value(sasa1, sasa2, sasa12)
                row_lst.append((pdb_id, chain1, chain2, bsasa, sasa12, sasa1, sasa2))

        except Exception:
            logger.exception("Error processing %s, skipping", pdb_id)
            continue

    columns = ["PDBID", "Chain1", "Chain2", "BSASA", "SASA12", "SASA1", "SASA2"]
    df = pd.DataFrame(row_lst, columns=columns)

    out_path = cif_dir / "bsasa.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote BSASA results (%d rows) to %s", len(df), out_path)

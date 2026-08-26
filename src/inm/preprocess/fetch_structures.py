"""Fetch PDB structure files and convert CIF to bio assembly MMTF."""

import logging
from pathlib import Path
from typing import Any, Dict, Set

import pandas as pd

logger = logging.getLogger(__name__)


def remap_pdb_id(pdb_id: str, remapping: Dict[str, str]) -> str:
    """Remap obsolete PDB IDs to current ones."""
    pdb_lower = pdb_id.lower()
    return remapping.get(pdb_lower, pdb_lower)


def extract_pdb_ids(sig_hits_path: Path, remapping: Dict[str, str]) -> Set[str]:
    """Extract unique PDB IDs from significant hits, applying remapping."""
    df = pd.read_csv(sig_hits_path)
    pdb_ids = set()
    for pdb70id in df["PDB70ID"].values:
        pdb_id = str(pdb70id).split("_")[0].lower()
        pdb_id = remap_pdb_id(pdb_id, remapping)
        pdb_ids.add(pdb_id)
    return pdb_ids


def convert_cif_to_bio_mmtf(cif_path: Path, output_path: Path) -> None:
    """Convert a CIF file to biological assembly MMTF."""
    import biotite.structure.io.mmtf as mmtf
    import biotite.structure.io.pdbx as pdbx

    logger.info("Converting %s -> %s", cif_path.name, output_path.name)
    cif_file = pdbx.PDBxFile.read(str(cif_path))
    assembly = pdbx.get_assembly(cif_file, model=1)
    mmtf_file = mmtf.MMTFFile()
    mmtf.set_structure(mmtf_file, assembly)
    mmtf_file.write(str(output_path))


def run(cfg: Dict[str, Any], output_dir: Any) -> None:
    """Fetch structures and convert CIF to bio.mmtf."""
    output_dir = Path(output_dir)
    remapping = cfg.get("obsolete_pdb_remapping", {})
    mirror_path = cfg.get("pdb_mirror_path")

    sig_path = output_dir / "hhblits_out" / "SignificantPreyPDB70PairAlign.csv"
    if not sig_path.exists():
        logger.warning("No significant hits file found at %s", sig_path)
        return

    struct_dir = output_dir / "significant_cifs"
    struct_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = extract_pdb_ids(sig_path, remapping)
    logger.info("Found %d unique PDB IDs", len(pdb_ids))

    for pdb_id in sorted(pdb_ids):
        mmtf_path = struct_dir / f"{pdb_id}.bio.mmtf"
        if mmtf_path.exists():
            logger.debug("Skipping %s — bio.mmtf already exists", pdb_id)
            continue

        # Try to find and convert CIF
        cif_path = struct_dir / f"{pdb_id}.cif"
        if not cif_path.exists() and mirror_path:
            # Fetch from local PDB mirror
            middle = pdb_id[1:3]
            mirror_cif = Path(mirror_path) / "mmCIF" / middle / f"{pdb_id}.cif.gz"
            if mirror_cif.exists():
                import gzip
                import shutil

                logger.info("Copying %s from mirror", pdb_id)
                with gzip.open(mirror_cif, "rb") as f_in:
                    with open(cif_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

        if cif_path.exists():
            try:
                convert_cif_to_bio_mmtf(cif_path, mmtf_path)
            except Exception as e:
                logger.error("Failed to convert %s: %s", pdb_id, e)
        else:
            logger.warning("No CIF file found for %s", pdb_id)

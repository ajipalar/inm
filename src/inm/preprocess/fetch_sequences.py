"""Fetch FASTA sequences from UniProt for each prey protein."""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Fetch FASTA files for all proteins in prey_table.csv."""
    output_dir = Path(output_dir)
    prey_table = pd.read_csv(output_dir / "prey_table.csv")
    seq_dir = output_dir / "input_sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    uids = prey_table["UniprotId"].values
    missing = [uid for uid in uids if not (seq_dir / f"{uid}.fasta").exists()]

    if not missing:
        logger.info("All %d FASTA files already exist, skipping fetch", len(uids))
        return

    logger.info("Fetching %d missing FASTA files from UniProt", len(missing))
    import biotite.database.uniprot

    for uid in missing:
        fasta_path = seq_dir / f"{uid}.fasta"
        if fasta_path.exists():
            continue
        try:
            logger.debug("Fetching %s", uid)
            biotite.database.uniprot.fetch(uid, "fasta", str(seq_dir))
        except TimeoutError:
            logger.warning("Timeout fetching %s, will retry on next run", uid)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", uid, e)

"""Run HHblits sequence search against PDB70 database."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def build_hhblits_command(
    fasta_path: Path,
    output_path: Path,
    db_path: str,
    cpu: int = 2,
) -> List[str]:
    """Build the hhblits command as a list of arguments."""
    return [
        "hhblits",
        "-i",
        str(fasta_path),
        "-o",
        str(output_path),
        "-d",
        str(db_path),
        "-cpu",
        str(cpu),
    ]


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Run HHblits for each protein, skipping those with existing output."""
    output_dir = Path(output_dir)
    prey_table = pd.read_csv(output_dir / "prey_table.csv")

    hhr_dir = output_dir / "hhblits_out"
    hhr_dir.mkdir(parents=True, exist_ok=True)

    db_path = cfg["hhblits"]["db_path"]
    cpu = cfg["hhblits"].get("cpu", 2)

    uids = prey_table["UniprotId"].values
    for uid in uids:
        fasta_path = output_dir / "input_sequences" / f"{uid}.fasta"
        output_path = hhr_dir / f"{uid}.hhr"

        if output_path.exists():
            logger.debug("Skipping %s — .hhr already exists", uid)
            continue

        cmd = build_hhblits_command(fasta_path, output_path, db_path, cpu)
        logger.info("Running HHblits for %s", uid)
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            logger.error("hhblits not found. Is HH-suite installed and on PATH?")
            raise
        except subprocess.CalledProcessError as e:
            logger.error("HHblits failed for %s: %s", uid, e)

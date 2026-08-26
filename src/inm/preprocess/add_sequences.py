"""Read FASTA files and add sequence columns to prey table."""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import biotite.sequence.io.fasta

logger = logging.getLogger(__name__)


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Add UniprotSeq and UniprotSeqLen columns to prey_table.csv."""
    output_dir = Path(output_dir)
    prey_table_path = output_dir / "prey_table.csv"
    seq_dir = output_dir / "input_sequences"

    df = pd.read_csv(prey_table_path)

    sequences = []
    for uid in df["UniprotId"].values:
        fasta_path = seq_dir / f"{uid}.fasta"
        fasta_file = biotite.sequence.io.fasta.FastaFile.read(str(fasta_path))
        # Get the first (and typically only) sequence
        header, seq = list(fasta_file.items())[0]
        sequences.append(seq)

    df["UniprotSeq"] = sequences
    df["UniprotSeqLen"] = [len(s) for s in sequences]
    df.to_csv(prey_table_path, index=False)
    logger.info("Added sequences for %d proteins", len(sequences))

"""Filter HHblits hits by significance thresholds."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Filter PreyPDB70PairAlign.csv by evalue, aligned cols, and identity."""
    output_dir = Path(output_dir)
    hhr_dir = output_dir / "hhblits_out"

    df = pd.read_csv(hhr_dir / "PreyPDB70PairAlign.csv")

    # Convert columns to numeric
    df["Aligned_cols"] = np.array(
        [int(i) for i in df["Aligned_cols"].values], dtype=int
    )
    df["Identities"] = np.array(
        [int(str(i).strip("%")) for i in df["Identities"].values], dtype=int
    )

    filters = cfg["filter"]
    max_evalue = filters["max_evalue"]
    min_aligned_cols = filters["min_aligned_cols"]
    min_identity_pct = filters["min_identity_pct"]

    before = len(df)
    df = df[df["Evalue"] <= max_evalue]
    df = df[df["Aligned_cols"] >= min_aligned_cols]
    df = df[df["Identities"] >= min_identity_pct]
    after = len(df)

    logger.info(
        "Filtered hits: %d -> %d (evalue<=%.1e, cols>=%d, id>=%d%%)",
        before,
        after,
        max_evalue,
        min_aligned_cols,
        min_identity_pct,
    )

    out_path = hhr_dir / "SignificantPreyPDB70PairAlign.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote significant hits to %s", out_path)

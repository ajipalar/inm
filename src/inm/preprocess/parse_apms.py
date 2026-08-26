"""Parse AP-MS Excel data into a prey table."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Parse AP-MS Excel file and write prey_table.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = cfg["xlsx_path"]
    sheet_nums: List[int] = cfg["sheet_nums"]
    prey_gene_col: str = cfg.get("prey_gene_col", "PreyGene")
    prey_col: str = cfg.get("prey_col", "Prey")
    viral_remapping: Dict[str, str] = cfg.get("viral_remapping", {})
    exclude_genes: List[str] = cfg.get("exclude_prey_genes", [])

    frames = []
    for sheet_num in sheet_nums:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_num)
        # Drop excluded genes
        for gene in exclude_genes:
            df = df[df[prey_gene_col] != gene]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Build deduplicated prey table
    prey_gene_list: List[str] = []
    prey_id_list: List[str] = []
    for _, row in combined.iterrows():
        pg = row[prey_gene_col]
        p = row[prey_col]
        # Apply viral remapping
        if pg in viral_remapping:
            p = viral_remapping[pg]
        if pg not in prey_gene_list:
            prey_gene_list.append(pg)
            prey_id_list.append(p)

    prey_table = pd.DataFrame(
        {
            "PreyGene": prey_gene_list,
            "UniprotId": prey_id_list,
        }
    )

    out_path = output_dir / "prey_table.csv"
    prey_table.to_csv(out_path, index=False)
    logger.info("Wrote prey table with %d entries to %s", len(prey_table), out_path)

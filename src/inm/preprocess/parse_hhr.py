"""Parse HHblits .hhr output files into structured alignment data."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def parse_meta(meta: str) -> Dict[str, str]:
    """Parse the meta line of an HHR alignment block into a dict."""
    parts = meta.split()
    return {p.split("=")[0]: p.split("=")[1] for p in parts if "=" in p}


def parse_hhr_file(filename) -> List[Tuple[str, str, str, str]]:
    """Parse an HHR file and return list of (header, meta, query, template) tuples."""
    with open(filename, "r") as f:
        lines = f.readlines()

    alignments = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("No ") and not line.startswith("No Hit"):
            header = lines[i + 1].strip()
            meta = lines[i + 2].strip()
            i += 3

            query = ""
            template = ""

            for j in range(i, len(lines)):
                if lines[j].strip() == "":
                    # Check if this is the end of the alignment block
                    # (empty line followed by another "No " or end of file)
                    if (
                        j + 1 >= len(lines)
                        or lines[j + 1].strip().startswith("No ")
                        or lines[j + 1].strip() == ""
                    ):
                        break
                    continue

                parts = lines[j].strip().split()
                if not parts:
                    continue

                if parts[0] == "Q" and parts[1] != "Consensus":
                    query += parts[3]
                elif (
                    parts[0] == "T"
                    and parts[1] != "Consensus"
                    and parts[1] != "ss_dssp"
                    and parts[1] != "ss_pred"
                ):
                    template += parts[3]

            alignments.append((header, meta, query, template))
        i += 1

    return alignments


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Parse all .hhr files, write hhr_alns.json and PreyPDB70PairAlign.csv."""
    output_dir = Path(output_dir)
    hhr_dir = output_dir / "hhblits_out"

    hhr_files = sorted(hhr_dir.glob("*.hhr"))
    if not hhr_files:
        logger.warning("No .hhr files found in %s", hhr_dir)
        return

    hhr_alns = {}
    rows = []
    for hhr_file in hhr_files:
        logger.debug("Parsing %s", hhr_file.name)
        query_uid = hhr_file.stem
        alns = parse_hhr_file(hhr_file)
        hhr_alns[hhr_file.name] = alns
        for header, meta, query, template in alns:
            meta_dict = parse_meta(meta)
            pdb70_id = header.split()[0].lstrip(">")
            rows.append(
                {
                    "QueryUID": query_uid,
                    "PDB70ID": pdb70_id,
                    "Probab": meta_dict.get("Probab"),
                    "Evalue": meta_dict.get("E-value"),
                    "Score": meta_dict.get("Score"),
                    "Aligned_cols": meta_dict.get("Aligned_cols"),
                    "Identities": meta_dict.get("Identities"),
                    "Similarity": meta_dict.get("Similarity"),
                    "Sum_probs": meta_dict.get("Sum_probs"),
                    "Template_Neff": meta_dict.get("Template_Neff"),
                    "Q": query,
                    "T": template,
                }
            )

    out_path = hhr_dir / "hhr_alns.json"
    with open(out_path, "w") as f:
        json.dump(hhr_alns, f)
    logger.info("Wrote %d parsed HHR files to %s", len(hhr_alns), out_path)

    csv_path = hhr_dir / "PreyPDB70PairAlign.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Wrote %d alignment rows to %s", len(rows), csv_path)

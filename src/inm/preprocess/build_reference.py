"""Build prey-prey reference table from chain mapping and BSASA data."""

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

logger = logging.getLogger(__name__)


def _get_prey_ids(chain_subframe: pd.DataFrame, chain_id: str) -> Set[str]:
    """Return set of prey (QueryID) values mapped to a given chain."""
    sel = chain_subframe["ChainID"] == chain_id
    prey_set = set(chain_subframe.loc[sel, "QueryID"].values)
    prey_set.discard("")
    return prey_set


def run(cfg: Dict[str, Any], output_dir) -> None:
    """Build prey-prey BSASA reference from chain mapping and BSASA data."""
    output_dir = Path(output_dir)
    cif_dir = output_dir / "significant_cifs"

    min_bsasa = cfg["filter"]["min_bsasa"]
    min_seq_id = cfg["filter"]["min_seq_id"]

    # 1. Read inputs
    chain_mapping = pd.read_csv(cif_dir / "chain_mapping_all.csv")
    bsasa = pd.read_csv(cif_dir / "bsasa.csv")

    logger.info(
        "Read chain_mapping: %d rows, %d unique QueryIDs, %d unique PDBIDs",
        len(chain_mapping),
        chain_mapping["QueryID"].nunique(),
        chain_mapping["PDBID"].nunique(),
    )
    logger.info("Read BSASA: %d rows", len(bsasa))

    # 2. Defensive filter: drop non-numeric BSASA values (e.g. "KeyError")
    bsasa["BSASA"] = pd.to_numeric(bsasa["BSASA"], errors="coerce")
    n_before = len(bsasa)
    bsasa = bsasa.dropna(subset=["BSASA"])
    n_dropped = n_before - len(bsasa)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with non-numeric BSASA", n_dropped)

    # 3. Filter BSASA by threshold
    bsasa = bsasa[bsasa["BSASA"] >= min_bsasa]
    logger.info("BSASA rows with >= %.0f: %d", min_bsasa, len(bsasa))

    # 4. Filter chain_mapping by sequence identity threshold
    chain_mapping["bt_aln_percent_seq_id"] = pd.to_numeric(
        chain_mapping["bt_aln_percent_seq_id"], errors="coerce"
    )
    chain_mapping = chain_mapping[chain_mapping["bt_aln_percent_seq_id"] >= min_seq_id]
    logger.info(
        "Chain mapping rows with seq_id >= %.2f: %d", min_seq_id, len(chain_mapping)
    )

    # 5. Build prey-prey reference
    rows: List[dict] = []
    bsasa_pdb_set = sorted(set(bsasa["PDBID"].values))
    logger.info("Processing %d PDBs", len(bsasa_pdb_set))

    for pdb_idx, pdb_id in enumerate(bsasa_pdb_set):
        if pdb_idx % 1000 == 0 and pdb_idx > 0:
            logger.info("Progress: %d / %d PDBs", pdb_idx, len(bsasa_pdb_set))

        chain_sub = chain_mapping[chain_mapping["PDBID"] == pdb_id]
        bsasa_sub = bsasa[bsasa["PDBID"] == pdb_id]

        # Collect all chains that appear in BSASA data for this PDB
        bsasa_chain_set = set(bsasa_sub["Chain1"].values) | set(
            bsasa_sub["Chain2"].values
        )

        # Map each chain to its prey IDs
        chain2prey = {
            chain: _get_prey_ids(chain_sub, chain) for chain in bsasa_chain_set
        }

        # Build chain-pair -> BSASA lookup
        chainpair2bsasa: Dict[frozenset, float] = {}
        for _, r in bsasa_sub.iterrows():
            key = frozenset((r["Chain1"], r["Chain2"]))
            chainpair2bsasa[key] = r["BSASA"]

        # For each chain pair with BSASA, cross-product prey lists
        for chain1, chain2 in combinations(sorted(bsasa_chain_set), 2):
            chain_key = frozenset((chain1, chain2))
            if chain_key not in chainpair2bsasa:
                continue

            prey_set1 = chain2prey[chain1]
            prey_set2 = chain2prey[chain2]

            if not prey_set1 or not prey_set2:
                continue

            bsasa_val = chainpair2bsasa[chain_key]
            for prey1 in prey_set1:
                for prey2 in prey_set2:
                    rows.append(
                        {
                            "Prey1": prey1,
                            "Prey2": prey2,
                            "PDBID": pdb_id,
                            "Chain1": chain1,
                            "Chain2": chain2,
                            "bsasa_lst": bsasa_val,
                        }
                    )

    df = pd.DataFrame(
        rows, columns=["Prey1", "Prey2", "PDBID", "Chain1", "Chain2", "bsasa_lst"]
    )

    out_path = cif_dir / "BSASA_reference.csv"
    df.to_csv(out_path, index=False)

    n_prey = set(df["Prey1"].values) | set(df["Prey2"].values) if len(df) > 0 else set()
    logger.info(
        "Wrote BSASA reference (%d rows, %d unique prey, %d unique PDBs) to %s",
        len(df),
        len(n_prey),
        df["PDBID"].nunique() if len(df) > 0 else 0,
        out_path,
    )

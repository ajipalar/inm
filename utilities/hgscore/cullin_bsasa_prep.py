"""Build the CRL5 AP-MS bait/prey dataframe from the supplementary xlsx source."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_BAIT2GENE = {
    "CBFB": "PEBB_HUMAN",
    "ELOB": "ELOB_HUMAN",
    "CUL5": "CUL5_HUMAN",
    "LRR1": "LLR1_HUMAN",
}
VIRAL_PROTEIN_REMAPPING = {
    "vifprotein": "P69723",
    "polpolyprotein": "Q2A7R5",
    "nefprotein": "P18801",
    "tatprotein": "P0C1K3",
    "gagpolyprotein": "P12493",
    "revprotein": "P69718",
    "envpolyprotein": "O12164",
}


def init_dfs(xlsx_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the 3 AP-MS sheets from the supplementary xlsx, dropping the mouse IgG control prey."""
    df1 = pd.read_excel(xlsx_path, 0)
    df2 = pd.read_excel(xlsx_path, 1)
    df3 = pd.read_excel(xlsx_path, 2)
    df1 = df1[df1["PreyGene"] != "IGHG1_MOUSE"]
    df2 = df2[df2["PreyGene"] != "IGHG1_MOUSE"]
    df3 = df3[df3["PreyGene"] != "IGHG1_MOUSE"]
    assert "IGHG1_MOUSE" not in df1.values
    assert "IGHG1_MOUSE" not in df2.values
    assert "IGHG1_MOUSE" not in df3.values
    return df1, df2, df3


def init_bait2uid(
    conditions: List[str],
    gene2uid: Dict[str, str],
    baits: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a {bait+condition: uniprot_id} lookup for the bait protein itself."""
    if baits is None:
        baits = DEFAULT_BAIT2GENE
    bait2uid = {}
    for bait, genename in baits.items():
        for condition in conditions:
            bait2uid[bait + condition] = gene2uid[genename]
    return bait2uid


def init_df_all(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    gene2uid: Dict[str, str],
    conditions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Concatenate the 3 AP-MS sheets, tagging each row with its bait's UniProt ID and remapping viral prey IDs."""
    if conditions is None:
        conditions = ["wt_MG132", "vif_MG132", "mock_MG132"]
    bait2uid = init_bait2uid(conditions, gene2uid)

    df1 = df1.copy()
    df2 = df2.copy()
    df3 = df3.copy()
    df1.loc[:, "BaitUID"] = [bait2uid[i] for i in df1["Bait"].values]
    df2.loc[:, "BaitUID"] = [bait2uid[i] for i in df2["Bait"].values]
    df3.loc[:, "BaitUID"] = [bait2uid[i] for i in df3["Bait"].values]

    df_all = pd.concat([df1, df2, df3])
    df_all.loc[:, "Prey"] = [
        VIRAL_PROTEIN_REMAPPING.get(i, i) for i in df_all["Prey"].values
    ]
    return df_all


def init_bait_and_condition_lookup(
    df_all: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build {'CBFBwt_MG132': 'CBFB', ...} and {'CBFBwt_MG132': 'wt', ...} lookups."""
    conditions = ["wt", "vif", "mock"]
    baits = ["CBFB", "ELOB", "CUL5", "LRR1"]
    bait2bait = {}
    bait2condition = {}
    for bait in baits:
        for condition in conditions:
            key = bait + condition + "_MG132"
            bait2bait[key] = bait
            bait2condition[key] = condition
    return bait2bait, bait2condition


def update_df_all_bait_and_condition(df_all: pd.DataFrame) -> pd.DataFrame:
    """Reindex df_all by prey gene stem and add 'bait'/'condition' columns parsed from the Bait column."""
    df_all = df_all.copy()
    df_all.index = [i.split("_")[0] for i in df_all["PreyGene"]]
    bait2bait, bait2condition = init_bait_and_condition_lookup(df_all)
    df_all.loc[:, "bait"] = [bait2bait[i] for i in df_all["Bait"].values]
    df_all.loc[:, "condition"] = [bait2condition[i] for i in df_all["Bait"].values]
    return df_all


def parse_spec(
    df: pd.DataFrame,
    spec_colname: str = "Spec",
    ctrl_colname: str = "ctrlCounts",
    n_spec: int = 4,
    n_ctrl: int = 12,
) -> pd.DataFrame:
    """Expand the pipe-delimited Spec/ctrlCounts columns into per-replicate r1..rN / c1..cN columns."""
    df = df.copy()
    rsel = [f"r{i}" for i in range(1, n_spec + 1)]
    csel = [f"c{i}" for i in range(1, n_ctrl + 1)]
    specs = np.array([list(map(int, i.split("|"))) for i in df[spec_colname].values])
    ctrls = np.array([list(map(int, i.split("|"))) for i in df[ctrl_colname].values])
    assert specs.shape == (len(df), n_spec)
    assert ctrls.shape == (len(df), n_ctrl)
    for i, rcol in enumerate(rsel):
        df.loc[:, rcol] = specs[:, i]
    for i, ccol in enumerate(csel):
        df.loc[:, ccol] = ctrls[:, i]
    return df

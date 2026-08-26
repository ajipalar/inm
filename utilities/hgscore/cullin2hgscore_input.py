"""Build the per-run/bait/prey spectral-count table used as HGSCore's R-tool input."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from cullin_bsasa_prep import (
    init_dfs,
    init_df_all,
    parse_spec,
    update_df_all_bait_and_condition,
)

THIS_DIR = Path(__file__).resolve().parent
INM_ROOT = THIS_DIR.parent.parent

DEFAULT_XLSX_PATH = INM_ROOT / "data" / "cullin" / "1-s2.0-S1931312819302537-mmc2.xlsx"
DEFAULT_GENE2UID_PATH = INM_ROOT / "data" / "table1.csv"
DEFAULT_DF_NEW_PKL_PATH = INM_ROOT / "notebooks" / "df_new.pkl"
DEFAULT_IDMAP_OUT_PATH = THIS_DIR / "cullin_hgscore_idmap.json"
DEFAULT_HGSCORE_INPUT_OUT_PATH = THIS_DIR / "cullin_hgscore_input.csv"


def build_hgscore_input(
    df_new: pd.DataFrame, rsel: List[str]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Flatten df_new's per-replicate spectral counts into HGSCore's long idRun/idBait/idPrey/countPrey/lenPrey format, skipping zero counts."""
    id_run, id_bait, id_prey, count_prey, len_prey = [], [], [], [], []
    run_name2id_run: Dict[str, int] = {}
    next_run_id = 1
    for _, row in df_new.iterrows():
        bait = row["bait"]
        condition = row["condition"]
        prey = row["PreyName"]
        aa_len = row["aa_seq_len"]
        for j, replicate in enumerate(rsel):
            count = row[replicate]
            run_name = f"{bait}_{condition}_{j}"
            if run_name not in run_name2id_run:
                run_name2id_run[run_name] = next_run_id
                next_run_id += 1
            if count != 0:
                id_run.append(run_name2id_run[run_name])
                id_bait.append(bait)
                id_prey.append(prey)
                count_prey.append(count)
                len_prey.append(aa_len)

    hgscore_df = pd.DataFrame(
        {
            "idRun": id_run,
            "idBait": id_bait,
            "idPrey": id_prey,
            "countPrey": count_prey,
            "lenPrey": len_prey,
        }
    )
    return hgscore_df, run_name2id_run


def run(
    xlsx_path: Path,
    gene2uid_path: Path,
    df_new_pkl_path: Path,
    idmap_out_path: Path,
    hgscore_input_out_path: Path,
) -> None:
    """Build the HGSCore input CSV and run-id map from the cullin AP-MS data."""
    gene2uid_df = pd.read_csv(gene2uid_path)
    gene2uid = {gene: uid for gene, uid in gene2uid_df.values}

    df1, df2, df3 = init_dfs(str(xlsx_path))
    df_all = init_df_all(df1, df2, df3, gene2uid)
    df_all = update_df_all_bait_and_condition(df_all)
    df_all = parse_spec(df_all)

    df_new = pd.read_pickle(df_new_pkl_path)
    rsel = [f"r{i}" for i in range(1, 5)]
    hgscore_df, run_name2id_run = build_hgscore_input(df_new, rsel)

    idmap_out_path.write_text(json.dumps(run_name2id_run))
    hgscore_df.to_csv(hgscore_input_out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx-path", type=Path, default=DEFAULT_XLSX_PATH)
    parser.add_argument("--gene2uid-path", type=Path, default=DEFAULT_GENE2UID_PATH)
    parser.add_argument("--df-new-pkl-path", type=Path, default=DEFAULT_DF_NEW_PKL_PATH)
    parser.add_argument("--idmap-out-path", type=Path, default=DEFAULT_IDMAP_OUT_PATH)
    parser.add_argument(
        "--hgscore-input-out-path", type=Path, default=DEFAULT_HGSCORE_INPUT_OUT_PATH
    )
    args = parser.parse_args()
    run(
        args.xlsx_path,
        args.gene2uid_path,
        args.df_new_pkl_path,
        args.idmap_out_path,
        args.hgscore_input_out_path,
    )


if __name__ == "__main__":
    main()

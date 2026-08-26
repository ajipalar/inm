"""Build spectral count table and composite table from AP-MS Excel data."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_spec_table_and_composites(
    xlsx_path: Union[str, Path],
    sheet_nums: list,
    header: int = 0,
    bait_col_name: str = "Bait",
    prey_col_name: str = "Prey",
    ms_score_col_name: str = "SaintScore",
    spec_col: str = "Spec",
    ctrl_col: Optional[str] = "ctrlCounts",
    prey_gene_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Parse AP-MS Excel data into a spectral count matrix and composite table.

    Returns a 3-tuple ``(spec_table, composites, prey_gene_map)`` where:

    * ``spec_table`` is a DataFrame indexed by prey with columns = individual
      spectral count observations (spec + controls interleaved).
    * ``composites`` is a per-row bait/prey/SAINT-score table.
    * ``prey_gene_map`` is a ``Dict[str, str]`` mapping UniProt ID → gene name,
      populated only when ``prey_gene_col`` is provided; otherwise ``{}``.
    """
    sid = []
    cb = []
    cp = []
    mscore = []
    sid_dict = {}
    k = 0
    prey_condition_dict = {}
    prey_gene_map: Dict[str, str] = {}

    for sheet_num in sheet_nums:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_num, header=header)
        for _, r in df.iterrows():
            bait_name = r[bait_col_name]
            assert isinstance(bait_name, str)
            if bait_name not in sid_dict:
                sid_dict[bait_name] = k
                k += 1
            SID = sid_dict[bait_name]
            bait = bait_name[0:4]
            sid.append(SID)
            cb.append(bait)
            prey = r[prey_col_name]
            cp.append(prey)
            saint = r[ms_score_col_name]
            assert 0 <= saint <= 1
            mscore.append(saint)
            if prey not in prey_condition_dict:
                prey_condition_dict[prey] = {}
            assert SID not in prey_condition_dict[prey]

            if prey_gene_col and prey not in prey_gene_map:
                prey_gene_map[prey] = str(r[prey_gene_col])

            spec = str(r[spec_col])
            spec = spec.split("|") if "|" in spec else [spec]

            if ctrl_col:
                ctrl = str(r[ctrl_col])
                ctrl = ctrl.split("|") if "|" in ctrl else [ctrl]
                prey_condition_dict[prey][SID] = {"spec": spec, "ctrl": ctrl}
            else:
                prey_condition_dict[prey][SID] = {"spec": spec}

    composites = pd.DataFrame({"SID": sid, "Bait": cb, "Prey": cp, "MSscore": mscore})

    # Gather per-prey spectral count vectors (spec counts + non-duplicate controls)
    found_controls: Dict = defaultdict(list)
    sim_dict: Dict = defaultdict(list)
    for prey in prey_condition_dict:
        for condition in prey_condition_dict[prey]:
            spec = prey_condition_dict[prey][condition]["spec"]
            if ctrl_col:
                ctrl = prey_condition_dict[prey][condition]["ctrl"]
                if ctrl in found_controls[prey]:
                    sim_dict[prey] += spec
                else:
                    # Deduplicate all-identical control replicates
                    if (
                        len(ctrl) >= 12
                        and (ctrl[0:4] == ctrl[4:8])
                        and (ctrl[4:8] == ctrl[8:12])
                    ):
                        ctrl = ctrl[0:4]
                    found_controls[prey].append(ctrl)
                    sim_dict[prey] += spec + ctrl
            else:
                sim_dict[prey] += spec

    max_conditions = max(len(v) for v in sim_dict.values())
    nprey = len(sim_dict)
    spec_array = np.zeros((nprey, max_conditions), dtype=int)
    index = []
    for z, (prey, conditions) in enumerate(sim_dict.items()):
        index.append(prey)
        spec_array[z, : len(conditions)] = np.array([int(c) for c in conditions])

    spec_table = pd.DataFrame(spec_array, index=index)
    return spec_table, composites, prey_gene_map


def run(cfg: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    """Build spec_table.tsv and composite_table.tsv from AP-MS Excel data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocess"]

    prey_gene_col = prep_cfg.get("prey_gene_col")

    spec_table, composites, prey_gene_map = get_spec_table_and_composites(
        xlsx_path=data_cfg["xlsx_path"],
        sheet_nums=prep_cfg["sheet_nums"],
        header=prep_cfg.get("header", 0),
        bait_col_name=prep_cfg["bait_col"],
        prey_col_name=prep_cfg["prey_col"],
        ms_score_col_name=prep_cfg["ms_score_col"],
        spec_col=prep_cfg.get("spec_col", "Spec"),
        ctrl_col=prep_cfg.get("ctrl_col", "ctrlCounts"),
        prey_gene_col=prey_gene_col,
    )

    spec_path = output_dir / "spec_table.tsv"
    comp_path = output_dir / "composite_table.tsv"
    spec_table.to_csv(spec_path, sep="\t", index=True, header=False)
    composites.to_csv(comp_path, sep="\t", index=False)

    if prey_gene_map:
        names_path = output_dir / "prey_names.tsv"
        pd.DataFrame(
            list(prey_gene_map.items()), columns=["UniProt", "GeneName"]
        ).to_csv(names_path, sep="\t", index=False)
        logger.info(
            "Wrote spec_table (%d prey) to %s, composite_table (%d rows) to %s,"
            " and prey_names.tsv (%d entries) to %s",
            len(spec_table),
            spec_path,
            len(composites),
            comp_path,
            len(prey_gene_map),
            names_path,
        )
    else:
        logger.info(
            "Wrote spec_table (%d prey) to %s and composite_table (%d rows) to %s",
            len(spec_table),
            spec_path,
            len(composites),
            comp_path,
        )

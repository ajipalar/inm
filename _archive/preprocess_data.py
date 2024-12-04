"""
Preprocesses data and references from the raw form to a format for modeling
and benchmarking.
Data is read from ../data/{DATASET}
Processed data is written to ../data/processed/{PROCESSED_NAME}
"""

import click
from collections import defaultdict 
from pathlib import Path
import numpy as np
import pandas as pd

_preprocess_data = {
    "dub" : preprocess_dub,
    "cullin" : preprocess_cullin,
    "tip49" : preprocess_tip49,
    "sars2" : preprocess_sars2,
    }

_preprocess_reference = {
    "biogrid" : preprocess_biogrid,
    "huri" : preprocess_huri,
    "huMAP2": preprocess_humap2
        }

_raw_data_paths = {
    "dub" : "../data/dub/41592_2011_BFnmeth1541_MOESM593_ESM.xls",
    "cullin" : "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    "tip49" : "../data/tip49/NIHMS252278-supplement-2.xls",
    "sars2" : "../data/gordon_sars_cov_2/41586_2020_2286_MOESM5_ESM.xlsx",
        }

_raw_reference_paths = {
        "biogrid" : "../data/biogrid",
        "huri" : "../data/huri",
        "huMAP2" : "../data/humap2",
}


@click.command()
@click.option("--ds-name")
@click.option("--overwrite", is_flag=True, default=False)
def main(ds_name):
    if ds_name in _raw_data_paths:
        preprocess_data(ds_name)
    elif ds_name in _raw_reference_paths:
        preprocess_reference(ds_name)
    else:
        raise ValueError(f"Unkown dataset name: {ds_name}")

def get_spec_table_and_composites(xlsx_path, sheet_nums,
                                  header=0,
                                  bait_col_name="Bait",
                                  prey_colname="Prey",
                                  ms_score_colname="SaintScore",
                                  ctrl_col='ctrlCounts'):
    # Make the composite table
    sid = []
    cb = []
    cp = []
    mscore = []
    sid_dict = {}
    k = 0
    prey_condition_dict = {}
    for sheet_num in range(sheet_nums): 
        df = pd.read_excel(xlsx_path, sheet_name=sheet_num, header=header)
        for i, r in df.iterrows():
            bait_name = r[bait_col_name]
            assert isinstance(bait_name, str)
            if bait_name not in sid_dict:
                sid_dict[bait_name] = k 
                k += 1
            SID = sid_dict[bait_name]
            bait = bait_name[0:4]
            sid.append(SID)
            cb.append(bait)
            prey = r[prey_colname]
            cp.append(prey)
            saint = r[ms_score_colname]
            assert 0 <= saint <= 1
            mscore.append(saint)
            if prey not in prey_condition_dict:
                prey_condition_dict[prey] = {} 
            assert SID not in prey_condition_dict[prey]
            spec = r['Spec']
            spec = str(spec)
            if "|" in spec:
                spec = spec.split("|")
            else:
                spec = [spec]
            if ctrl_col:
                ctrl = r[ctrl_col]
                ctrl = str(ctrl)
                if "|" in ctrl:
                    ctrl = ctrl.split("|")
                else:
                    ctrl = [ctrl]
                prey_condition_dict[prey][SID] = {"spec" : spec, "ctrl" : ctrl}
            else:
                prey_condition_dict[prey][SID] = {"spec" : spec} 
    composites = pd.DataFrame(
                {"SID" : sid, "Bait" : cb,
                 "Prey" : cp, "MSscore" : mscore})
    # Gather the conditions
    found_controls = defaultdict(list) 
    sim_dict = defaultdict(list) 
    for prey in prey_condition_dict:
        for condition in prey_condition_dict[prey]:
            spec = prey_condition_dict[prey][condition]["spec"]
            if ctrl_col:
                ctrl = prey_condition_dict[prey][condition]["ctrl"]
                if ctrl in found_controls[prey]:
                    sim_dict[prey] += (spec) 
                else:
                    if (ctrl[0:4] == ctrl[4:8]) and (ctrl[4:8] == ctrl[8:12]):
                        ctrl = ctrl[0:4]
                    found_controls[prey].append(ctrl)
                    sim_dict[prey] += (spec + ctrl)
            else:
                sim_dict[prey] += (spec)
    max_conditions = 0
    for prey, conditions in sim_dict.items():
        max_conditions = max(max_conditions, len(conditions))
    nprey = len(sim_dict)
    spec_table = np.zeros((nprey, max_conditions), dtype=int)
    index = []
    for z, (prey, conditions) in enumerate(sim_dict.items()):
        assert len(conditions) > 0
        index.append(prey)
        conditions = [int(i) for i in conditions]
        spec_table[z, 0:len(conditions)] = np.array(conditions)
    spec_table = pd.DataFrame(spec_table, index = index) 
    return spec_table, composites

def preprocess_cullin():
    spec_table, composites = get_spec_table_and_composites(_raw_data_paths["cullin"], 3) 
    composites.to_csv("../data/processed/cullin/composite_table.tsv", sep="\t", index=False)
    spec_table.to_csv("../data/processed/cullin/spec_table.tsv", sep="\t", index=True, header=False)
    return df

def preprocess_dub():
    spec_table, composites = get_spec_table_and_composites(_raw_data_paths["dub"], 1, ctrl_col=None,
                                                           header=2, ms_score_colname="SAINT")
    composites.to_csv("../data/processed/dub/composite_table.tsv", sep="\t", index=False)
    spec_table.to_csv("../data/processed/dub/spec_table.tsv", sep="\t", index=True, header=False)

def preprocess_tip49():
    spec_table, composites = get_spec_table_and_composites(_raw_data_paths["tip49"], 1, 
                                                           header=2, ms_score_colname="SAINT")
    composites.to_csv("../data/processed/tip49/composite_table.tsv", sep="\t", index=False)
    spec_table.to_csv("../data/processed/tip49/spec_table.tsv", sep="\t", index=True, header=False)

def preprocess_sars2():
    spec_table, composites = get_spec_table_and_composites(_raw_data_paths["sars2"], 1,
                                                           prey_colname="Preys",
                                                           ctrl_col="CtrlCounts",
                                                           header=0,
                                                           ms_score_colname="SaintScore")
    composites.to_csv("../data/processed/gordon_sars_cov_2/composite_table.tsv", sep="\t", index=False)
    spec_table.to_csv("../data/processed/gordon_sars_cov_2/spec_table.tsv", sep="\t", index=True, header=False)


#References 

def preprocess_biogrid():
    ...

def preprocess_huri():
    ...

def preprocess_humap2():
    ...

def preprocess_data(ds_name):
    _preprocess_data[ds_name]()

def preprocess_reference(ds_name):
    _preprocess_reference[ds_name]()


if __name__ == "__main__":
    main()

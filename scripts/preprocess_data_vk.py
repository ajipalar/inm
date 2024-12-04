import click
from collections import defaultdict 
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import shutil

logger = logging.getLogger(__name__)

@click.command()
@click.option("--ds-name")
@click.option("--dont-remap", is_flag=True, default=False)
def main(ds_name, dont_remap):
    out_path = "../data/processed/" + ds_name
    do_remap = not dont_remap 
    if ds_name in _raw_data_paths:
        preprocess_data(ds_name, enforce_bait_remapping=do_remap)
    elif ds_name in _raw_reference_paths:
        preprocess_reference(ds_name)
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")

def remove_corrupt_rows(df, bait_colname, prey_colname, spoke_score_colname):
    """Removes corrupt rows where identifiers are incorrectly formatted."""
    sel = []
    for i, r in df.iterrows():
        val = isinstance(r[bait_colname], str) and isinstance(r[prey_colname], str)
        sel.append(val)
    sel = np.array(sel)
    n_corrupt = np.sum(~sel)
    if n_corrupt > 0:
        logging.warning(f"N corrupt lines : {n_corrupt}")
        for i, r in df.loc[~sel, :].iterrows():
            logging.warning(f"Corrupt line {i} : {r[bait_colname], r[prey_colname], r[spoke_score_colname]}. Removing.")
    df = df[sel] 
    logging.info(f"DF SHAPE : {df.shape}")
    assert isinstance(df, pd.DataFrame)
    return df

def get_empty_spec_table(xlsx_path, sheet_nums, header, bait_colname, prey_colname, spoke_score_colname, n_spec_replicates, n_ctrl_replicates):
    colnames = []
    prey_names = []
    excel_file = pd.ExcelFile(xlsx_path)

    if sheet_nums is None:
        sheet_nums = len(excel_file.sheet_names)

    total_sheets = len(excel_file.sheet_names)
    if sheet_nums > total_sheets:
        logging.warning(f"Requested {sheet_nums} sheets but only {total_sheets} are available. Adjusting to {total_sheets}.")
        sheet_nums = total_sheets

    for sheet_number in range(sheet_nums):
        if sheet_number >= total_sheets:
            logging.warning(f"Sheet index {sheet_number} is invalid. Skipping.")
            continue

        df = pd.read_excel(xlsx_path, sheet_name=sheet_number, header=header)
        df = remove_corrupt_rows(
                df,
                bait_colname=bait_colname,
                prey_colname=prey_colname,
                spoke_score_colname=spoke_score_colname)

        for bait in set(df[bait_colname].values):
            bait_name = bait.strip()  # Remove any leading/trailing whitespace
            for i in range(n_spec_replicates):
                colnames.append(bait_name + f"_r{i}")
            for i in range(n_ctrl_replicates):
                colnames.append(bait_name + f"_c{i}")
            sel = df[bait_colname] == bait
            for i, r in df[sel].iterrows():
                prey_names.append(r[prey_colname].strip())

    prey_names = list(set(prey_names))
    n = len(prey_names)
    m = len(colnames)
    spec_table = pd.DataFrame(np.zeros((n, m)), columns=colnames, index=prey_names)      
    return spec_table

def get_spec_table(
        xlsx_path,
        sheet_nums=None,  # Optional sheet number
        header=0,
        bait_colname="Bait",
        prey_colname="Prey",
        spoke_score_colname="SaintScore",
        ctrl_colname="ctrlCounts",
        spec_colname="Spec",
        spec_count_sep="|",
        enforce_bait_remapping=True):

    logging.info(f"PARAMS")
    logging.info(f"    In path: {xlsx_path}")
    logging.info(f"    sheet_nums {sheet_nums}")
    logging.info(f"    bait_colname: {bait_colname}")
    logging.info(f"    prey_colname: {prey_colname}")
    logging.info(f"    spoke_score_colname: {spoke_score_colname}")
    logging.info(f"    ctrl_colname: {ctrl_colname}")
    logging.info(f"    spec_colname: {spec_colname}")
    logging.info(f"    spec_count_sep: {spec_count_sep}")

    # Load the Excel file to get sheet information
    excel_file = pd.ExcelFile(xlsx_path)

    if sheet_nums is None or sheet_nums > len(excel_file.sheet_names):
        sheet_nums = len(excel_file.sheet_names)

    df_sample = pd.read_excel(xlsx_path, sheet_name=0, header=header)
    if spec_colname in df_sample.columns and ctrl_colname in df_sample.columns:
        n_spec_replicates = len(df_sample[spec_colname].iloc[0].split(spec_count_sep))
        n_ctrl_replicates = len(df_sample[ctrl_colname].iloc[0].split(spec_count_sep))
    else:
        raise ValueError(f"Specified columns '{spec_colname}' or '{ctrl_colname}' not found in the data.")

    spec_table = get_empty_spec_table(
        xlsx_path=xlsx_path,
        sheet_nums=sheet_nums,
        header=header,
        bait_colname=bait_colname,
        prey_colname=prey_colname,
        spoke_score_colname=spoke_score_colname,
        n_spec_replicates=n_spec_replicates,
        n_ctrl_replicates=n_ctrl_replicates)

    colnames = []
    prey_names = []
    seen_control_tables = []
    keep_controls = []
    keep_experiments = []

    for sheet_number in range(sheet_nums):
        if sheet_number >= len(excel_file.sheet_names):
            logging.warning(f"Sheet number {sheet_number} is out of range. Skipping.")
            continue

        df = pd.read_excel(xlsx_path, sheet_name=sheet_number, header=header)
        df = remove_corrupt_rows(df, bait_colname, prey_colname, spoke_score_colname)
        unique_baits = set(df[bait_colname].values)

        for bait in unique_baits:
            bait_name = bait.strip()
            sel = df[bait_colname] == bait

            for _, r in df[sel].iterrows():
                prey_name = r[prey_colname].strip()
                evals = [int(k) for k in r[spec_colname].split(spec_count_sep)]
                cvals = [int(k) for k in r[ctrl_colname].split(spec_count_sep)]

                if len(evals) < n_spec_replicates:
                    evals += [0] * (n_spec_replicates - len(evals))
                elif len(evals) > n_spec_replicates:
                    evals = evals[:n_spec_replicates]

                if len(cvals) < n_ctrl_replicates:
                    cvals += [0] * (n_ctrl_replicates - len(cvals))
                elif len(cvals) > n_ctrl_replicates:
                    cvals = cvals[:n_ctrl_replicates]

                experiment_columns = [f"{bait_name}_r{i}" for i in range(n_spec_replicates)]
                control_columns = [f"{bait_name}_c{i}" for i in range(n_ctrl_replicates)]

                if prey_name not in spec_table.index:
                    logging.warning(f"Prey {prey_name} not found in spec_table index. Skipping assignment.")
                    continue

                spec_table.loc[prey_name, experiment_columns] = evals
                spec_table.loc[prey_name, control_columns] = cvals

            control_table = df.loc[sel, [prey_colname, ctrl_colname]]
            keep_experiments += experiment_columns

            if len(seen_control_tables) == 0:
                keep_controls += control_columns
                seen_control_tables.append(control_table)
            else:
                for seen_table in seen_control_tables:
                    temp = None
                    shared_prey = set(control_table[prey_colname].values).intersection(
                        set(seen_table[prey_colname].values))
                    shared_prey = list(shared_prey)

                    sel1 = [k in shared_prey for k in control_table[prey_colname].values]
                    sel2 = [k in shared_prey for k in seen_table[prey_colname].values]

                    t1 = control_table[sel1].copy()
                    t2 = seen_table[sel2].copy()

                    t1 = t1.sort_values(prey_colname)
                    t2 = t2.sort_values(prey_colname)

                    assert np.alltrue(t1[prey_colname].values == t2[prey_colname].values)
                    if np.alltrue(t1[ctrl_colname].values == t2[ctrl_colname].values):
                        break
                    else:
                        temp = control_table
                if temp is not None:
                    seen_control_tables.append(control_table)
                    keep_controls += control_columns

    all_columns_to_keep = keep_experiments + keep_controls
    spec_table = spec_table.loc[:, all_columns_to_keep]

    return spec_table

def get_mapping_dict(xlsx_path):
    fpath = Path(xlsx_path).parent / "bait_remapping" 
    keyword = fpath.parent.stem 
    to_path = Path("../data/processed") / keyword
    to_path = to_path / "bait_remapping"
    to_path = str(to_path)
    fpath = str(fpath)
    df = pd.read_csv(fpath, names=['b', 'p'])
    d = {r['b']:r['p'] for i, r in df.iterrows()}
    shutil.copy(fpath, to_path)
    return d

def get_spec_table_and_composites(xlsx_path,
                                  sheet_nums=None,
                                  header=0,
                                  bait_col_name="Bait",
                                  prey_colname="Prey",
                                  ms_score_colname="SaintScore",
                                  ctrl_col='ctrlCounts',
                                  spec_colname="Spec",
                                  spec_count_sep="|",
                                  bait_parser_strat="all",
                                  control_handle_strat="all",
                                  prey_parser_strat="general",
                                  enforce_bait_remapping=True,
                                  return_many=False):
    # Load the Excel file and get the number of sheets
    excel_file = pd.ExcelFile(xlsx_path)
    total_sheets = len(excel_file.sheet_names)

    # If sheet_nums is None or greater than total_sheets, set it to total_sheets
    if sheet_nums is None or sheet_nums > total_sheets:
        sheet_nums = total_sheets

    # Log parameters
    logging.info(f"PARAMS")
    logging.info(f"    In path: {xlsx_path}")
    logging.info(f"    sheet_nums: {sheet_nums}")
    logging.info(f"    bait_col_name: {bait_col_name}")
    logging.info(f"    prey_colname: {prey_colname}")
    logging.info(f"    ms_score_colname: {ms_score_colname}")
    logging.info(f"    ctrl_col: {ctrl_col}")
    logging.info(f"    spec_colname: {spec_colname}")
    logging.info(f"    spec_count_sep: {spec_count_sep}")
    logging.info(f"    bait_parser_strat: {bait_parser_strat}")
    logging.info(f"    control_handle_strat: {control_handle_strat}")
    logging.info(f"    prey_parser_strat: {prey_parser_strat}")
    logging.info(f"    enforce_bait_remapping: {enforce_bait_remapping}")

    # If bait remapping is enforced, get the mapping dictionary
    if enforce_bait_remapping:
        bait_mapping_dict = get_mapping_dict(xlsx_path)

    # Lists for composite table
    sid = []
    cb = []
    cp = []
    mscore = []
    sid_dict = {}
    k = 0
    prey_condition_dict = {}

    # Loop through the valid sheets
    for sheet_num in range(sheet_nums):
        logging.info(f"Processing sheet {sheet_num}")
        df = pd.read_excel(xlsx_path, sheet_name=sheet_num, header=header)
        logging.info(f"DF SHAPE: {df.shape}")

        # Remove corrupt rows
        sel = df.apply(lambda r: isinstance(r[bait_col_name], str) and isinstance(r[prey_colname], str), axis=1)
        n_corrupt = (~sel).sum()
        if n_corrupt > 0:
            logging.warning(f"N corrupt lines: {n_corrupt}")
            for i, r in df.loc[~sel, :].iterrows():
                logging.warning(f"Corrupt line {i}: {r[bait_col_name], r[prey_colname], r[ms_score_colname]}")
        df = df[sel]
        logging.info(f"DF SHAPE after removing corrupt rows: {df.shape}")
        assert isinstance(df, pd.DataFrame)

        # Process each row in the DataFrame
        for i, r in df.iterrows():
            bait_name = r[bait_col_name]
            assert isinstance(bait_name, str)
            if bait_name not in sid_dict:
                sid_dict[bait_name] = k
                k += 1
            SID = sid_dict[bait_name]
            bait = bait_name
            if enforce_bait_remapping and bait in bait_mapping_dict:
                bait = bait_mapping_dict[bait]
            assert isinstance(bait, str), bait
            sid.append(SID)
            cb.append(bait)
            prey_name = r[prey_colname]
            assert isinstance(prey_name, str), (i, prey_name, r)
            cp.append(prey_name)
            saint = r[ms_score_colname]
            assert 0 <= saint <= 1
            mscore.append(saint)
            if prey_name not in prey_condition_dict:
                prey_condition_dict[prey_name] = {}
            assert SID not in prey_condition_dict[prey_name]
            spec = r[spec_colname]
            spec = str(spec)
            if spec_count_sep in spec:
                spec = spec.split(spec_count_sep)
            else:
                spec = [spec]
            if ctrl_col:
                ctrl = r[ctrl_col]
                ctrl = str(ctrl)
                if spec_count_sep in ctrl:
                    ctrl = ctrl.split(spec_count_sep)
                else:
                    ctrl = [ctrl]
                prey_condition_dict[prey_name][SID] = {"spec": spec, "ctrl": ctrl}
            else:
                prey_condition_dict[prey_name][SID] = {"spec": spec}

    # Create composites DataFrame
    composites = pd.DataFrame({"SID": sid, "Bait": cb, "Prey": cp, "MSscore": mscore})

    # Gather the conditions
    found_controls = defaultdict(list)
    sim_dict = defaultdict(list)
    for prey in prey_condition_dict:
        for condition in prey_condition_dict[prey]:
            spec = prey_condition_dict[prey][condition]["spec"]
            if ctrl_col:
                sim_dict[prey] += spec
            else:
                sim_dict[prey] += spec

    # Create spec_table DataFrame
    max_conditions = max(len(conditions) for conditions in sim_dict.values())
    nprey = len(sim_dict)
    spec_table = np.zeros((nprey, max_conditions), dtype=int)
    index = []
    for z, (prey, conditions) in enumerate(sim_dict.items()):
        assert len(conditions) > 0
        index.append(prey)
        conditions = np.array([int(i) for i in conditions])
        assert np.alltrue(conditions >= 0)
        spec_table[z, 0:len(conditions)] = conditions
    spec_table = pd.DataFrame(spec_table, index=index)

    if return_many:
        return spec_table, composites, prey_condition_dict, sid_dict, sim_dict
    else:
        return spec_table, composites

def preprocess_mda231(prey_colname="PreyGene", enforce_bait_remapping=True):
    spec_table, composites = get_spec_table_and_composites(
            _raw_data_paths["mda231"], 1,
            prey_colname=prey_colname,
            ctrl_col="ctrlCounts",
            header=0,
            ms_score_colname="SaintScore",
            bait_parser_strat="all",
            enforce_bait_remapping=enforce_bait_remapping)
    log_unmapped_bait(composites)
    write_tables(spec_table, composites, output_dir = Path("../data/processed/mda231"))

def log_unmapped_bait(d):
    bait = set(d['Bait'])
    prey = set(d['Prey'])
    u = bait - prey
    logging.info(f"N Bait {len(bait)}")
    if len(u) > 0:
        logging.warning(f"{len(u)} bait not found in prey")
        logging.warning(f"unfound bait {u}")

def write_tables(spec_table, composites, output_dir):
    spec_table.sort_index(inplace = True, axis = 0)
    spec_table.sort_index(inplace = True, axis = 1)
    composites.to_csv(output_dir / "composite_table.tsv", sep="\t", index=False)
    spec_table.to_csv(output_dir / "spec_table.tsv", sep="\t", index=True, header=True)

def preprocess_data(ds_name, enforce_bait_remapping):
    _preprocess_data[ds_name](enforce_bait_remapping=enforce_bait_remapping)

def preprocess_reference(ds_name):
    _preprocess_reference[ds_name]()

_preprocess_data = {
    
    "mda231" : preprocess_mda231,
    
    }

_raw_data_paths = {
    "dub" : "../data/dub/41592_2011_BFnmeth1541_MOESM593_ESM.xls",
    "cullin" : "../data/cullin/mmc2.xlsx",
    "tip49" : "../data/tip49/NIHMS252278-supplement-2.xls",
    "mda231" : "../data/mda231/mda231.xlsx",
    "gordon_sars_cov_2": "../data/gordon_sars_cov_2/41586_2020_2286_MOESM5_ESM.xlsx",
        }

if __name__ == "__main__":
    main()
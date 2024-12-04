import click
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import shutil
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    """Remove rows with corrupted bait or prey identifiers."""
    sel = df.apply(lambda r: isinstance(r[bait_colname], str) and isinstance(r[prey_colname], str), axis=1)
    n_corrupt = (~sel).sum()
    if n_corrupt > 0:
        logging.warning(f"N corrupt lines: {n_corrupt}")
        for i, r in df.loc[~sel, :].iterrows():
            logging.warning(f"Corrupt line {i}: {r[bait_colname]}, {r[prey_colname]}, {r[spoke_score_colname]}. Removing.")
    df = df[sel]
    logging.info(f"DF SHAPE: {df.shape}")
    return df

def get_empty_spec_table(df_list, bait_colname, prey_colname, spoke_score_colname, n_spec_replicates, n_ctrl_replicates):
    colnames = []
    prey_names = set()

    for df in df_list:
        df = remove_corrupt_rows(df, bait_colname=bait_colname, prey_colname=prey_colname, spoke_score_colname=spoke_score_colname)

        for bait in df[bait_colname].unique():
            bait_name = bait.strip().upper()  # Standardize bait name
            for i in range(n_spec_replicates):
                colnames.append(f"{bait_name}_r{i}")
            for i in range(n_ctrl_replicates):
                colnames.append(f"{bait_name}_c{i}")

            sel = df[bait_colname].str.strip().str.upper() == bait_name
            for prey in df[sel][prey_colname].values:
                prey_name = prey.strip().upper()  # Standardize prey name
                prey_names.add(prey_name)

    spec_table = pd.DataFrame(np.zeros((len(prey_names), len(colnames))), columns=colnames, index=sorted(prey_names))
    logging.info(f"Spec table created with columns: {spec_table.columns}")
    return spec_table

def get_spec_table(
        xlsx_path,
        sheet_nums=None,
        header=0,
        bait_colname="Bait",
        prey_colname="PreyGene",
        spoke_score_colname="SaintScore",
        ctrl_colname="ctrlCounts",
        spec_colname="Spec",
        spec_count_sep="|",
        enforce_bait_remapping=True):

    logging.info(f"PARAMS")
    logging.info(f"    In path: {xlsx_path}")
    logging.info(f"    sheet_nums: {sheet_nums}")
    logging.info(f"    bait_colname: {bait_colname}")
    logging.info(f"    prey_colname: {prey_colname}")
    logging.info(f"    spoke_score_colname: {spoke_score_colname}")
    logging.info(f"    ctrl_colname: {ctrl_colname}")
    logging.info(f"    spec_colname: {spec_colname}")
    logging.info(f"    spec_count_sep: {spec_count_sep}")

    # Load the Excel file once
    excel_file = pd.ExcelFile(xlsx_path)

    # Update `sheet_nums` to use all available sheets if not specified or incorrect
    if sheet_nums is None or sheet_nums > len(excel_file.sheet_names):
        sheet_nums = len(excel_file.sheet_names)

    # Load all required sheets into DataFrames
    df_list = [excel_file.parse(sheet_name=sheet, header=header) for sheet in range(sheet_nums)]

    # Determine the number of replicates dynamically based on the first row of the data
    df_sample = df_list[0]
    if spec_colname in df_sample.columns and ctrl_colname in df_sample.columns:
        n_spec_replicates = len(df_sample[spec_colname].iloc[0].split(spec_count_sep))
        n_ctrl_replicates = len(df_sample[ctrl_colname].iloc[0].split(spec_count_sep))
    else:
        raise ValueError(f"Specified columns '{spec_colname}' or '{ctrl_colname}' not found in the data.")

    # Create spec_table using the updated `get_empty_spec_table()` function
    spec_table = get_empty_spec_table(
        df_list=df_list,
        bait_colname=bait_colname,
        prey_colname=prey_colname,
        spoke_score_colname=spoke_score_colname,
        n_spec_replicates=n_spec_replicates,
        n_ctrl_replicates=n_ctrl_replicates)

    # Populate spec_table with values
    for idx, df in enumerate(df_list):
        logging.info(f"Processing sheet {idx + 1}/{sheet_nums}")
        start_time = time.time()
        df = remove_corrupt_rows(df, bait_colname, prey_colname, spoke_score_colname)
        unique_baits = df[bait_colname].unique()

        for bait in unique_baits:
            bait_name = bait.strip().upper()

            # Filter rows for the specific bait
            sel = df[bait_colname].str.strip().str.upper() == bait_name

            # Extract unique prey names for this bait
            prey_names = df[sel][prey_colname].str.strip().str.upper().unique()

            for prey_name in prey_names:
                prey_name = prey_name.strip().upper()

                if prey_name not in spec_table.index:
                    logging.warning(f"Prey {prey_name} not found in spec_table index. Skipping assignment.")
                    continue

                # Get spec and control values for the specific prey
                evals = [int(k) for k in df[sel & (df[prey_colname].str.strip().str.upper() == prey_name)][spec_colname].iloc[0].split(spec_count_sep)]
                cvals = [int(k) for k in df[sel & (df[prey_colname].str.strip().str.upper() == prey_name)][ctrl_colname].iloc[0].split(spec_count_sep)]

                # Ensure `evals` matches `n_spec_replicates`
                evals = (evals + [0] * n_spec_replicates)[:n_spec_replicates]

                # Ensure `cvals` matches `n_ctrl_replicates`
                cvals = (cvals + [0] * n_ctrl_replicates)[:n_ctrl_replicates]

                # Get experiment and control columns
                experiment_columns = [f"{bait_name}_r{i}" for i in range(n_spec_replicates)]
                control_columns = [f"{bait_name}_c{i}" for i in range(n_ctrl_replicates)]

                # Assign values to spec_table
                spec_table.loc[prey_name, experiment_columns] = evals
                spec_table.loc[prey_name, control_columns] = cvals

        logging.info(f"Finished processing sheet {idx + 1}/{sheet_nums} in {time.time() - start_time:.2f} seconds")

    logging.info(f"Final spec_table columns: {spec_table.columns}")
    return spec_table

def preprocess_spec_table(input_path,
                          output_dir,
                          sheet_nums,
                          prey_colname,
                          enforce_bait_remapping=False,
                          filter_kw=None,
                          mode="general"):

    # Get spec_table and composites
    spec_table = get_spec_table(
        xlsx_path=input_path,
        sheet_nums=sheet_nums,
        prey_colname=prey_colname,
        enforce_bait_remapping=enforce_bait_remapping
    )

    # Log unmapped baits
    logging.info(f"Spec table created with shape: {spec_table.shape}")

    # Write the tables to the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_table.to_csv(output_dir / "spec_table.tsv", sep="\t", index=True, header=True)

def preprocess_data(ds_name, enforce_bait_remapping):
    if ds_name == "mda231":
        spec_table = get_spec_table(
            xlsx_path=_raw_data_paths[ds_name],
            bait_colname="Bait",
            prey_colname="PreyGene",
            spoke_score_colname="SaintScore",
            ctrl_colname="ctrlCounts",
            spec_colname="Spec",
            enforce_bait_remapping=enforce_bait_remapping
        )
        output_dir = Path("../data/processed/mda231")
        output_dir.mkdir(parents=True, exist_ok=True)
        spec_table.to_csv(output_dir / "spec_table.tsv", sep="\t", index=True, header=True)
    else:
        raise NotImplementedError(f"Processing for dataset {ds_name} is not implemented.")

def preprocess_reference(ds_name):
    raise NotImplementedError(f"Reference processing for {ds_name} is not implemented.")

_raw_data_paths = {
    "dub": "../data/dub/41592_2011_BFnmeth1541_MOESM593_ESM.xls",
    "cullin": "../data/cullin/mmc2.xlsx",
    "tip49": "../data/tip49/NIHMS252278-supplement-2.xls",
    "mda231": "../data/mda231/mda231.xlsx",
    "gordon_sars_cov_2": "../data/gordon_sars_cov_2/41586_2020_2286_MOESM5_ESM.xlsx",
}

_raw_reference_paths = {
    "biogrid": "../data/biogrid",
    "huri": "../data/huri",
    "huMAP2": "../data/humap2",
}

if __name__ == "__main__":
    main()

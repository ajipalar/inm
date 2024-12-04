# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import sys
sys.path.append("../../scripts/")

import preprocess_data
from preprocess_data import (
get_spec_table_and_composites
)
import numpy as np
from pathlib import Path
with open("0_model23_ll_lp_13.pkl", "rb") as f:
    d = pkl.load(f)
# -

spec_table, composites, prey_condition_dict, sid_dict, sim_dict = get_spec_table_and_composites(
    str(Path("../") / Path(preprocess_data._raw_data_paths['cullin'])),
    3,
    prey_colname="PreyGene", 
    enforce_bait_remapping = False,
    return_many = True
)

inv_sid_dict = {val : key for key, val in sid_dict.items()}

spec_table = np.zeros
for prey in prey_condition_dict:
    for condition in prey_condition_dict[prey]:
        condition_basename = inv_sid_dict[condition]
        
        
        break
    break

prey_condition_dict['vifprotein'].keys()

prey_condition_dict['ZER1'].keys()

prey_condition_dict['vifprotein'][0]

sim_dict

sid_dict

prey_condition_dict

sid_dict

condition

# +
# Set up the spec_table
spec_array = np.zeros(spec_table.shape)
index = []
column_names = []

# Find the prey with the maximal number of keys

for basename in sid_dict:
    basename = basename.removesuffix("_MG132")
    for is_control in ['e', 'c']:
        for replicate in range(1, 5):
            column_name = basename + "_" + is_control + "_" + str(replicate)
            column_names.append(column_name)
            

# +
# 1. Read in every sheet name - (key the sheet names by condition)
# - Register the prey_name
# - the Spec rows (4)
# - the control rows (12)



# +
colnames = []
prey_names = []


for sheet_number in range(3):
    df = pd.read_excel("../../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx", sheet_name=sheet_number, header=0)
    for bait in set(df['Bait'].values):
        bait_name = bait.removesuffix("_MG132")
        for i in range(4):
            colnames.append(bait_name + f"_r{i}")
        for i in range(12):
            colnames.append(bait_name + f"_c{i}")
        sel = df['Bait'] == bait
        for i, r in df[sel].iterrows():
            prey_names.append(r['PreyGene'].removesuffix("_HUMAN"))
        prey_names = list(set(prey_names))

n = len(prey_names)
m = len(colnames)
outdf = pd.DataFrame(np.zeros((n, m)), columns=colnames, index=prey_names)      
# Populate the table

colnames = []
prey_names = []

from collections import defaultdict


seen_control_tables = []
keep_controls = []
keep_experiments = []
for sheet_number in range(3):
    df = pd.read_excel("../../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx", sheet_name=sheet_number, header=0)
    for bait in set(df['Bait'].values):
        bait_name = bait.removesuffix("_MG132")
        for i in range(4):
            colnames.append(bait_name + f"_r{i}")
        for i in range(12):
            colnames.append(bait_name + f"_c{i}")
        sel = df['Bait'] == bait
        for i, r in df[sel].iterrows():
            prey_name = r['PreyGene'].removesuffix("_HUMAN")
            e_colname = bait_name + f"_r"
            c_colname = bait_name + f"_c"
            evals = [int(k) for k in r['Spec'].split("|")]
            cvals = [int(k) for k in r['ctrlCounts'].split("|")]

            # What controls to keep?
            experiment_columns = [e_colname + str(i) for i in range(4)]
            outdf.loc[prey_name, experiment_columns] = evals
            control_columns = [c_colname + str(i) for i in range(12)]
            outdf.loc[prey_name, control_columns] = cvals
        control_table = df.loc[sel, ["PreyGene", "ctrlCounts"]]
        keep_experiments += experiment_columns
        if len(seen_control_tables) == 0:
            # Keep the controls
            keep_controls += control_columns
            seen_control_tables.append(control_table)
        else:
            for seen_table in seen_control_tables:
                temp = None
                # Check the table against seen tables at the prey intersections
                shared_prey = set(control_table['PreyGene'].values).intersection(
                    set(seen_table['PreyGene'].values))
                shared_prey = list(shared_prey)
                # If the control tables are equal don't keep
                sel1 = [k in shared_prey for k in control_table['PreyGene'].values]
                sel2 = [k in shared_prey for k in seen_table['PreyGene'].values]

                t1 = control_table[sel1].copy()
                t2 = seen_table[sel2].copy()

                # Sort both tables the same way
                t1 = t1.sort_values("PreyGene")
                t2 = t2.sort_values("PreyGene")

                assert np.alltrue(t1['PreyGene'].values == t2['PreyGene'].values)
                if np.alltrue(t1['ctrlCounts'].values == t2['ctrlCounts'].values):
                    # Do not keep the control columns
                    break
                else:
                    temp = control_table
            # if temp is set then add to the seen tables
            if temp is not None:
                seen_control_tables.append(control_table)
                keep_controls += control_columns            

# Check each bait and get 
all_columns_to_keep = keep_experiments + keep_controls
outdf = outdf.loc[:, all_columns_to_keep]
# -

outdf.sort_index(ascending=False)

all_columns_to_keep

outdf.loc[['vifprotein', 'ZER1'], all_columns_to_keep]

np.corrcoef(outdf.loc['vifprotein', all_columns_to_keep], outdf.loc['ZER1', all_columns_to_keep])

np.corrcoef(spec_table.loc['vifprotein', :], spec_table.loc['ZER1', :])

shared_prey = devf()

np.alltrue(outdf == outdf)

outdf.sort_values('CUL5wt_r0')

outdf == outdf

len(control_dict)

for k in outdf.T['vifprotein']:
    print(k)
    

prey_controls

outdf.loc[['vifprotein', 'ZER1'], :]

outdf

r['Spec'].split('|')

n = len(prey_names)
m = len(colnames)
outdf = pd.DataFrame(np.zeros((n, m)), columns=colnames, index=prey_names)

outdf

r['PreyGene']

prey_condition_dict

for prey in prey_condition_dict:
    found_sids = []
    for sid in inv_sid_dict:
        if sid in prey_condition_dict:
            # Parse the spectral counts for both.
        else:
            # 0-fill
        

x = 'CBFBwt_MG132'

x.removesuffix("_MG132")

x

inv_sid_dict

# +
spec_array = np.zeros(spec_table.shape)
index = []



for z, (prey, conditions) in enumerate(sim_dict.items()):
    x = [int(i) for i in conditions]
    index.append(prey)
    spec_array[z, 0:len(x)] = x
spec_table2 = pd.DataFrame(spec_array, index = index)

# +
# Iterate over sheets saving conditions
# 
# -

sid_dict.keys()

sim_dict['ZER1']

prey_condition_dict['ZER1']

a, b = preprocess_data.handle_controls("all", prey_condition_dict, 'ZER1')

prey_condition_dict['vifprotein'][0]

prey_condition_dict['ZER1'][2]

sim_dict['vifprotein']

spec_table

sim_dict['vifprotein']

prey_condition_dict['vifprotein']

df = pd.read_csv("av_A_edgelist.tsv", sep="\t")

sum(df['b'] == 'vifprotein')

s1 = df['b'] == 'vifprotein'

plt.hist(df.loc[s1, 'w'].values, bins=100)
plt.show()

vif_edges = df[s1]

vif_edges

spec_table = pd.read_csv("spec_table.tsv", sep="\t", header=None, index_col=0)

top_vif_edges = vif_edges.loc[vif_edges['w'] > 0.99, 'a']

top_vif_edges

sel = []
for name in spec_table.index:
    if name in top_vif_edges.values:
        sel.append(name)
sel = ['vifprotein'] + sel



spec_table.loc[['vifprotein', 'ZER1'], [1, 2, 3 ,4, 17, 18, 19, 20, 21, 22, 23, 24]]

prey_condition_dict['vifprotein']

prey_condition_dict['ZER1']

sid_dict

# - High Similarity

for k in spec_table.loc['vifprotein', :]:
    print(k)


def top_nppi(df, name):
    return sum(df.loc[df['b']==name, 'w']>0.99)


names = ['vifprotein', 'nefprotein', 'tatprotein']

outdf = {name: [top_nppi(df, name)] for name in names}
outdf = pd.DataFrame(outdf)



df[s1].iloc[0:40]

spec_table.loc[['vifprotein', 'ZER1']].iloc[:, 0:]

for idx, k in enumerate(spec_table.loc['vifprotein', :]):
    print(idx, k)

df[df['a'] == 'PEBB']

df[df['a'] == 'ELOB']

df

# - ZER1. Substrate receptor for CRL2
#

"""
Find pairs of Prey that share a PDB File
"""
import pandas as pd
import sys
from pathlib import Path
from itertools import combinations

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

df = pd.read_csv("hhblits_out/SignificantPreyPDB70PairAlign.csv")
query_key = 'QueryUID'
pdb70_key = "PDB70ID"
prey_set = list(set(df[query_key].values))
df.loc[:, 'pdb_id'] = [i.split("_")[0] for i in df[pdb70_key].values]

pdb_set = set(df['pdb_id'].values)

output_cols = ["Query1", "Query2", "PDB701", "PDB702"]
subframe_cols = ["QueryUID", "PDB70ID"]
vals = []

for i, pdb_id in enumerate(pdb_set):
    assert len(pdb_id.split("_")) == 1, pdb_id
    subframe = df.loc[df['pdb_id'] == pdb_id, subframe_cols]


    # The subframe has nrows

    assert len(subframe) > 0
    if len(subframe) == 1:
        # skip
        ...
    else:
        rows = [i[1] for i in subframe.iterrows()]  # remove the label
        pairs = list(combinations(rows, 2))
        output = [(i[0][query_key], i[1][query_key], i[0][pdb70_key], i[1][pdb70_key]) for i in pairs]
        vals = vals + output

df = pd.DataFrame(vals, columns=output_cols)
nself_pairs = 0
nnonself_pairs = 0
ntotal_pairs = len(df)

self_pairs = []
non_self_pairs = []
for i, r in df.iterrows():
    if r["PDB701"] != r["PDB702"]:
        nnonself_pairs +=1
        non_self_pairs.append(r)
    else:
        nself_pairs +=1
        self_pairs.append(r)

del df
selfdf = pd.DataFrame(self_pairs) 
nonselfdf = pd.DataFrame(non_self_pairs)

eprint(f"Total pairs: {ntotal_pairs} that share PDB")
eprint(f" Self pairs: {nself_pairs} that map to the same chain")
eprint(f" Non  pairs: {nnonself_pairs} that map to different chains")

selfdf.to_csv("hhblits_out/SignificantSelfPairs.csv", index=False)
nonselfdf.to_csv("hhblits_out/SignificantNonSelfPairs.csv", index=False)

self_pair_pdbs = set([i.split("_")[0] for i in selfdf["PDB701"].values])
nonself_pdbs = set([i.split("_")[0] for i in nonselfdf['PDB701'].values])
nonself_pdbs = nonself_pdbs.union(set([i.split("_")[0] for i in nonselfdf['PDB702'].values]))

pd.DataFrame(list(self_pair_pdbs)).to_csv("hhblits_out/self_pair_pdbs.csv",index=False, header=False)
pd.DataFrame(list(nonself_pdbs)).to_csv("hhblits_out/nonself_pair_pdbs.csv",index=False, header=False)

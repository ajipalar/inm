"""
Filter Out The Significant Hits
"""

import pandas as pd
import numpy as np

df = pd.read_csv("hhblits_out/PreyPDB70PairAlign.csv")
df.loc[:, "Aligned_cols"] = np.array([int(i) for i in df["Aligned_cols"].values], dtype=int) 
df.loc[:, "Identities"] = np.array([int(i.strip("%")) for i in df["Identities"].values], dtype=int)
assert df["Evalue"].values.dtype == np.float64

df = df[df['Evalue'] <= 1e-7]
df = df[df['Aligned_cols'] >= 88]

df = df[df["Identities"] >= 30]
df.to_csv("hhblits_out/SignificantPreyPDB70PairAlign.csv", index=False)

sig_pdbs = list(set([i.split("_")[0] for i in df['PDB70ID'].values]))


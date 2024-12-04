"""
Filter by % sequence identity
"""
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf
import biotite.sequence.align
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
import sys

df = pd.read_csv("significant_cifs/chain_mapping.csv")

key = "bt_aln_percent_seq_id"
df.loc[:, key] = [float(i) for i in df[key].values]
sel = df[key] >= 0.3 
filter_df = df[sel]

filter_df.to_csv("significant_cifs/filter_chain_mapping_1-9a.csv", index=False)





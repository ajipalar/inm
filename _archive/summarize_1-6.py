import pandas as pd
import numpy as np
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

df = pd.read_csv("hhblits_out/PreyPDB70PairAlign.csv")
dfs = pd.read_csv("hhblits_out/SignificantPreyPDB70PairAlign.csv")

def h(x):
    return '{:,}'.format(x)

def lenset(df, col):
    return h(len(set(df[col].values)))

def npdbs(df):
    return h(len(set([i.split("_")[0] for i in df["PDB70ID"].values])))
eprint("HHBlits Pair Alns\n")
eprint(f"N ALN                  {h(len(df))}")
eprint(f"N PDB70                {npdbs(df)}")
eprint(f"N PDB70 Chains         {lenset(df, 'PDB70ID')}")
eprint(f"N UID                  {lenset(df, 'QueryUID')}")

eprint("\nevalue < 1e-7, match>=88, 30%  seq id\n")
eprint(f"N SIG ALN              {h(len(dfs))}")
eprint(f"N SIG PDB70            {npdbs(dfs)}")
eprint(f"N SIG PDB70 Chains     {lenset(dfs, 'PDB70ID')}")
eprint(f"N SIG UID              {lenset(dfs, 'QueryUID')}")

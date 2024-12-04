import click
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def h(x):
    return '{:,}'.format(x)

def lset(df, colname):
    return len(set(df[colname].values))

def npdbs(df):
    return lset(df, 'PDBID')

def nnan_rows(df):
    nan_rows = np.any(pd.notna(df) == False, axis=1)
    assert len(nan_rows) == len(df)
    return np.sum(nan_rows)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def pp(x):
    eprint(h(x))

stages = ["FASTA", "HHR", ]
cols = ["NPDB", "NUID", "NHHR", "NALN"] 

nrows, ncols = len(stages), len(cols)
shape = (nrows, ncols)
df = np.zeros(shape, dtype=int)

df = pd.DataFrame(data=df, columns=cols, index=stages)

nfasta = len([i for i in Path("input_sequences").iterdir() if i.suffix == ".fasta"])
nhhr = len([i for i in Path("hhblit_out").iterdir() if i.suffix == ".hhr"])

df.loc["FASTA", "NUID"] = nfasta
df.loc["HHR", "NALN"] = nhhr



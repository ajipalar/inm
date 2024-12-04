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

@click.command()
@click.option("--df-path", required=True, type=str)
@click.option("--pdbs", is_flag=True, default=False)
@click.option("--nans", is_flag=True, default=False)
@click.option("--keys", type=list, default=None, help="A list of column names to summarize")
def main(df_path, pdbs, nans, key):
    df = pd.read_csv(df_path)
    eprint(df_path)
    eprint(f"N {h(len(df))}")
    if pdbs:
       eprint(f"N pdbs {h(npdbs(df))}")
    if nans:
       eprint(f"N nan rows {h(nnan_rows(df))}")

    if keys:
        for key in keys: 
            eprint(f"N {key} {h(lset(df, key))}")


if __name__ == "__main__":
    main()

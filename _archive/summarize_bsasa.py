import pandas as pd
from pathlib import Path
from itertools import combinations
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf as mmtf
import biotite.sequence.align
import sys
import numpy as np
import click

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def pp(x):
    return '{:,}'.format(x)

def summarize(df, name):
    eprint(name)
    n_failed = []
    for i, r in df.iterrows():
        nans = np.any(pd.isna(r))
        n_failed.append(nans)
    
    n_failed = np.sum(n_failed)
    n_total = len(df)
    
    sel = df["BSASA"] >= 500
    n_500 = np.sum(sel)
    
    n_pdbs = len(set(df['PDBID'].values))

    eprint(f"  N total pairs {pp(n_total)}")
    eprint(f"  N failed rows {pp(n_failed)}")
    eprint(f"  N PDBS {pp(n_pdbs)}")
    eprint(f"N above 500 {pp(n_500)}")

    df = df[sel]
    n_failed = []
    for i, r in df.iterrows():
        nans = np.any(pd.isna(r))
        n_failed.append(nans)
    
    n_failed = np.sum(n_failed)
    n_total = len(df)

    n_pdbs = len(set(df['PDBID'].values))

    eprint(f"  N failed rows {pp(n_failed)}")
    eprint(f"  N PDBS {pp(n_pdbs)}")


@click.command()
@click.option("--concat-only", is_flag=True, default=False, help="Only show BSASA_concat.csv")
def main(concat_only):
    if concat_only:
        files = [Path("significant_cifs/BSASA_concat.csv")]
    else:
        files = [i for i in Path("significant_cifs").iterdir() if (("bsasa" in str(i)) and (i.suffix == ".csv"))]
    for file_path in files:
        df = pd.read_csv(str(file_path))
        
        df.loc[:, "BSASA"] = np.array([float(i) if i != "KeyError" else np.nan for i in df["BSASA"].values])
        summarize(df, file_path.name)

if __name__ == "__main__":
  main()

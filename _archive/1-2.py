"""
Add the sequences
"""

import pandas as pd
import sys
sys.path = sys.path + ["../../../pyext/src/"]
from pathlib import Path
import biotite.database.uniprot

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

df = pd.read_csv("table1.csv") 

assert len(df) == len(set(df['PreyGene']))

while len([i for i in Path("input_sequences").iterdir() if i.suffix==".fasta"]) < 3062: # This condition can terminate
    try:
        for j in df['UniprotId'].values:
            eprint(f"Fetching {j}")
            uniprot = biotite.database.uniprot.fetch(j, "fasta", "input_sequences")
    except TimeoutError:
        eprint(f"Timeout {j}")


try:
    for j in df['UniprotId'].values:
        eprint(f"Fetching {j}")
        uniprot = biotite.database.uniprot.fetch(j, "fasta", "input_sequences")
except TimeoutError:
    eprint(f"Timeout {j}")


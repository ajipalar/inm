import pandas as pd
from pathlib import Path
from itertools import combinations
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf as mmtf
import biotite.sequence.align
import sys
import numpy as np

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

bsasa = [str(i) for i in Path(".").iterdir() if ((i.suffix == (".csv") and ("bsasa_" in i.name)) and ("copy" not in str(i)))]
bsasa = sorted(bsasa)
splits = [i.removesuffix(".csv").split("_") for i in bsasa]
for i in splits:
    assert len(i) == 3, (i)

start_stop = [(int(i[1]), int(i[2])) for i in splits]
start_stop = sorted(start_stop)
k = 0

for start, stop in start_stop:
    assert stop > start, (start, stop)
    assert start == k, (start, stop, k)
    if start < k:
        eprint(f"{start} {stop} {k}")
    k = stop

bsasa = [pd.read_csv(i) for i in bsasa]
bsasa = pd.concat(bsasa)
bsasa.to_csv("BSASA_concat.csv", index=False)


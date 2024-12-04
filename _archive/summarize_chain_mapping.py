import pandas as pd
import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def h(x):
    return '{:,}'.format(x)

d = pd.read_csv("significant_cifs/chain_mapping.csv")

eprint(f"N mappings {len(d)}")
for col in d:
    n = len(d[col].unique())
    eprint(f"N {col}    {h(n)}")


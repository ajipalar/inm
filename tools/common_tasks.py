import pickle as pkl
from pathlib import Path

def pklload(x):
    with open(x, "rb") as f:
        return pkl.load(f)

def lsdir(x):
    if isinstance(x, str):
        x = Path(x)
    return list(x.iterdir())
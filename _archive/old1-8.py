"""
Fetch PDB Files.

How to handle edge cases?
- Large structures cannot be represented by a single PDB file.
- Could you multiple PDB Files to represent a single structure
- Could use the mmcif
- Could use an MMTF
- How many edge cases are there?
"""
import pandas as pd
import sys
from pathlib import Path
import biotite.database.rcsb

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

df = pd.read_csv("hhblits_out/SignificantNonSelfPairs.csv")

pdb_ids = set([i.split("_")[0] for i in df['PDB701'].values])
pdb_dir = Path("pdb_files")

failed = []

for pdb_id in pdb_ids:
    if pdb_id + ".pdb" not in [i.name for i in pdb_dir.iterdir() if i.suffix == ".pdb"]:
        try:
            file_path = biotite.database.rcsb.fetch(pdb_id, "pdb", pdb_dir)
        except biotite.database.RequestError:
            try:
                file_path = biotite.database.rcsb.fetch(pdb_id, "cif", "big_pdb_files")
            except biotite.database.RequestError:
                file_path = str(pdb_id)
                eprint(f"Request Error {file_path}. Skipping.")
                failed.append(file_path)

failed_log = pd.DataFrame({"1-8Failed": failed})

failed_log.to_csv("1-8Failed_log.csv", index=False, header=False)

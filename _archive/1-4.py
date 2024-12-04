import subprocess
import pandas as pd
from pathlib import Path

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

fastas = [i for i in Path("input_sequences").iterdir() if i.suffix == ".fasta"]

for fasta_file in fastas:
    uid = fasta_file.name.removesuffix(".fasta") 
    outf = f"hhblits_out/{uid}.hhr"
    hhblits_cmd = f"hhblits -i input_sequences/{uid}.fasta -o {outf}  -d ~/databases/pdb70/pdb70"
    hhr_files = [i.name for i in Path("hhblits_out").iterdir() if i.suffix == ".hhr"]
    
    if f"{uid}.hhr" not in hhr_files:    
        subprocess.call(hhblits_cmd, shell=True, stdout=open(outf, 'w'))

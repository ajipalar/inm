import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf
import biotite.sequence.align
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

filter_df=pd.read_csv("significant_cifs/filter_chain_mapping_1-9a.csv")

def nuni(x):
    return len(set(x.values))

nuniprot = nuni(filter_df["QueryID"])
npdb = nuni(filter_df["PDBID"])
npairs = len(filter_df)

eprint("Filter Chain Mapping")
eprint(f"N uniprot ids : {nuniprot}")
eprint(f"N PDB ids     : {npdb}")
eprint(f"N pairs       : {npairs}")

df=pd.read_csv("significant_cifs/chain_mapping.csv")

nuniprot = nuni(df["QueryID"])
npdb = nuni(df["PDBID"])
npairs = len(df)

eprint("Chain Mapping")
eprint(f"N uniprot ids : {nuniprot}")
eprint(f"N PDB ids     : {npdb}")
eprint(f"N pairs       : {npairs}")

pdbs = set([i for i in Path("significant_cifs").iterdir() if ".bio.mmtf" in str(i)]) 
eprint(f"N .bio.mmtf's {len(pdbs)}")


fastas = set([i for i in Path("input_sequences").iterdir() if i.suffix == ".fasta"  ])
eprint(f"N .fasta {len(fastas)}")

seqs = []
for path in fastas:
    fasta_file = biotite.sequence.io.fasta.FastaFile.read(str(path))
    header, sequence = list(fasta_file.items())[0]
    seqs.append(sequence)

seqs_under_88 = [i for i in seqs if len(i) < 88] 
eprint(f"N .fasta under 88 {len(seqs_under_88)}")




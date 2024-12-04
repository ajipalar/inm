import pandas as pd
import sys
sys.path = sys.path + ["../../../pyext/src/"]
import cullin_benchmark_test as cb
from pathlib import Path
import biotite.database.uniprot
import biotite.sequence.io.fasta

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

fastas = [i for i in Path("input_sequecnes").iterdir() if i.suffix == ".fasta"]
assert len(fastas) == 3062
df = pd.read_csv("table1.csv")
assert len(df) == 3062

fasta_map = {}
for fasta_path in fastas:
    uid = fasta_path.name.remove_suffix(".fasta")
    fasta_file = biotite.sequence.io.fasta.FastaFile.read(str(fasta_path))
    seq = biotite.sequence.io.fasta.get_sequencen(fasta_file)
    fasta_map[uid] = seq

uniprot_seq = []
for i, row in df.iterrows():
    uniprot_seq.append(fasta_map[row['UniprotId']])

assert len(uniprot_seq) == len(df)
df['UniprotSeq'] = uniprot_seq
df['UniprotSeqLen'] = [len(i) for i in uniprot_seq]
df.to_csv("table1.csv", index=False)


"""
1. Load in the SignificantNonself Pair Prey File
2. For each pdb file
   2a. Load in the query sequences
   2b. For each query sequences
       2aa. Load in the sequence-chain-id pairs for the pdb file
       2ac. Filter to polypeptide sequecnes > 88 amino acids long
       2ad. Do a pairwise sequence alignment to every sequence in the pdb file
       2ae. If the alignment is > 88 aa, evalue < 1e-7, and 30% sequence id then add the sequence to the remap hit table


Exclusion Criteria
- A uniprot sequence is < 88 amino acids (52 sequences) 
- A pdb modeled pdb sequence is < 88 amino acids
"""

import pandas as pd
from pathlib import Path
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf
import biotite.sequence.align
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

sig_alns = pd.read_csv("hhblits_out/SignificantNonSelfPairs.csv")
expected_pdbs = [i.split("_")[0].lower() for i in sig_alns['PDB701'].values]
sig_alns.loc[:, "pdb_id"] = expected_pdbs
expected_pdbs = set(expected_pdbs)
actual_pdbs = set([i.name.removesuffix(".mmtf") for i in Path("significant_cifs").iterdir() if i.suffix == ".mmtf"])
diff = expected_pdbs.symmetric_difference(actual_pdbs)
#assert len(diff) == 0, (len(diff), diff, actual_pdbs, expected_pdbs)

df = sig_alns

blossum62 = biotite.sequence.align.SubstitutionMatrix.std_protein_matrix()
uids = set(df["Query1"].values).union(df["Query2"].values)
sequences = {}

ProtSeq = biotite.sequence.ProteinSequence

non_canonical = {
"MSE": "M",
"AYA": "A",
"5F0": "X",
"SEP": "S",
"CSX": "C",
"MEQ": "Q",
"D2T": "D",
"4D4": "X",
"FME": "M",
"ACE": "X",
"HIC": "H",
"CSO": "C",
"2MR": "R",
"DAL": "A",
"ACB": "D",
"1ZN": "X",
"FGA": "E",
"MAA": "A",
"MLZ": "K",
"UXY": "K",
"IAS": "D",
"NLE": "L",
"3PA": "X",
"GGL": "E",
"4BA": "X",
"NH2": "X",
"5CT": "X",
"M3L": "K",
"BP4": "X",
"TYS": "Y",
"TPO": "T",
"6V1": "X",
"GLZ": "X",
"CSS": "C",
"AME": "M",
"DDE": "X",
"0TD": "D",
"CSD": "A",
"0TD": "D",
"DAR": "R",
"PEA": "X",
"ASJ": "X",
"BAL": "X",
"KCX": "K",
"MYR": "X",
"YCM": "C",
"SAC": "S",
"YCM": "C",
"PCA": "E",
"MLY": "K",
"MS6": "M",
"THC": "T",
"HY3": "P",
"V5N": "H",
"NMM": "R",
"PTR": "Y",
"CAS": "C",
"4HH": "XX",
"PHQ": "X",
"CYG": "XXX",
"AAC": "X",
"0QE": "G",
"0QE": "G",
"PYR": "A", 
"ORN": "R",    # L-Ornithine. Reason - Arg -> Orn
"SAH": "C",    # S-Adenosyl-l-homocysteine. Reason looks like a nucleotide on cycteine
"S12": "S",
"3K4": "X",    # Aziridine
"U6A": "T",
}

def get_sequence(atom_array, chain_id, min_length=88):
    """
    If the chain length is less than min length, an empty chain is returned
    """
    mask = atom_array.chain_id == chain_id
    resids = atom_array.res_id[mask]
    res_names = atom_array.res_name[mask]
    Natoms = len(resids)
    N = max(resids) - min(resids)
    seq = ["X"] * N 
    if len(seq) < min_length:
        return ""
    mapp = {}
    for i in range(Natoms):
        resname = res_names[i]
        if len(resname) == 3:
            if resname in non_canonical:
                resname = non_canonical[resname]
            else:
                resname = ProtSeq.convert_letter_3to1(resname)
        mapp[resids[i]] = resname
    for i in range(N): 
        if i in mapp:
            seq[i] = mapp[i]
    return "".join(seq) 

def get_sequence_dict(atom_array):
    chain_ids = set(atom_array.chain_id)
    return {chain_id: get_sequence(atom_array, chain_id) for chain_id in chain_ids}

for uid in uids:
    fasta_file = biotite.sequence.io.fasta.FastaFile.read(f"input_sequences/{uid}.fasta")
    assert len(fasta_file) == 1, uid
    header, sequence = list(fasta_file.items())[0]
    sequences[uid]=sequence

assert len(uid.keys()) == 3062
bio_assemblies = {}  # pdb_id : atom_array
bio_sequences = {}   # pdb_id : {chain_id : sequence} 
# sequences -> uid : sequence
QueryID = []
ChainID = []
PDBID = []
bt_evalue = []
bt_aln_score = []
bt_psid = []
bt_aln_Q = []
bt_aln_T = []
Q = []
T = []

alpha = biotite.sequence.ProteinSequence.alphabet
for pdb_id in expected_pdbs:
    fpath = f"significant_cifs/{pdb_id}.bio.mmtf"
    eprint(f"reading {fpath}")
    read_f = biotite.structure.io.mmtf.MMTFFile.read(fpath)
    bio_array = biotite.structure.io.mmtf.get_structure(read_f, model=1)
    #bio_assemblies[pdb_id] = bio_array
    #mask = bio_array.hetero == False  # Exclude heteroatoms
    mask = biotite.structure.filter_amino_acids(bio_array)
    bio_array = bio_array[mask]
    seq_dict = get_sequence_dict(bio_array)
    new_dict = {}
    for chain, seq in seq_dict.items():
        seq = seq.replace("U", "C")
        seq = seq.replace("J", "X")
        seq = seq.replace("Z", "X")
        seq = seq.replace("B", "X")
        seq = seq.replace("O", "X")
        
        for letter in seq:
            assert letter in alpha, (letter, chain, pdb_id)
        new_dict[chain] = seq
    seq_dict = new_dict.copy()
        
    bio_sequences[pdb_id] = seq_dict
eprint("pdb sequences mutated")
# bio_assemblies
# bio_sequences
new_dict = {}
uid_sequences = sequences

for uid, seq in uid_sequences.items():
    seq = seq.replace("U", "C")
    seq = seq.replace("J", "X")
    seq = seq.replace("Z", "X")
    seq = seq.replace("B", "X")
    seq = seq.replace("O", "X")
    for letter in seq:
        assert letter in alpha, (letter, uid)
    new_dict[uid] = seq
uid_sequences = new_dict.copy()
eprint("uid sequences mutated")

for pdb_id in expected_pdbs:
    sel = df['pdb_id'] == pdb_id
    subframe = df[sel]
    sub_uids = set(subframe["Query1"].values).union(subframe["Query2"].values)
    for sub_uid in sub_uids:
        sub_uid_seq = uid_sequences[sub_uid]
        if len(sub_uid_seq) >= 88:
            for pdb_chain_id, pdb_seq in bio_sequences[pdb_id].items():
                if len(pdb_seq) >= 88: 
                    eprint(f"Aligning {sub_uid} {pdb_chain_id} in {pdb_id}")
                    # Do a pairwise sequence alignment
                    useq = biotite.sequence.ProteinSequence(sub_uid_seq)
                    pdbseq = biotite.sequence.ProteinSequence(pdb_seq)

                    alignments = biotite.sequence.align.align_optimal(useq, pdbseq, blossum62,
                                    local=False)
                    aln=alignments[0]
                    score = aln.score

                    u1, p2 = aln.get_gapped_sequences()
                    QueryID.append(sub_uid)
                    ChainID.append(pdb_chain_id)
                    PDBID.append(pdb_id)
                    bt_aln_score.append(score)
                    bt_evalue.append(None)

                    percent_seq_id = biotite.sequence.align.get_sequence_identity(aln, mode="not_terminal")
                    bt_psid.append(percent_seq_id)
                    bt_aln_Q.append(u1)
                    bt_aln_T.append(p2)

                    Q.append(sub_uid_seq)
                    T.append(pdb_seq)
                    
out_df = pd.DataFrame({"QueryID": QueryID, "ChainID": ChainID,
        "PDBID": PDBID, "bt_aln_score": bt_aln_score, "bt_aln_evalue": bt_evalue,
        "bt_aln_percent_seq_id": bt_psid, "bt_aln_Q": bt_aln_Q, "bt_aln_T": bt_aln_T, "Q": Q, "T": T})

print(out_df)
out_df.to_csv("significant_cifs/chain_mapping.csv", index=False)

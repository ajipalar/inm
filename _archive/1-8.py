"""
Read in all significant hhblits alignments, read in PDB sequences, do pairwise biotite alignments,
do BSASA calculations, save the chain_pairing, save the BSASA calculation.
"""

import click
import pandas as pd
from pathlib import Path
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf as mmtf
import biotite.sequence.align
import numpy as np
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

production = True# Turn this on to enable checks
eprint(f"PRODUCTION {production}")
# Globals
blossum62 = biotite.sequence.align.SubstitutionMatrix.std_protein_matrix()
ProtSeq = biotite.sequence.ProteinSequence
alpha = biotite.sequence.ProteinSequence.alphabet
min_length = 88

obsolete2current = {"3jaq": "6gsm",
    "6t8j": "8bxx",
    "4fxg": "5jpm",
    "3unr": "4yta",
    "6emd": "6i2d",
    "2f83": "6i58",
    "6ers": "6i2c",
    "5lho": "5lvz",
    "6emb": "6i2a",
    "6fbs": "6fuw",
    "5fl8": "5jcs",
    "5dd2": "5zz0",
    "4iqq": "5noo",
    "4xam": "5jtw",
    "1a2k": "5bxq",
    "7ulm": "8ecg",
    "7s1e": "8g4j",
    "4fxk": "5jpn" }

current2obsolete = {val:key for key,val in obsolete2current.items()}

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
"CME": "C", 
"CAF": "C",
"LLP": "K",
"SCY": "C",
"SCH": "C",
"CMT": "C",
"DSN": "S",
"NEP": "H",
"MCS": "C",
"TPQ": "Y",
"SNN": "X",
"OCS": "C",
"OMT": "M",
"MHO": "M",
"RGP": "E",
"FTR": "W",
"4J4": "C",
}

def get_sequence_dict(atom_array):
    chain_ids = set(atom_array.chain_id)
    return {chain_id: get_sequence(atom_array, chain_id) for chain_id in chain_ids}

global_lost = []
def get_sequence(atom_array, chain_id, min_length=min_length):
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
                try:
                    resname = ProtSeq.convert_letter_3to1(resname)
                except:
                    global_lost.append(resname)
                    resname = "X"
        mapp[resids[i]] = resname
    for i in range(N): 
        if i in mapp:
            seq[i] = mapp[i]
    return "".join(seq) 


def h(x):
    return '{:,}'.format(x)

eprint("Reading bio.mmtf paths")
bio_paths = [i for i in Path("significant_cifs/").iterdir() if "bio.mmtf" in str(i)]
eprint("Reading SigPDB70Chain")
sig_alns = pd.read_csv("hhblits_out/SigPDB70Chain.csv")

# Rename the obsolete pdb_ids
eprint("Rename obsolete pdb ids")
tmp = []
expected_pdbs = []
for i, r in sig_alns.iterrows():
    pdb_id, chain_id = r['PDB70_Chain'].split("_")
    pdb_id = pdb_id.lower()

    if pdb_id in obsolete2current:
        pdb_id = obsolete2current[pdb_id]

    pdb70_chain = pdb_id.upper() + "_" + chain_id 
    expected_pdbs.append(pdb_id)
    tmp.append(pdb70_chain)

sig_alns.loc[:, "PDB70_Chain"] = np.array(tmp)

sig_alns.loc[:, "pdb_id"] = expected_pdbs

uids = set(sig_alns["QueryUID"].values)
uid_sequences = {}
eprint("Reading FASTA sequences")
for uid in uids:
    fasta_file = biotite.sequence.io.fasta.FastaFile.read(f"input_sequences/{uid}.fasta")
    assert len(fasta_file) == 1, uid
    header, sequence = list(fasta_file.items())[0]
    uid_sequences[uid]=sequence

pdb_ids = set(sig_alns['pdb_id'].values)

mmtf_files = set([i.name.removesuffix(".bio.mmtf") for i in bio_paths])

if not production:
    expected_pdbs = mmtf_files 

if production:
    sym_diff = pdb_ids.symmetric_difference(mmtf_files)
    assert len(sym_diff) == 0, len(sym_diff)

# Mutate the uniprot sequences

new_dict = {}
eprint("Mutating U, J, Z, B and O")
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

new_dict = {}

pdb_bio_sequences = {}
# Mutate the PDB Sequences


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


@click.command()
@click.option("--start", required=True, type=int)
@click.option("--stop", required=True, type=int)
def main(start, stop, expected_pdbs=expected_pdbs):
    eprint("Begin main")
    assert start < stop, (start, stop)
    expected_pdbs = sorted(set(expected_pdbs))[start:stop]
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
            if len(seq) >= min_length:
                seq = seq.replace("U", "C")
                seq = seq.replace("J", "X")
                seq = seq.replace("Z", "X")
                seq = seq.replace("B", "X")
                seq = seq.replace("O", "X")
            
                for letter in seq:
                    assert letter in alpha, (letter, chain, pdb_id)
                new_dict[chain] = seq
    
        seq_dict = new_dict.copy()
        pdb_bio_sequences[pdb_id] = seq_dict
    
        eprint("pdb sequences mutated")
        eprint(set(global_lost))

    for pdb_id in expected_pdbs:
        fpath = f"significant_cifs/{pdb_id}.bio.mmtf"
        eprint(f"reading {fpath}")
        read_f = biotite.structure.io.mmtf.MMTFFile.read(fpath)
        bio_array = biotite.structure.io.mmtf.get_structure(read_f, model=1)
        #bio_assemblies[pdb_id] = bio_array
        #mask = bio_array.hetero == False  # Exclude heteroatoms
        mask = biotite.structure.filter_amino_acids(bio_array)
        bio_array = bio_array[mask]
    
        sel = sig_alns['pdb_id'] == pdb_id
        subframe = sig_alns[sel]
        sub_uids = set(subframe["QueryUID"].values)
    
        for sub_uid in sub_uids:
            sub_uid_seq = uid_sequences[sub_uid]
            if len(sub_uid_seq) >= min_length:
                for pdb_chain_id, pdb_seq in pdb_bio_sequences[pdb_id].items():
                    if len(pdb_seq) >= min_length: 
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
    
    out_df.to_csv(f"significant_cifs/chain_mapping_{start}_{stop}.csv", index=False)
    print(out_df)

if __name__ == "__main__":
    main()

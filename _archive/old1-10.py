"""
1. We know that each uid mapped to at least one chain in 
2. Come up with a score cutoff, rank stat, or e-value
3. Match the chains
4. Compute BSA and any other strucutre charactersitcs
   1. Only compute BSASA for chains of interest
"""
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

chain_mapping = pd.read_csv("significant_cifs/chain_mapping.csv")
expected_pdbs = [i.split("_")[0].lower() for i in chain_mapping['PDBID'].values]

columns = ["PDBID", "chain_id_1", "chain_id_2", "sasa_1", "sasa_2", "bsasa"]

bio_assemblies = {}

pdb_ids = []
chainA = []
chainB = []
sasa1 = []
sasa2 = []
sasa12_lst = []

bsasa = []

for pdb_id in expected_pdbs:
    fpath = f"significant_cifs/{pdb_id}.bio.mmtf"
    eprint(f"reading {fpath}")
    read_f = biotite.structure.io.mmtf.MMTFFile.read(fpath)
    bio_array = biotite.structure.io.mmtf.get_structure(read_f, model=1)
    amino_mask  = biotite.structure.filter_amino_acids(bio_array)
    bio_array = bio_array[amino_mask]
    subframe = chain_mapping[chain_mapping["PDBID"] == pdb_id]
    chains_of_interest = set(subframe['ChainID'].values)
    chain_pairs = list(combinations(chains_of_interest, 2))
    
    # Assays direct interaction
    monomer_sasa = {}
    for chain_id in chains_of_interest:
        bio_at_chain=bio_array[bio_array.chain_id == chain_id]
        atom_sasa = biotite.structure.sasa(bio_at_chain)
        assert not np.all(np.isnan(atom_sasa)), (chain_id, pdb_id)
        chain_sasa = biotite.structure.apply_chain_wise(bio_at_chain, atom_sasa, np.sum)
        assert chain_sasa.shape == (1,), (chain_sasa.shape, chain_id, pdb_id)
        assert not np.isnan(chain_sasa), (chain_sasa, chain_id, pdb_id)
        monomer_sasa[chain_id] = chain_sasa
        
    for chain1, chain2 in chain_pairs:
        sel1 = bio_array.chain_id == chain1
        sel2 = bio_array.chain_id == chain2
        sel3 = sel1 | sel2
        bio_at_both = bio_array[sel3]
        atom_sasa = biotite.structure.sasa(bio_at_both)
        

        assert not np.all(np.isnan(atom_sasa)), (pdb_id, chain1, chain2)
        sasa12 = np.sum(atom_sasa)
        assert sasa12.shape == (1,)
        assert not np.isnan(sasa12), (pdb_id, chain1, chain2)

        s1 = monomer_sasa[chain1]
        s2 = monomer_sasa[chain2]

        bsasa12 = s1 + s2 - sasa12
        pdb_ids.append(pdb_id)
        chainA.append(chain1)
        chainB.append(chain2)
        sasa1.append(s1)
        sasa2.append(s2)
        sasa12_lst.append(sasa12)
        bsasa.append(bsasa12)

df = pd.DataFrame({"PDBID": pdb_ids, "chainA": chainA, "chainB": chainB, "bsasa": bsasa, "sasa12": sasa12, "sasa1": sasa1, "sasa2": sasa2})
df.to_csv("significant_cifs/bsasa.csv", index=False)

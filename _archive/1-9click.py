
"""
Output Columns:

1. Read in every bio.mmtf file.
into memory.

2. Compute the BSASA.

3. Save to an csv with columns as follows.

PDBID, Chain1, Chain2, BSASA, SASA1, SASA2
"""

import pandas as pd
from pathlib import Path
from itertools import combinations
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf as mmtf
import biotite.sequence.align
import sys
import click
import numpy as np

skip_lst = ["6u42"]
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

@click.command()
@click.option("--start", type=int)
@click.option("--end", type=int)
def main(start, end):
    p = Path("significant_cifs")
    
    row_lst = []
    bio_mmtfs = [i for i in p.iterdir() if ".bio.mmtf" in str(i)]
    bio_mmtfs = sorted(bio_mmtfs)
    bio_mmtfs = bio_mmtfs[start:end]
    i = 0
    for mmtf_file_path in bio_mmtfs:
        eprint(f"Reading {str(mmtf_file_path)}")
        pdb_id = mmtf_file_path.name[0:4] 
        if pdb_id in skip_lst:
            eprint(f"Skipping {pdb_id}")
            continue
        mmtf_file = mmtf.MMTFFile.read(str(mmtf_file_path))
    
        
        atom_array = mmtf.get_structure(mmtf_file, model=1)
        mask1 = biotite.structure.filter_canonical_amino_acids(atom_array)
        mask2 = atom_array.element != "H"
    
        mask3 = mask1 & mask2 
    
        chain_set = set(atom_array[mask3].chain_id)
        chain_pairs = list(combinations(chain_set, 2))
    
        monomer_sasa_dict = {}
        try:
            for chain in chain_set:
                sel = atom_array.chain_id == chain
                #atom_chain = atom_array[sel]
                
                mask_sel = (mask3) & sel
    
                filtered_arr = atom_array[mask_sel]
                sasa_array = biotite.structure.sasa(filtered_arr)
                if np.any(np.isnan(sasa_array)):
                    eprint(f"NaN in {chain} {pdb_id} SASA")
    
                monomer_sasa_dict[chain] = np.sum(sasa_array) 
    
            for pair in chain_pairs:
                chain1, chain2 = pair
                sasa1 = monomer_sasa_dict[chain1]
                sasa2 = monomer_sasa_dict[chain2]
    
                sel1 = atom_array.chain_id == chain1
                sel2 = atom_array.chain_id == chain2
                sel3 = sel1 | sel2
                
                
                mask_sel3 = (mask3) & sel3
    
                #atom_chains = atom_array[sel3]
                filtered_arr = atom_array[mask_sel3]
                sasa_array = biotite.structure.sasa(filtered_arr)
    
                if np.any(np.isnan(sasa_array)):
                    eprint(f"NaN in {chain1}-{chain2} {pdb_id} SASA")
                
                sasa12 = np.sum(sasa_array)
    
                bsasa = sasa1 + sasa2 - sasa12
                row_lst.append((pdb_id, chain1, chain2, bsasa, sasa12, sasa1, sasa2))
        except KeyError as error:
           eprint(f"ERROR: {pdb_id, error}")
           error = "KeyError"
           chain1 = error
           chain2 = error
           bsasa = error
           sasa12 = error
           sasa1 = error
           sasa2 = error
           row_lst.append((pdb_id, chain1, chain2, bsasa, sasa12, sasa1, sasa2))
    columns = ["PDBID", "Chain1", "Chain2", "BSASA", "SASA12", "SASA1", "SASA2"]
    df = pd.DataFrame(row_lst, columns=columns)
    df.to_csv(f"significant_cifs/bsasa_{start}_{end}.csv", index=False)    

if __name__ == "__main__":
    main()

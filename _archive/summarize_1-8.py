import pandas as pd
from pathlib import Path
import biotite.sequence.io.fasta
import biotite.sequence
import biotite.structure.io.mmtf as mmtf
import biotite.sequence.align
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def h(x):
    return '{:,}'.format(x)

bio_paths = [i for i in Path("significant_cifs/").iterdir() if "bio.mmtf" in str(i)]

bio_data = {}
n_all_chains = 0
n_can_nuc_chains = 0
n_carb_chains = 0
n_can_amino_chains = 0
npdbs = 0
for bio_path in bio_paths:
    pdb_id = bio_path.name.removesuffix(".bio.mmtf")
    bio_file = mmtf.MMTFFile.read(str(bio_path))
    array = mmtf.get_structure(bio_file, model=1)
    n_all_chains += len(set(array.chain_id))

    can_nuc_mask = biotite.structure.filter_canonical_nucleotides(array)
    nuc_array = array[can_nuc_mask]
    n_can_nuc_chains += len(set(nuc_array.chain_id))
    
    can_carb_mask = biotite.structure.filter_carbohydrates(array)
    carb_array = array[can_carb_mask]
    n_carb_chains += len(set(carb_array.chain_id))

    can_amino_mask = biotite.structure.filter_canonical_amino_acids(array)
    can_amino_array = array[can_amino_mask]
    n_can_amino_chains += len(set(can_amino_array.chain_id)) 
    npdbs += 1

eprint(f"N .bio.mmtf                     {h(npdbs)}")
eprint(f"N all chains                    {h(n_all_chains)}")
eprint(f"N canonical nucleotitde chains  {h(n_can_nuc_chains)}")
eprint(f"N carbohydrate chains           {h(n_carb_chains)}")
eprint(f"N canonical amino chains        {h(n_can_amino_chains)}")

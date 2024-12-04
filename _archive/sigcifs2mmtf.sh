#!/bin/bash
conda activate wyndevpy39
for file in $(ls significant_cifs/*.cif); do
    pdb_id=${file:17:-4}
    if [ -f significant_cifs/${pdb_id}.mmtf ]; then
        echo "BASH SKIP ${pdb_id}"
    else
        echo "BASH RUN ${pdb_id}"
        python3 pdb2mmtf.py -i $file -o significant_cifs -from "cif" -to "mmtf" 
    fi
done

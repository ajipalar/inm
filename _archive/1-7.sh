#!/bin/bash
# Fetch the significant PDBs and populate the significant cifs directory

start=${1:-1}
stop=${2:-50000}

for pdb_id in $(cat hhblits_out/SignificantPDB70_PDBIDs.csv | sed -n "$start,${stop}p"); do
    lower_pdb=${pdb_id,,}

    if [[ "$lower_pdb" == "3jaq" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 6GSM"
       lower_pdb="6gsm"
    elif [[ "$lower_pdb" == "6t8j" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 8BXX"
       lower_pdb="8bxx"
    elif [[ "$lower_pdb" == "4fxg" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 5jpm"
       lower_pdb="5jpm"
    elif [[ "$lower_pdb" == "3unr" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 4yta"
       lower_pdb="4yta"
    elif [[ "$lower_pdb" == "6emd" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 6i2d"
       lower_pdb="6i2d"
    elif [[ "$lower_pdb" == "2f83" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 6i58"
       lower_pdb="6i58"
    elif [[ "$lower_pdb" == "6ers" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 6ers"
       lower_pdb="6i2c"
    elif [[ "$lower_pdb" == "5lho" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 5lho"
       lower_pdb="5lvz"
    elif [[ "$lower_pdb" == "6emb" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 6i2a"
       lower_pdb="6i2a"
    elif [[ "$lower_pdb" == "6fbs" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 6fuw"
       lower_pdb="6fuw"
    elif [[ "$lower_pdb" == "5fl8" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 5jcs"
       lower_pdb="5jcs"
    elif [[ "$lower_pdb" == "5dd2" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 5zz0"
       lower_pdb="5zz0"
    elif [[ "$lower_pdb" == "4iqq" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 5noo"
       lower_pdb="5noo"
    elif [[ "$lower_pdb" == "4xam" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 5jtw"
       lower_pdb="5jtw"
    elif [[ "$lower_pdb" == "1a2k" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 5bxq"
       lower_pdb="5bxq"
    elif [[ "$lower_pdb" == "7ulm" ]]; then
       echo "Obsolete PDB ID $lower_pdb"
       echo "Remapping to 8ecg"
       lower_pdb="8ecg"
    fi
       
    middle=${lower_pdb:1:2}
    filename=all_bioassembly_cif/mmCIF/$middle/${lower_pdb}.cif.gz

    if [[ ! -f "$filename" ]]; then
        echo "Error: File does not exist" >&2
        echo "$filename"
        exit 1
    fi

    rsync -v --ignore-existing $filename significant_cifs/

    if [[ ! -f "${lower_pdb}.cif" ]]; then
        gunzip -c significant_cifs/${lower_pdb}.cif.gz > significant_cifs/${lower_pdb}.cif
    else
       echo "File significant_cifs/${lower_pdb}.cif already exists." >&2
    fi
done

for pdbid in $(cat hhblits_out/nonself_pair_pdbs.csv); do
    pdb=${pdbid,,}
    rsync -v --ignore-existing all_bioassembly_cif/mmCIF/"${pdb:1:2}/$pdb.cif.gz" significant_cifs/
    gunzip significant_cifs/$pdb.cif.gz
done

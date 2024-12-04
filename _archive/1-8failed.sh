#!/bin/bash
for pdb_id in $(cat 1-8Failed_log.csv); do
    echo $pdb_id
    wget https://files.rcsb.org/pub/pdb/compatible/pdb_bundle/ol/7old/7old-pdb-bundle.tar.gz big_pdb_files/
done

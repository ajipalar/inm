#!/bin/bash
python3 ../hhsuite_static/scripts/cif2fasta.py cif2fasta.py -i <all_cifs> -o pdb100.fas -c <num_cores> -p pdb_filter.dat

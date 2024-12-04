#!/bin/bash
export PATH=$(pwd)/hhsuite_static/scripts/:$(pwd)/hhsuite_static/bin/:$PATH
which hhblits
cd hhsuite_static/scripts
python3 cif2fasta.py -i ../../significant_cifs -o ../../sig_cifs.fas -c 16 -p ../../significant_cifs/pdb_filter.dat

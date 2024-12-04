#!/bin/bash


#rsync -av --ignore-existing ajikarunia@dt2.wynton.ucsf.edu:~/Projects/benchmark_results/significant_cifs/BSASA_concat.csv significant_cifs/BSASA_concat.csv
rsync -av --ignore-existing ajikarunia@dt2.wynton.ucsf.edu:~/Projects/benchmark_results/significant_cifs/BSASA_reference.csv significant_cifs/BSASA_reference.csv
rsync -av --ignore-existing ajikarunia@dt2.wynton.ucsf.edu:~/Projects/benchmark_results/1-s2.0-S1931312819302537-mmc2.xlsx .
rsync -av --ignore-existing ajikarunia@dt2.wynton.ucsf.edu:~/Projects/benchmark_results/table1.csv .
rsync -av --ignore-existing ajikarunia@dt2.wynton.ucsf.edu:~/Projects/benchmark_results/significant_cifs/chain_mapping.csv significant_cifs/chain_mapping.csv
rsync -av --ignore-existing ajikarunia@dt2.wynton.ucsf.edu:~/Projects/benchmark_results/significant_cifs/chain_mapping_all.csv significant_cifs/chain_mapping_all.csv


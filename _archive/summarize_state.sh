#!/bin/bash

NFASTA=$(ls input_sequences/*.fasta | wc -l)
NHHR=$(ls hhblits_out/*.hhr | wc -l)
NALIGNMENTS=$(cat hhblits_out/PreyPDB70PairAlign.csv | wc -l)
NSIGALI=$(cat hhblits_out/SignificantPreyPDB70PairAlign.csv | wc -l)
NSELFPREYPAIR=$(cat hhblits_out/SignificantSelfPairs.csv | wc -l) 
NNONSELFPREYPAIR=$(cat hhblits_out/SignificantNonSelfPairs.csv | wc -l)

NPDBFILES=$(ls pdb_files/*.pdb | wc -l)
NCIFFILES=$(ls big_pdb_files/*.cif | wc -l)
NRENUPDB=$(ls renumbered_pdb_files/*.pdb | wc -l)
NRENUCIF=$(ls renumbered_big_pdb_files/*.cif | wc -l)
NRENMMTF=$(ls renumbered_mmtf/*.mmtf | wc -l)
NFAILED=$(cat "1-8Failed_log.csv" | wc -l)
NNONSELFBIOCIF=$(ls significant_cifs/*.cif | wc -l) 

NSELFPAIRPDB=$(cat hhblits_out/self_pair_pdbs.csv | wc -l)
NSELFNONPAIRPDB=$(cat hhblits_out/nonself_pair_pdbs.csv | wc -l)
NDIFFMMTF=$(ls significant_cifs/*.mmtf | wc -l)

# Number of chains mapped

NMAPPEDCHAINS=$(cat significant_cifs/filter_chain_mapping_1-9a.csv | wc -l) # Plus one

echo "N-Fasta,$NFASTA"
echo "N-hhr,$NHHR"
echo "N-ali,$NALIGNMENTS"
echo "N-sig-ali,$NSIGALI"
echo "N-PreyMap2Same,$NSELFPREYPAIR"
echo "N-PreyMap2Diff,$NNONSELFPREYPAIR"
echo "N-SamePreyPDBS,$NSELFPAIRPDB"
echo "N-DiffPDBS,$NSELFNONPAIRPDB"
echo "N-SigMMTF,$NDIFFMMTF"
echo "NMappedChains plus one,$NMAPPEDCHAINS"
echo "N-pdb-files,$NPDBFILES"
echo "N-big-files,$NCIFFILES"
echo "N-nonself-bio-cifs,$NNONSELFBIOCIF"
echo "N-repdb,$NRENUPDB"
echo "N-failed-pdb,$NFAILED"
echo "N-renucif,$NRENUCIF"
echo "N-renmmtf,$NRENMMTF"



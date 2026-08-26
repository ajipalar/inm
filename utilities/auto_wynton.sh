#!/bin/bash

MODEL1=mini_model23_n_mock_10k  
MODEL2=mini_model23_n_all_20k

while true; do
	git pull origin main
	qsub qsub_run_tool_high_mem_figs.sh $MODEL1
	qsub qsub_run_tool_high_mem_figs.sh $MODEL2

	tar -czvf ../results/"$MODEL1"_merged/figs300.tar.gz ../results/"$MODEL1"_merged/*300.png
	tar -czvf ../results/"$MODEL2"_merged/figs300.tar.gz ../results/"$MODEL2"_merged/*300.png

	sleep $1 
done

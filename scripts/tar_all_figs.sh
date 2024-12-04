#!/bin/bash

for fname in $(ls ../results)
do
    tar -czvf ../results/"$fname"/figs300.tar.gz  ../results/"$fname"/*300.png
done

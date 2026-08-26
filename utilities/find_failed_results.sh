#!/bin/bash
for i in {$1..$2}
do
l ../results/$3$4_rseed_$i/0_$3_$i.pkl
done

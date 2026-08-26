#!/bin/zsh
conda activate py39

python3 mini_se_run.py &
python3 mini_se_sc_run.py &
python3 mini_se_sr_run.py &
python3 mini_se_sr_sc_run.py &
python3 mini_dev_run.py &

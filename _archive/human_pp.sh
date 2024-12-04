#!/bin/bash
bash summarize_state.sh | awk -F ',' '{printf "%-20s %-20s\n", $1, $2}'

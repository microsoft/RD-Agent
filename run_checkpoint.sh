#!/bin/bash

n=5

for ((i=1; i<=n; i++))
do
    output_path="log_researcher_$i"
    echo "Index: $i | Output Path: $output_path"
    
    python run_checkpoint.py \
        --path log_checkpoint \
        --output_path "$output_path" \
        --n_process 4 \
        --n_loops 2 \
        --max_num 4
    
    wait
done
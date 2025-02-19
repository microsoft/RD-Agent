#!/bin/bash

competition_ids=(
    "aerial-cactus-identification"
    "aptos2019-blindness-detection"
    "denoising-dirty-documents"
    "detecting-insults-in-social-commentary"
    "dog-breed-identification"
    "dogs-vs-cats-redux-kernels-edition"
    "histopathologic-cancer-detection"
    "jigsaw-toxic-comment-classification-challenge"
    "leaf-classification"
    "mlsp-2013-birds"
    "new-york-city-taxi-fare-prediction"
    "nomad2018-predict-transparent-conductors"
    "plant-pathology-2020-fgvc7"
    "random-acts-of-pizza"
    "ranzcr-clip-catheter-line-classification"
    "siim-isic-melanoma-classification"
    "spooky-author-identification"
    "tabular-playground-series-dec-2021"
    "tabular-playground-series-may-2022"
    "text-normalization-challenge-english-language"
    "text-normalization-challenge-russian-language"
    "the-icml-2013-whale-challenge-right-whale-redux"
)

# Maximum number of parallel jobs
MAX_JOBS=3
active_jobs=0

# Ensure the log directory exists
mkdir -p log

for competition_id in "${competition_ids[@]}"; do
    echo "Running for competition: $competition_id"
    # Run the command in the background
    timeout $((3*60*60)) dotenv run -- env LOG_TRACE_PATH="log/${competition_id}_baseline" python rdagent/app/data_science/loop.py --competition "$competition_id" > "log/${competition_id}_baseline.log" 2>&1 &
    
    # Increment the active jobs counter
    active_jobs=$((active_jobs + 1))
    
    # If the number of active jobs reaches the maximum, wait for one to finish
    if [[ $active_jobs -ge $MAX_JOBS ]]; then
        wait -n  # Wait for any single background job to finish
        active_jobs=$((active_jobs - 1))  # Decrement the active jobs counter
    fi
done

# Wait for all remaining background jobs to finish
wait
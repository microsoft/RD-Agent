#!/bin/bash

LOG_FOLDER="./log" # baseline - {competition_id}_baseline
OUTPUT_FOLDER="./log2" # researcher - {competition_id}_researcher

# Create OUTPUT_FOLDER if it doesn't exist
if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

# Loop through directories in LOG_FOLDER
for f in "$LOG_FOLDER"/*; do
    if [ -d "$f" ] && [[ "$f" == *"baseline"* ]]; then
        # Source and destination paths
        src="$f"
        dst="$OUTPUT_FOLDER/$(basename "$f" | sed 's/baseline/researcher/')"

        # Remove destination if it exists and copy source to destination
        if [ -d "$dst" ]; then
            rm -rf "$dst"
        fi
        cp -r "$src" "$dst"

        # Call the Python function to get the first valid loop
        valid_loop=$(python3 -c "from scripts.exp.researcher.utils import get_first_valid_submission; print(get_first_valid_submission('$dst'))")

        # # Set the checkpoint path
        # checkpoint="$dst/__session__/$valid_loop/4_record"
        
        # # Run checkpoint for a single loop
        # echo "Running a single loop for checkpoint $checkpoint"
        # dotenv run -- env LOG_TRACE_PATH="$dst" python rdagent/app/data_science/loop.py --path "$checkpoint" --loop_n 2
    fi
done
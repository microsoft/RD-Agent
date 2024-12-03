#!/bin/bash

# Test directory setup
TEST_DIR="test_run"
mkdir -p "$TEST_DIR/results"
mkdir -p "$TEST_DIR/logs"

# Define relative paths inside the folder RDAgent
ENV_DIR="scripts/exp/ablation/env" # The folder of environments to apply
PYTHON_SCRIPT="rdagent/app/kaggle/loop.py" # The main file for running 

# Run the experiment
echo "Running experiments..."
dotenv run -- ./scripts/exp/tools/run_envs.sh -d "$ENV_DIR" -j 4 -- \
    python "$PYTHON_SCRIPT" \
    --competition "spaceship-titanic" \ 

# Cleanup (optional - comment out if you want to keep results)
# rm -rf "$TEST_DIR"
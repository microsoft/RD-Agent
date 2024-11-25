#!/bin/bash

# Test directory setup
TEST_DIR="test_run"
mkdir -p "$TEST_DIR/results"
mkdir -p "$TEST_DIR/logs"

# Define paths
ENV_DIR="/home/v-xisenwang/RD-Agent/scripts/exp/ablation/env"
PYTHON_SCRIPT="/home/v-xisenwang/RD-Agent/rdagent/app/kaggle/loop.py"

# Run the experiment
echo "Running experiments..."
dotenv run -- ./scripts/exp/tools/run_envs.sh -d "$ENV_DIR" -j 4 -- \
    python "$PYTHON_SCRIPT" \
    --competition "spaceship-titanic" \ 

# Cleanup (optional - comment out if you want to keep results)
# rm -rf "$TEST_DIR"
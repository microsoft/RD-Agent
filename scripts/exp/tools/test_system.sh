#!/bin/bash

# Test directory setup
TEST_DIR="test_run"
mkdir -p "$TEST_DIR/results"
mkdir -p "$TEST_DIR/logs"

# Test 1: Environment loading verification
echo "Testing environment loading..."
./scripts/exp/tools/run_envs.sh -d scripts/exp/ablation/env -j 1 -- env | grep "if_using"

# Test 2: Run actual experiments
echo "Running experiments with different configurations..."
./scripts/exp/tools/run_envs.sh -d scripts/exp/ablation/env -j 4 -- \
    python -m rdagent.app.kaggle.loop \
    --competition "titanic" \
    --result_path "${TEST_DIR}/results/$(basename {} .env)_result.json"

# Test 3: Result collection
echo "Collecting and analyzing results..."
EXP_DIR="$TEST_DIR" python scripts/exp/tools/collect.py

# Display results location
echo "Test results available at: $TEST_DIR"

# Cleanup
rm -rf "$TEST_DIR"
#!/bin/bash

# Test directory setup
TEST_DIR="test_run"
mkdir -p "$TEST_DIR/results"

# Test 1: Environment loading
echo "Testing environment loading..."
./scripts/exp/tools/run_envs.sh -d scripts/exp/ablation/env -j 1 -- env | grep "if_using"

# Test 2: Parallel execution
echo "Testing parallel execution..."
./scripts/exp/tools/run_envs.sh -d scripts/exp/ablation/env -j 4 -- \
    echo "Processing env with RAG setting: $if_using_vector_rag"

# Test 3: Result collection
echo "Testing result collection..."
EXP_DIR="$TEST_DIR" python scripts/exp/tools/collect.py

# Cleanup
rm -rf "$TEST_DIR"
#!/bin/bash
echo "=== Example Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
python "$(dirname "$0")/train.py"

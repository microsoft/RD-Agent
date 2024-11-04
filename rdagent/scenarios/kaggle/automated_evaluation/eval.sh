#!/bin/bash

# Comments
cat << "EOF" > /dev/null
Experiment Setup Types:
1. DS-Agent Mini-Case
2. RD-Agent Basic
3. RD-Agent Pro
4. RD-Agent Max

Each setup has specific configurations for:
- base_model (4o|mini|4o)
- rag_param (No|Simple|Advanced)
- if_MAB (True|False)
- if_feature_selection (True|False)
- if_hypothesis_proposal (True|False)
EOF

# Get current time and script directory
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
current_time=$(date +"%Y%m%d_%H%M%S")
export SCRIPT_DIR
export current_time

# Parse command line arguments
PARALLEL=1
CONF_PATH=./
COMPETITION=""
SETUP_TYPE=""

while getopts ":sc:k:t:" opt; do
    case $opt in
        s)
        echo "Disable parallel running (run experiments serially)" >&2
        PARALLEL=0
        ;;
        c)
        echo "Setting conf path $OPTARG" >&2
        CONF_PATH=$OPTARG
        ;;
        k)
        echo "Setting Kaggle competition $OPTARG" >&2
        COMPETITION=$OPTARG
        ;;
        t)
        echo "Setting setup type $OPTARG" >&2
        SETUP_TYPE=$OPTARG
        ;;
        \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1
        ;;
    esac
done

# Validate required parameters
if [ -z "$COMPETITION" ] || [ -z "$SETUP_TYPE" ]; then
    echo "Error: Competition (-k) and setup type (-t) are required"
    exit 1
fi

# Create necessary directories
mkdir -p "${SCRIPT_DIR}/results/${current_time}"
mkdir -p "${SCRIPT_DIR}/logs/${current_time}"

# Configure experiment based on setup type
configure_experiment() {
    local setup=$1
    case $setup in
        "mini-case")
            echo "if_using_vector_rag=True" > "${SCRIPT_DIR}/override.env"
            echo "if_using_graph_rag=False" >> "${SCRIPT_DIR}/override.env"
            echo "if_action_choosing_based_on_UCB=True" >> "${SCRIPT_DIR}/override.env"
            echo "model_feature_selection_coder=True" >> "${SCRIPT_DIR}/override.env"
            echo "hypothesis_gen=False" >> "${SCRIPT_DIR}/override.env"
            ;;
        "basic")
            echo "if_using_vector_rag=False" > "${SCRIPT_DIR}/override.env"
            echo "if_using_graph_rag=False" >> "${SCRIPT_DIR}/override.env"
            echo "if_action_choosing_based_on_UCB=False" >> "${SCRIPT_DIR}/override.env"
            echo "model_feature_selection_coder=True" >> "${SCRIPT_DIR}/override.env"
            echo "hypothesis_gen=True" >> "${SCRIPT_DIR}/override.env"
            ;;
        "pro")
            echo "if_using_vector_rag=True" > "${SCRIPT_DIR}/override.env"
            echo "if_using_graph_rag=False" >> "${SCRIPT_DIR}/override.env"
            echo "if_action_choosing_based_on_UCB=True" >> "${SCRIPT_DIR}/override.env"
            echo "model_feature_selection_coder=True" >> "${SCRIPT_DIR}/override.env"
            echo "hypothesis_gen=True" >> "${SCRIPT_DIR}/override.env"
            ;;
        "max")
            echo "if_using_vector_rag=True" > "${SCRIPT_DIR}/override.env"
            echo "if_using_graph_rag=True" >> "${SCRIPT_DIR}/override.env"
            echo "if_action_choosing_based_on_UCB=True" >> "${SCRIPT_DIR}/override.env"
            echo "model_feature_selection_coder=True" >> "${SCRIPT_DIR}/override.env"
            echo "hypothesis_gen=True" >> "${SCRIPT_DIR}/override.env"
            ;;
    esac
}

# Execute experiment
run_experiment() {
    local setup_type=$1
    local competition=$2
    
    configure_experiment "$setup_type"
    
    # Run the main experiment loop
    python -m rdagent.app.kaggle.loop \
        --competition "$competition" \
        --setup "$setup_type" \
        --result_path "${SCRIPT_DIR}/results/${current_time}/result.json" \
        >> "${SCRIPT_DIR}/logs/${current_time}/experiment.log" 2>&1
    
    # Store experiment setup and results
    cat > "${SCRIPT_DIR}/results/${current_time}/experiment_info.json" << EOF
{
    "setup": {
        "competition": "$competition",
        "setup_type": "$setup_type",
        "timestamp": "$current_time"
    },
    "results": $(cat "${SCRIPT_DIR}/results/${current_time}/result.json")
}
EOF
}

# Run the experiment
run_experiment "$SETUP_TYPE" "$COMPETITION"

# Cleanup
trap 'rm -f "${SCRIPT_DIR}/override.env"' EXIT

echo "Experiment completed. Results are stored in ${SCRIPT_DIR}/results/${current_time}"
 
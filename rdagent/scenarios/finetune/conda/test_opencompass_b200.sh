#!/bin/bash
# test_opencompass_b200.sh - Test OpenCompass compatibility on B200 GPU
# Usage: ./test_opencompass_b200.sh [MODEL_PATH]

set -e

ENV_NAME="opencompass_test"
MODEL_PATH="${1:-Qwen/Qwen2.5-0.5B-Instruct}"  # Use smallest model for fast testing
WORK_DIR="./opencompass_test_results"

echo "========================================"
echo "OpenCompass B200 Compatibility Test"
echo "========================================"
echo "Environment: $ENV_NAME"
echo "Model: $MODEL_PATH"
echo "Work Dir: $WORK_DIR"
echo "========================================"

# Step 1: Create conda environment
echo ""
echo "=== Step 1: Creating conda environment ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists, removing..."
    conda env remove -n $ENV_NAME -y
fi
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Step 2: Install dependencies
echo ""
echo "=== Step 2: Installing dependencies ==="
echo "Installing PyTorch 2.9.0 with CUDA 12.8..."
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0

echo "Installing vLLM..."
pip install "vllm>=0.12.0"

echo "Installing OpenCompass..."
pip install "opencompass @ git+https://github.com/Jensen246/opencompass.git"

echo "Installing math evaluation dependencies..."
pip install math_verify latex2sympy2_extended

# Step 3: Verify PyTorch CUDA
echo ""
echo "=== Step 3: Verifying PyTorch CUDA support ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
    # Verify B200 architecture
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cap[0]}.{cap[1]}')
    if cap[0] >= 10:
        print('✅ B200 (sm_100+) architecture detected!')
    else:
        print('⚠️ Not a B200 GPU')
else:
    print('❌ CUDA not available!')
    exit(1)
"

# Step 4: Verify vLLM
echo ""
echo "=== Step 4: Verifying vLLM ==="
python -c "
import vllm
print(f'vLLM version: {vllm.__version__}')
print('✅ vLLM imported successfully')
"

# Step 5: Generate OpenCompass config
echo ""
echo "=== Step 5: Generating OpenCompass config ==="
mkdir -p $WORK_DIR

cat > config.py << EOF
from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import *

datasets = sum([v for k, v in locals().items() if (k == 'datasets' or k.endswith('_datasets')) and isinstance(v, list)], [])

# Limit samples for faster testing
for ds in datasets:
    if 'reader_cfg' not in ds:
        ds['reader_cfg'] = {}
    ds['reader_cfg']['test_range'] = '[0:5]'  # Only test 5 samples

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='test-model',
        path='$MODEL_PATH',
        model_kwargs=dict(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            dtype='bfloat16',
            max_model_len=4096,
        ),
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=8,
        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        ),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = '$WORK_DIR'
EOF

echo "Config generated: config.py"

# Step 6: Run OpenCompass
echo ""
echo "=== Step 6: Running OpenCompass evaluation ==="
opencompass config.py --work-dir $WORK_DIR

# Done
echo ""
echo "========================================"
echo "✅ Test completed successfully!"
echo "Results saved to: $WORK_DIR"
echo "========================================"

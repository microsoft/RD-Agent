#!/bin/bash
# run_opencompass_test.sh - Quick test for OpenCompass (env already installed)
# Usage: ./run_opencompass_test.sh [MODEL_PATH]

MODEL_PATH="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
WORK_DIR="./opencompass_test_results"

echo "Model: $MODEL_PATH"
echo "Work Dir: $WORK_DIR"

mkdir -p $WORK_DIR

cat > config.py << EOF
from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import *

datasets = sum([v for k, v in locals().items() if (k == 'datasets' or k.endswith('_datasets')) and isinstance(v, list)], [])

for ds in datasets:
    if 'reader_cfg' not in ds:
        ds['reader_cfg'] = {}
    ds['reader_cfg']['test_range'] = '[0:5]'

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
        generation_kwargs=dict(temperature=0.0, top_p=1.0, top_k=1),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

work_dir = '$WORK_DIR'
EOF

opencompass config.py --work-dir $WORK_DIR

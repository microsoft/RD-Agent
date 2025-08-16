"""
Prompt templates specifically for LLM fine-tuning
"""

from pathlib import Path


def get_data_processing_prompt(dataset: str, runtime_info: str, data_samples: str) -> str:
    """Get prompt for data processing task"""
    return f"""
# Data Processing Task

## Objective
Convert dataset `{dataset}` to LLaMA-Factory compatible format and save to appropriate location.

## Current Runtime Environment
```
{runtime_info}
```

## Dataset Samples
Here are the first few samples from the dataset:
```json
{data_samples}
```

## LLaMA-Factory Data Format Requirements

### Alpaca Format (Recommended)
```json
[
  {{
    "instruction": "User instruction",
    "input": "Input content (optional)", 
    "output": "Expected output"
  }}
]
```

### ShareGPT Format
```json
[
  {{
    "conversations": [
      {{"from": "human", "value": "User message"}},
      {{"from": "gpt", "value": "Assistant reply"}}
    ]
  }}
]
```

## Task Requirements

1. **Analyze Original Data Format**: Identify the field structure of the dataset
2. **Choose Appropriate Format**: Select Alpaca or ShareGPT format based on data characteristics
3. **Data Conversion**: Write Python code for format conversion
4. **Save Data**: Save processed data as `/workspace/llm_finetune/data/processed_dataset.json`
5. **Create Configuration**: Create dataset_info.json configuration file
6. **Data Statistics**: Output statistics before and after processing

## Output Files
- `/workspace/llm_finetune/data/processed_dataset.json`: Processed training data
- `/workspace/llm_finetune/data/dataset_info.json`: Dataset configuration file
- Output data processing statistics to console

Please write a complete Python script to accomplish the above tasks.
"""


def get_finetuning_prompt(model: str, dataset: str, runtime_info: str, llamafactory_guide: str) -> str:
    """Get prompt for fine-tuning task"""
    return f"""
# LLM Fine-tuning Task

## Objective
Use LLaMA-Factory framework to fine-tune model `{model}` on dataset `{dataset}`.

## Current Runtime Environment
```
{runtime_info}
```

## LLaMA-Factory Usage Guide
{llamafactory_guide}

## Available Resources
- Processed dataset: `/workspace/llm_finetune/data/processed_dataset.json`
- Dataset configuration: `/workspace/llm_finetune/data/dataset_info.json`
- Base model: `{model}`
- Output directory: `/workspace/llm_finetune/output/`

## Fine-tuning Configuration Suggestions

### Basic Parameters
- model_name: {model}
- dataset: processed_dataset
- output_dir: /workspace/llm_finetune/output
- num_train_epochs: 3
- learning_rate: 5e-5
- per_device_train_batch_size: 2
- gradient_accumulation_steps: 4
- warmup_steps: 100
- logging_steps: 10
- save_steps: 500

### LoRA Parameters (Recommended for fast fine-tuning)
- lora_r: 16
- lora_alpha: 32
- lora_dropout: 0.1
- lora_target: q_proj,v_proj

## Task Requirements

1. **Environment Check**: Verify LLaMA-Factory installation and GPU availability
2. **Parameter Configuration**: Set appropriate training parameters
3. **Start Training**: Use llamafactory-cli for fine-tuning
4. **Monitor Training**: Output training progress and metrics
5. **Save Model**: Ensure fine-tuned model is properly saved
6. **Test Inference**: Simple test of the fine-tuned model

## Output Requirements
- Fine-tuned model saved in `/workspace/llm_finetune/output/`
- Training logs and metrics output to console
- Output model path and basic information after completion

Please write a complete Python script to accomplish the above fine-tuning task. Use llamafactory-cli command-line tool for training.
"""


def get_llamafactory_guide() -> str:
    """Get LLaMA-Factory usage guide"""
    return """
## LLaMA-Factory CLI Usage Guide

### Basic Command Format
```bash
llamafactory-cli train --config_file config.yaml
```

### Main Parameter Descriptions

#### Model Parameters
- `model_name`: Model name or path
- `model_revision`: Model version (optional)
- `quantization_bit`: Quantization bits (4, 8, 16)
- `rope_scaling`: RoPE scaling method

#### Data Parameters  
- `dataset`: Dataset name (defined in dataset_info.json)
- `template`: Conversation template (e.g., default, alpaca, chatgl)
- `max_source_length`: Maximum input sequence length
- `max_target_length`: Maximum output sequence length

#### Training Parameters
- `output_dir`: Output directory
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation steps
- `warmup_steps`: Warmup steps
- `logging_steps`: Logging interval
- `save_steps`: Model save interval

#### LoRA Parameters
- `finetuning_type`: lora
- `lora_rank`: LoRA rank
- `lora_alpha`: LoRA alpha parameter
- `lora_dropout`: LoRA dropout rate
- `lora_target`: Target modules (e.g., q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)

#### Example Configuration File config.yaml
```yaml
model_name: Qwen/Qwen2.5-1.5B-Instruct
dataset: processed_dataset
template: default
finetuning_type: lora
output_dir: /workspace/llm_finetune/output
num_train_epochs: 3.0
learning_rate: 5.0e-5
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
lr_scheduler_type: cosine
warmup_steps: 100
fp16: true
logging_steps: 10
save_steps: 500
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target: q_proj,v_proj
```
"""

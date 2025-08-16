# Simplified LLM Fine-tuning Pipeline Guide

## 📋 Overview

This guide introduces how to use the simplified LLM fine-tuning pipeline, which consists of two steps:
1. **Data Processing**: Convert raw dataset to LLaMA-Factory format
2. **Model Fine-tuning**: Use LLaMA-Factory for fine-tuning in Docker environment

## 🔧 Environment Setup

### 1. Set Environment Variables

Add the following configuration to your `.env` file:

```bash
# LLM fine-tuning related configuration
FT_FILE_PATH=/path/to/your/finetune/workspace
DS_CODER_COSTEER_ENV_TYPE=docker

# API configuration (if needed)
OPENAI_API_KEY=your_api_key
CHAT_MODEL=gpt-4o
```

### 2. Create Working Directory

```bash
# Create fine-tuning working directory
mkdir -p /path/to/your/finetune/workspace/{dataset,model,output,prev_model}
export FT_FILE_PATH=/path/to/your/finetune/workspace
```

### 3. Install Dependencies

```bash
# Ensure Docker is installed and accessible
docker --version

# Install Python dependencies
pip install -e .
```

## 🚀 Usage

### Basic Command

```bash
dotenv run -- python rdagent/app/finetune/llm/loop.py \
  --dataset shibing624/alpaca-zh \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --simple true
```

### Parameter Description

- `--dataset`: Dataset name (e.g., `shibing624/alpaca-zh`)
- `--model`: Base model name (e.g., `Qwen/Qwen2.5-1.5B-Instruct`)
- `--simple`: Whether to use simplified version (recommended to set to `true`)

## 📂 Directory Structure

```
FT_FILE_PATH/
├── dataset/           # Raw datasets
│   └── shibing624/
│       └── alpaca-zh/
├── model/            # Base models
│   └── Qwen/
│       └── Qwen2.5-1.5B-Instruct/
├── output/           # Fine-tuning output
└── prev_model/       # Previous models (optional)
```

After running, it will also create in current directory:
```
./llm_finetune_output/   # Fine-tuning results output (visible outside Docker)
```

## 🔄 Execution Process

### Step 1: Data Processing
- Agent analyzes raw dataset format
- Automatically converts to LLaMA-Factory supported Alpaca or ShareGPT format
- Generates dataset configuration file
- Output files saved in Docker container at `/workspace/llm_finetune/data/`

### Step 2: Model Fine-tuning
- Agent gets runtime environment information
- Configures LoRA fine-tuning parameters
- Uses `llamafactory-cli` for training
- Fine-tuning results saved in `/workspace/llm_finetune/output/` (mapped to local `./llm_finetune_output/`)

## 🎯 Features

### Simplified Design
- ✅ No complex iterative loops
- ✅ No trace mechanism
- ✅ Step-by-step execution, easy to debug
- ✅ One-time completion, no manual intervention needed

### Docker Environment
- ✅ Isolated fine-tuning environment
- ✅ Pre-installed LLaMA-Factory and dependencies
- ✅ Data and results visible outside container
- ✅ GPU acceleration support

### Intelligent Code Generation
- ✅ Automatically analyzes dataset format
- ✅ Generates data processing scripts
- ✅ Configures fine-tuning parameters
- ✅ Integrates LLaMA-Factory best practices

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```bash
   # Ensure necessary environment variables are set
   echo $FT_FILE_PATH
   echo $DS_CODER_COSTEER_ENV_TYPE
   ```

2. **Docker Permission Issues**
   ```bash
   # Ensure current user can run Docker
   docker run hello-world
   ```

3. **Dataset Download Failure**
   ```bash
   # Manually download dataset to specified directory
   mkdir -p $FT_FILE_PATH/dataset/your-dataset-name
   # Place data files in that directory
   ```

4. **Out of Memory**
   - Adjust Docker memory limit (default 48GB)
   - Use smaller models for testing

### Debug Mode

View detailed logs:
```bash
export RDAGENT_LOG_LEVEL=DEBUG
dotenv run -- python rdagent/app/finetune/llm/loop.py ...
```

Enter Docker container for debugging:
```bash
# 1. Start container
dotenv run -- python -m rdagent.app.utils.ws_ft dataset model "sleep 3600"

# 2. Enter container
docker exec -it $(docker ps --filter 'status=running' -l --format '{{.Names}}') bash

# 3. Debug inside container
cd /workspace/llm_finetune
ls -la data/ output/
```

## 📝 Examples

### Complete Example

```bash
# 1. Set environment
export FT_FILE_PATH=/home/user/llm_finetune_workspace
export DS_CODER_COSTEER_ENV_TYPE=docker

# 2. Create directory
mkdir -p $FT_FILE_PATH

# 3. Run fine-tuning
dotenv run -- python rdagent/app/finetune/llm/loop.py \
  --dataset shibing624/alpaca-zh \
  --model Qwen/Qwen2.5-1.5B-Instruct

# 4. Check results
ls -la ./llm_finetune_output/
```

### Using Local Dataset

```bash
# 1. Prepare dataset
mkdir -p $FT_FILE_PATH/dataset/my-custom-dataset
cp your_data.json $FT_FILE_PATH/dataset/my-custom-dataset/

# 2. Run fine-tuning
dotenv run -- python rdagent/app/finetune/llm/loop.py \
  --dataset my-custom-dataset \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

## 🎉 Output After Completion

After fine-tuning is complete, you will find results at:
- `./llm_finetune_output/`: Fine-tuned model and training logs
- Docker container logs with training process and metrics
- Detailed output from data processing and fine-tuning

Enjoy your LLM fine-tuning journey! 🚀
# RD-Agent API Configuration Guide

## 📖 Background

RD-Agent now supports configuring different API keys and endpoints for Chat and Embedding models. This is particularly useful in the following scenarios:

- **Mixed Deployment**: Chat using mentor's LiteLLM deployment, Embedding using other services
- **Cost Optimization**: Chat using high-performance models, Embedding using more economical services
- **Availability Assurance**: Quick switch to backup services when one service is unavailable
- **Service Isolation**: Different functionalities using different API quotas and limits

## 🚀 Quick Start

### Method 1: Universal Configuration (Recommended for Beginners)

If your Chat and Embedding use the same API service:

```bash
# Set universal API key
export OPENAI_API_KEY="sk-your-api-key-here"
export BACKEND="rdagent.oai.backend.LiteLLMAPIBackend"
export CHAT_MODEL="gpt-4o"
export EMBEDDING_MODEL="text-embedding-3-small"
# Optional: Custom endpoint
export OPENAI_BASE_URL="https://your-endpoint.com/v1"
```

### Method 2: Separate Configuration (Recommended for Advanced Users)

If you need Chat and Embedding to use different API services:

```bash
# Chat configuration (using LiteLLM)
export CHAT_OPENAI_API_KEY="sk-chat-key"
export CHAT_OPENAI_BASE_URL="https://xxx-litellm.com/v1"
export CHAT_MODEL='gpt-4o'

# Embedding configuration (using other service)
# Use siliconflow as example, pay attention to the **openai/** prefix
export EMBEDDING_OPENAI_API_KEY="sk-embedding-service-key"
export EMBEDDING_OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
export EMBEDDING_MODEL="openai/BAAI/bge-m3"

# Backend configuration
export BACKEND="rdagent.oai.backend.LiteLLMAPIBackend"
```

## 🛠️ Configuration Methods

### Method 1: Environment Variables

```bash
# Set environment variables directly
export CHAT_OPENAI_API_KEY="sk-your-chat-key"
export EMBEDDING_OPENAI_API_KEY="sk-your-embedding-key"
export BACKEND="rdagent.oai.backend.LiteLLMAPIBackend"
...
```

### Method 2: Configuration File (Recommended)

Create a `.env` file:

```bash
# .env file content
BACKEND=rdagent.oai.backend.LiteLLMAPIBackend

# Chat configuration
CHAT_MODEL=gpt-4o
CHAT_OPENAI_API_KEY=sk-your-chat-key
CHAT_OPENAI_BASE_URL=https://your-chat-endpoint.com/v1
CHAT_TEMPERATURE=0.7

# Embedding configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_OPENAI_API_KEY=sk-your-embedding-key
EMBEDDING_OPENAI_BASE_URL=https://your-embedding-endpoint.com/v1
```

Then load the configuration:

```bash
export PYTHONPATH=.
dotenv run -- python test_separate_configs.py
```

### Method 3: Code Configuration

```python
import os

# Set environment variables in code
os.environ["CHAT_OPENAI_API_KEY"] = "sk-your-chat-key"
os.environ["EMBEDDING_OPENAI_API_KEY"] = "sk-your-embedding-key"
os.environ["BACKEND"] = "rdagent.oai.backend.LiteLLMAPIBackend"
...
```

## 🧪 Testing Configuration

### Using the Provided Test Script

We provide a `test_separate_configs.py` test script that can:

- ✅ **Automatically detect configuration mode** (Universal vs Separate)
- ✅ **Test Chat and Embedding functionality**
- ✅ **Analyze configuration source and validity**
- ✅ **Provide configuration recommendations**

```bash
# Run the test script
python test_separate_configs.py
```

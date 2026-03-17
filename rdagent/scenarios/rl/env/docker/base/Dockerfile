# Base 镜像：PyTorch 2.9.1 + TRL + transformers（训练+评测通用）
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# LLM post-training 库（trl 会自动安装兼容的 transformers、accelerate、datasets）
# Also include `litellm` for AutoRL-Bench evaluation adapters (e.g. GSM8K).
# 注意：transformers 4.57.x 解决 tokenizer save_pretrained 与 vLLM 的兼容性问题
# transformers 5.0 移除了 Qwen2TokenizerFast，导致保存格式不兼容
RUN pip install --no-cache-dir trl==0.27.0 peft verl==0.7.0 litellm>=1.73 "transformers>=4.50,<5.0"

# 默认入口
CMD ["bash"]

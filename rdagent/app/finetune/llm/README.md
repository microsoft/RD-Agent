# LLM Fine-tuning

Fine-tune Large Language Models using LLaMA-Factory.

## Installation

From project root directory:

```bash
make install-llama-factory
```

This will:
- Clone LLaMA-Factory to `git_ignore_folder/LLaMA-Factory`
- Install with `[torch,metrics]` extras
- Use `--no-build-isolation` for CUDA compatibility

## Requirements

- Python 3.10+
- CUDA GPU (recommended)
- PyTorch with CUDA support

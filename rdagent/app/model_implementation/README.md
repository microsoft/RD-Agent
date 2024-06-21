
# Preparation

## Install Pytorch
CPU CUDA will be enough for verify the implementation

Please install pytorch based on your system.
Here is an example on my system
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install torch_geometric

```

# Tasks

## Task Extraction
From paper to task.
```bash
python rdagent/app/model_implementation/task_extraction.py
# It may based on rdagent/document_reader/document_reader.py
```

## Complete workflow
From paper to implementation
``` bash
# Similar to
# rdagent/app/factor_extraction_and_implementation/factor_extract_and_implement.py
```

## Paper benchmark
```bash
python rdagent/app/model_implementation/eval.py

TODO:
- Is evaluation reasonable
```

## Evolving

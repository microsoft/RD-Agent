# Tasks

## Task Extraction
From paper to task.
```bash
# python rdagent/app/model_implementation/task_extraction.py
# It may based on rdagent/document_reader/document_reader.py
python rdagent/components/task_implementation/model_implementation/task_extraction.py ./PaperImpBench/raw_paper/
```

## Complete workflow
From paper to implementation
``` bash
# Similar to
# rdagent/app/factor_extraction_and_implementation/factor_extract_and_implement.py
```

## Paper benchmark
```bash
# TODO: it does not work well now.
python rdagent/app/model_implementation/eval.py
```

TODO:
- Create reasonable benchmark
  - with uniform input
  - manually create task
- Create reasonable evaluation metrics

## Evolving

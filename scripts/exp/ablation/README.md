# Introduction

This document outlines the environment configurations for the ablation studies. Each environment file corresponds to a specific experimental case, with some cases currently unavailable for implementation.

| Name      | .env         | Description                               | Available? |
|-----------|--------------|-------------------------------------------|------------|
| basic | basic.env | Standard case of RDAgent                         | Yes       | 
| minicase  | minicase.env | Enables minicase and DS-Agent             | Yes       |
| pro       | pro.env     | Standard case with vector RAG             | Yes        |
| max       | max.env     | Enables all features                      | No         |

## Notes

- Each `.env` file represents a distinct case for experimentation. Future implementations will include the unavailable cases.
- There is potential for integrating `CHAT_MODEL` in the future to facilitate comparisons between different models in experiments.

## Common Environment Variables

| Variable Name                     | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `MINICASE`                       | Set to `True` to enable the previous implementation of DS-Agent.           |
| `IF_USING_MLE_DATA`              | Set to `True` to use MLE benchmark data; requires `KG_LOCAL_DATA_PATH=/data/userdata/share/mle_kaggle`. |
| `KG_IF_USING_VECTOR_RAG`         | Set to `True` to enable vector RAG.                                       |
| `KG_IF_USING_GRAPH_RAG`          | Set to `False` to disable graph RAG.                                      |
| `KG_IF_ACTION_CHOOSING_BASED_ON_UCB` | Set to `True` to enable action selection based on UCB.                |

## Future Work

- Implement additional environment configurations as needed.
- Explore the integration of different models for comparative analysis in ablation studies.









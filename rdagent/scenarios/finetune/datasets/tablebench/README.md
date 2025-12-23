---
language:
- en
size_categories:
- 1K<n<10K
license: mit
configs:
- config_name: test
  data_files:
  - split: test
    path: data/test-*
  splits:
  - name: test
    num_examples: 886
- config_name: train
  data_files:
  - split: train
    path: data/train-*
  splits:
  - name: train
    num_examples: ~10K
---

# TableBench: Table Question Answering Dataset

Dataset for TableBench: A Comprehensive and Complex Benchmark for Table Question Answering.

> TableBench is a **comprehensive** and **complex** benchmark designed to evaluate Table Question Answering (TableQA) capabilities, covering **18 question categories** across **4 major categories** with **886** carefully curated test cases. 

## Overview

The **TableBench dataset** consists of two main components:

1. **TableBench (Test)**: 886 high-quality test cases for evaluation across 4 major reasoning categories
2. **TableInstruct (Train)**: Large-scale training dataset with diverse table QA examples

TableBench substantially pushes the boundaries of large language models in complex TableQA scenarios, aligning closely with the "Reasoning Complexity of Questions" dimension in real-world Table QA applications.

### Task Categories

The benchmark covers **4 major categories** with **18 sub-tasks**:

1. **Fact Checking**: Verify factual statements against table data
   - Simple fact verification, cross-table validation, temporal consistency

2. **Numerical Reasoning**: Mathematical computations and comparisons
   - Arithmetic operations, aggregations, comparative analysis

3. **Data Analysis**: Complex analytical reasoning
   - Impact analysis, correlation analysis, trend forecasting, statistical analysis

4. **Visualization**: Chart generation and interpretation
   - Bar charts, line charts, pie charts, scatter plots

### Data Sources

**Test Data (TableBench)**:
- Repository: [Multilingual-Multimodal-NLP/TableBench](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench)
- 886 carefully curated and verified test cases
- Enhanced version released April 2025 with error corrections

**Train Data (TableInstruct)**:
- Repository: [Multilingual-Multimodal-NLP/TableInstruct](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableInstruct)
- Large-scale instruction tuning dataset for table QA
- Diverse question types and reasoning patterns

### Data Fields

The dataset contains the following key fields:

- `question`: The table question or task description
- `table`: The table data (various formats: CSV, JSON, markdown)
- `answer`: The ground truth answer or expected output
- `category`: Major category (Fact Checking, Numerical Reasoning, Data Analysis, Visualization)
- `subcategory`: Specific sub-task type
- `reasoning_steps`: Optional chain-of-thought reasoning (for training data)

### Evaluation Metrics

Different metrics are used based on task type:

| Task Type | Metric | Description |
|-----------|--------|-------------|
| Fact Checking | Exact Match (EM) | Exact match of predicted statement |
| Numerical Reasoning | Exact Match (EM) | Correctness of numerical outputs |
| Impact Analysis | Exact Match (EM) | Precise match of influential factors |
| Correlation/Trend/Stats | EM_with_error_10 | ±10% numerical margin of error |
| Other Data Analysis | ROUGE-L | For open-ended textual responses |
| Visualization | Pass@1 | Correct chart generated on first attempt |

## CoT Quality Assessment

**IMPORTANT**: Consider enhancing reasoning chains during training preparation.

| Dimension | Value |
|-----------|-------|
| baseline_quality | medium-high |
| task_type | table_qa |
| polish_difficulty | medium |

**Baseline**: Training data (TableInstruct) contains reasoning examples, but test data focuses on final answers. For complex reasoning tasks (Data Analysis, Numerical Reasoning), generating detailed step-by-step CoT can significantly improve model performance.

**Recommendation**: For Data Analysis and Numerical Reasoning categories, expand reasoning chains to include:
- Table understanding and schema identification
- Step-by-step computation or logical reasoning
- Intermediate results and verification
- Final answer with confidence indicators

## Example

### Fact Checking
```json
{
  "question": "Based on the table, verify if the statement is true: 'Company A had higher revenue than Company B in Q4 2023'",
  "table": "| Company | Q4 2023 Revenue |\n|---------|----------------|\n| A       | $2.5M          |\n| B       | $3.1M          |",
  "answer": "False",
  "category": "Fact Checking",
  "subcategory": "simple_fact_verification"
}
```

### Numerical Reasoning
```json
{
  "question": "What is the total revenue across all quarters for Product X?",
  "table": "| Quarter | Product X Revenue |\n|---------|------------------|\n| Q1      | 150              |\n| Q2      | 200              |\n| Q3      | 175              |\n| Q4      | 225              |",
  "answer": "750",
  "category": "Numerical Reasoning",
  "subcategory": "aggregation"
}
```

### Data Analysis
```json
{
  "question": "Analyze the correlation between marketing spend and sales growth. What is the correlation coefficient?",
  "table": "| Month | Marketing ($K) | Sales Growth (%) |\n|-------|----------------|------------------|\n| Jan   | 50             | 12               |\n| Feb   | 75             | 18               |\n| Mar   | 60             | 15               |",
  "answer": "0.95",
  "category": "Data Analysis",
  "subcategory": "correlation_analysis"
}
```


## License

This dataset is released under the MIT License.


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

The TableInstruct dataset contains the following fields:

- `id` (string): Unique identifier for each sample
- `qtype` (string): Major task category (4 values)
  - `FactChecking`, `NumericalReasoning`, `DataAnalysis`, `Visualization`
- `qsubtype` (string): Specific sub-task type (18 values)
  - Examples: `Counting`, `Aggregation`, `Comparison`, `CorrelationAnalysis`, etc.
- `instruction` (string): Complete instruction template with task guidelines
  - Contains the full prompt template defining how to approach the task
  - Includes role definition, guidelines, code format requirements
  - Typically 800-15,000 characters depending on instruction type
- `instruction_type` (string): Reasoning strategy type (4 values)
  - `DP` (Direct Prompting), `TCoT` (Textual Chain-of-Thought)
  - `PoT` (Program-of-Thought), `SCoT` (Structured Chain-of-Thought)
- `table` (string): Table data in JSON format
  - Structure: `{"columns": [...], "data": [[...], [...], ...]}`
- `question` (string): Specific question about the table
- `response` (string): Model's answer including reasoning process

**TableBench Test Dataset Fields**:

- `question`: The table question or task description
- `table`: The table data (JSON format)
- `answer`: The ground truth answer
- `category`: Major category
- `subcategory`: Specific sub-task type

<!-- - `question`: The table question or task description
- `table`: The table data (various formats: CSV, JSON, markdown)
- `answer`: The ground truth answer or expected output
- `category`: Major category (Fact Checking, Numerical Reasoning, Data Analysis, Visualization)
- `subcategory`: Specific sub-task type
- `reasoning_steps`: Optional chain-of-thought reasoning (for training data) -->

### Instruction Types and Reasoning Strategies
Tablebench training data (TableInstruct) supports multiple instruction types content that define how the model approaches reasoning and generates answers. Understanding these types is crucial for dataset filtering and fine-tuning strategy selection.

### Available Instruction Type
**1. Direct Prompting(DP)**
**Characteristics**:
- Provides solutions directly without intermediate reasoning steps
- Simplest instruction format focused on immediate answer generation
- Best for straightforward fact-checking and simple queries
**Instruction Template Pattern**：
  You are a table analyst. Your task is to answer questions based on the table content.
  Read the table below in JSON format: [TABLE]
  Question: [QUESTION]
  Answer directly.
  **Response Format**:
  [Direct Answer]

**2. Textual Chain-of-Thought (TCoT)**
**Characteristics**:
- LLMs incrementally derive intermediate steps through textual reasoning
- Natural language explanations for each reasoning step
- Suitable for complex reasoning requiring logical deduction

**Instruction Template Pattern**:
  You are a table analyst. Your task is to answer questions based on the table content.
  [Guidelines for step-by-step reasoning]
  Think step by step
  Show your reasoning process
  Provide the final answer
  ***Response Format**:
  Let's analyze this step by step:
  [First reasoning step]
  [Second reasoning step]
  ...
  Final Answer: [Answer]

 
#### 3. Program-of-Thought (PoT)

**Characteristics**:
- Decomposes problems into executable Python code
- Separates computation from reasoning using programming
- Ideal for numerical reasoning and computational tasks
- Most common type in TableInstruct for analytical tasks

**Instruction Template Pattern** (actual from dataset):
  You are a data analyst proficient in Python. Your task is to write executable Python
  code to analyze the table and then answer questions.
  [Guidelines]
  1. Based on the question, write out your analytical approach, then write Python code
  2. The code needs to be concise and easy to understand
  3. Code blocks need to strictly start with
  '''
  import pandas as pd
  df = pd.read_csv('table.csv')
  ...
  print(f'Final Answer: {answer}')
  '''
  4.Your analysis must be based entirely on the above data
  5.Generate executable code with results using print function
  6.Ensure to load the table with: df = pd.read_csv('table.csv')


#### 4. Symbolic Chain-of-Thought (SCoT)

**Characteristics**:
- A methodology that utilizes Python-based instructions to facilitate logical reasoning
- Combines symbolic reasoning with executable code verification
- Three primary steps repeated until a definitive conclusion is derived
- Distinguishes itself from PoT by emphasizing iterative analysis-generation-simulation cycles

**Three-Step Process**:
- **STEP-1**: Analyzing the available information to determine the next move
- **STEP-2**: Generating instructions using Python programming language commands
- **STEP-3**: Simulating the outcomes by executing the instructions and analyzing the results

**Instruction Template Pattern**:
  You are a table analyst. Use symbolic reasoning with iterative Python commands.
  Process:
  STEP-1: Analyze available information to determine the next move
  STEP-2: Generate Python programming language commands
  STEP-3: Simulate outcomes by executing instructions and analyzing results
  Repeat these three steps until reaching a definitive conclusion




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


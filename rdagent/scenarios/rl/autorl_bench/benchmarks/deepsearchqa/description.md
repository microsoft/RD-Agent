# DeepSearchQA Task

## Objective
Answer complex questions that require multi-step web search, spanning 17 domains.

## Dataset
- Source: google/deepsearchqa (HuggingFace)
- Size: 900 questions
- Local protocol: split with a fixed random seed into 100 training questions / 200 evaluation questions (remaining samples unused)
- Answer type: Single Answer (35%) / Set Answer (65%)

## Rollout process

The model alternates search and inference via the ReAct format:

Question: "Which countries had GDP > X and..."
Thought: I need to find GDP data first.
Action: search[GDP per capita rankings 2023]
Observation: [search result summarization]
Thought: Now I need to filter by condition Y.
Action: search[condition Y countries list]
Observation: [search result summarization]
Thought: I have enough information.
Action: answer[Country A, Country B]

## Evaluation indicators
- Answers are scored by LLM Judge (recommended gemini-2.5-flash)
- Set Answer must contain all items in gold
- Final score = number of correct answers / total number of questions × 100

## Search backend configuration
- Uses `duckduckgo-search` package by default (no configuration required, but there may be frequency limits)
- It is recommended to configure the `SERPAPI_KEY` environment variable for more stable search results

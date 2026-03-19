# DeepSearchQA 任务

## 目标
回答需要多步网络搜索的复杂问题，涵盖 17 个领域。

## 数据集
- 来源: google/deepsearchqa (HuggingFace)
- 规模: 900 题
- 本地协议: 固定随机种子切分为 100 题训练 / 800 题评测
- 答案类型: Single Answer (35%) / Set Answer (65%)

## Rollout 流程

模型通过 ReAct 格式交替搜索和推理：

Question: "Which countries had GDP > X and..."
Thought: I need to find GDP data first.
Action: search[GDP per capita rankings 2023]
Observation: [search result summarization]
Thought: Now I need to filter by condition Y.
Action: search[condition Y countries list]
Observation: [search result summarization]
Thought: I have enough information.
Action: answer[Country A, Country B]

## 评测指标
- 答案由 LLM Judge 打分（推荐 gemini-2.5-flash）
- Set Answer 需包含 gold 中的所有项目
- 最终分数 = 正确数 / 总题数 × 100

## 搜索后端配置
- 默认使用 `duckduckgo-search` 包（无需配置，但可能有频率限制）
- 推荐配置 `SERPAPI_KEY` 环境变量以获得更稳定的搜索结果

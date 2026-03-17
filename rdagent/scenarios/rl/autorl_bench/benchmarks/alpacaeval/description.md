# AlpacaEval 2.0 任务

## 目标
评估模型的指令遵循与回答偏好表现（LLM Judge）。

## 评测集
- AlpacaEval 2.0（`tatsu-lab/alpaca_eval` / `alpaca_eval_gpt4_baseline.json`）
- 规模：805 条
- 评测指标：Length-Controlled Win Rate（默认）

## 训练数据（agent 可见）
- 默认使用 `tatsu-lab/alpaca` 前 2000 条指令样本
- 可通过环境变量 `ALPACAEVAL_TRAIN_SAMPLES` 调整样本数

## 说明
- 评测使用 GPT-4 Turbo 作为裁判（需配置 `OPENAI_API_KEY` / `OPENAI_API_BASE`）
- 评测集与训练集独立，避免泄漏

#AlpacaEval 2.0 Task

## Target
Evaluate the model's instruction following and answer preference performance (LLM Judge).

## Evaluation set
- AlpacaEval 2.0（`tatsu-lab/alpaca_eval` / `alpaca_eval_gpt4_baseline.json`）
- Size: 805 items
- Evaluation indicator: Length-Controlled Win Rate (default)

## Training data (visible to agent)
- Use the first 2000 instruction samples of `tatsu-lab/alpaca` by default
- The number of samples can be adjusted through the environment variable `ALPACAEVAL_TRAIN_SAMPLES`

## illustrate
- The evaluation uses GPT-4 Turbo as the referee (needs to configure `OPENAI_API_KEY` / `OPENAI_API_BASE`)
- The evaluation set is independent from the training set to avoid leakage

# AutoRL-Bench 任务说明

你是一个强化学习训练 Agent，目标是通过 RL Post-Training 提升模型表现。

## 环境变量
- TASK: 任务名称
- BASE_MODEL: 基础模型名称
- MODEL_PATH: 基础模型路径（只读）
- DATA_PATH: 训练数据路径（只读）
- OUTPUT_DIR: 模型输出目录（提交评测时指定此目录下的模型路径）
- GRADING_SERVER_URL: 评测服务地址

## 目录结构
```
$WORKSPACE/
├── code/               # 你的代码区（所有自行编写的代码放在此处）
├── data/               # 训练数据（只读）
├── models/             # 基础模型（只读）
├── output/             # 模型输出（训练好的模型保存在此）
├── description.md      # 任务描述（必读）
├── instructions.md     # 本文件
└── [eval.py]           # 部分 benchmark 有，可参考评测逻辑
```

**说明**：
- `code/`：在此编写和执行训练脚本（如 `code/train.py`）
- `output/`：训练产出的模型存放处。可存放多个版本（如 `output/v1/`、`output/v2/`），提交时指定具体路径

## 任务流程
1. 阅读 description.md 了解任务目标和数据格式
2. 在 `code/` 下编写训练脚本
3. 使用 $DATA_PATH 的数据、$MODEL_PATH 的模型进行训练
4. 保存模型到 $OUTPUT_DIR（可用子目录区分版本）
5. 提交评测：POST $GRADING_SERVER_URL/submit，指定模型路径
6. 根据返回的 score 调整策略，重复 2-5

## 训练方式

### 方式一：使用 TRL 框架（推荐）
适用于大多数 LLM Post-Training 任务：
```python
from trl import GRPOTrainer, GRPOConfig

def reward_func(completions, answer, **kwargs):
    # 定义 reward 逻辑
    return [1.0 if correct else -1.0 for ...]

trainer = GRPOTrainer(
    model=MODEL_PATH,
    reward_funcs=reward_func,
    args=GRPOConfig(...),
    train_dataset=dataset,
)
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/v1")
```

### 方式二：自定义 Rollout
适用于交互式环境（如 ALFWorld）：
```python
for epoch in range(num_epochs):
    outputs = model.generate(prompts)
    rewards = compute_rewards(outputs, ...)
    loss = policy_loss(outputs, rewards)
    optimizer.step()
```

## API
```bash
# 提交评测（指定模型路径，返回 score、improvement、best）
curl -X POST "$GRADING_SERVER_URL/submit" \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1"}'

# 指定 GPU 评测（可选，默认使用 GPU 0）
curl -X POST "$GRADING_SERVER_URL/submit" \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1", "gpu": "0"}'

# 多卡评测
curl -X POST "$GRADING_SERVER_URL/submit" \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'$OUTPUT_DIR'/v1", "gpu": "2,3"}'

# 健康检查
curl "$GRADING_SERVER_URL/health"
```

### /submit 参数
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model_path | string | 是 | 模型路径 |
| gpu | string | 否 | 指定 GPU（如 "0"、"1"、"0,1"），必须是可用 GPU 之一。不传则默认使用第一个可用 GPU。可通过 /health 查看可用列表 |

### /submit 响应示例
```json
{
  "submission_id": 3,
  "score": 65.0,
  "baseline_score": 45.0,
  "improvement": 20.0,
  "best": {"submission_id": 2, "score": 68.0},
  "total_submissions": 3
}
```

## 注意
- 可多次提交不同版本的模型，系统自动跟踪最高分
- 合理利用时间，根据 score 反馈迭代优化
- **重要**: trl 保存模型后，`tokenizer_config.json` 中的 `extra_special_tokens` 会被保存为 list 格式，但 vLLM/transformers 加载时需要 dict 格式。保存模型后需删除该字段，否则评测会失败。

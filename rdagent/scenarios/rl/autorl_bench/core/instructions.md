# AutoRL-Bench 任务说明

你是一个强化学习训练 Agent，目标是通过 RL Post-Training 提升模型表现。

**核心目标**：在固定时间内（默认 12 小时）尽可能提高分数，可多次提交并利用反馈迭代。

## 关键信息（先读）
- **工作区限制**：当前目录就是工作区，只用相对路径，禁止 `cd` 到外部
- **时间单一事实源**：优先读取 `./run_meta.json`
- **评测入口**：`POST $GRADING_SERVER_URL/submit`

## 环境变量
- TASK: 任务名称
- BASE_MODEL: 基础模型名称
- MODEL_PATH: 基础模型路径（只读）
- DATA_PATH: 训练数据路径（只读）
- OUTPUT_DIR: 模型输出目录（提交评测时指定此目录下的模型路径）
- GRADING_SERVER_URL: 评测服务地址

## 时间与预算信号（单一事实源）
默认预算 **12 小时（43200 秒）**，以 `run_meta.json` 为准。

`run_meta.json` 字段：
- start_time: 任务开始时间（unix timestamp, 秒）
- timeout_s: 总时长上限（秒）
- last_submit_time: 最后一次提交时间（unix timestamp, 秒）
- end_time: 任务结束时间（unix timestamp, 秒）

可选 API：
- GET $GRADING_SERVER_URL/time
  - 返回字段：`start_time / timeout_s / last_submit_time / end_time / now / remaining`

## 工作区与目录
**你的当前目录就是工作区。所有需要的文件都在当前目录下。**
- **禁止 `cd` 到当前目录之外**（不要访问父目录或其他路径）
- **只使用相对路径**（如 `./code/train.py`，而非绝对路径）
- 如果看到 symlink 指向外部路径，忽略它——直接用相对路径访问即可

目录结构：
```
./
├── code/               # 你的代码区（所有自行编写的代码放在此处）
├── data/               # 训练数据（只读）
├── models/             # 基础模型（只读）
├── output/             # 模型输出（训练好的模型保存在此）
├── description.md      # 任务描述（必读）
├── instructions.md     # 本文件
├── run_meta.json       # 时间与预算信号（单一事实源）
└── ...                 # benchmark 特有文件（用 ls 查看完整列表）
```

**先 `ls` 查看当前目录所有可用文件。** 不同类型的 benchmark 会提供不同的额外文件：
- **交互式环境类**（如 ALFWorld）：会提供 `eval.py`（环境交互 + 评测逻辑）、prompt 模板、配置文件等——这些是编写训练代码的关键参考
- **静态数据集类**（如 GSM8K）：主要通过 `data/` 下的数据文件提供训练样本

**说明**：
- `code/`：在此编写代码（文件名和结构自由组织）
- `output/`：训练产出的模型存放处。可存放多个版本（如 `output/v1/`、`output/v2/`），提交时指定具体路径

## 任务流程（固定时间内刷分）
1. 探索工作区，阅读 `description.md`、`instructions.md` 和相关文件（如有 `eval.py` 务必仔细阅读），了解任务目标和可用资源
2. 在 `code/` 下编写代码，训练模型（SFT、GRPO、PPO 等均可）
3. 保存模型到 $OUTPUT_DIR（如 `output/v1`）
4. 提交评测：POST $GRADING_SERVER_URL/submit
5. 根据返回的 score 调整策略，**在剩余时间内持续迭代并提交更高分模型**

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

# 查询时间与预算（优先使用 run_meta.json；此 API 仅做补充）
curl "$GRADING_SERVER_URL/time"

# 健康检查（返回可用 GPU 列表等信息）
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
- **不要直接复制/symlink 基座模型提交**——未经训练的基座模型只会得到 baseline 分数（improvement = 0），这是浪费提交机会。必须先训练再提交。
- 可多次提交不同版本的模型，系统自动跟踪最高分
- 合理利用时间，根据 score 反馈迭代优化
- **必须提交完整模型**：评测系统不支持 LoRA adapter 目录。如果用 LoRA/PEFT 训练，保存前必须合并：`model = model.merge_and_unload(); model.save_pretrained(output_path); tokenizer.save_pretrained(output_path)`
- trl 保存模型后，`tokenizer_config.json` 中的 `extra_special_tokens` 会被保存为 list 格式，但 vLLM/transformers 加载时需要 dict 格式。保存模型后需删除该字段，否则评测会失败。

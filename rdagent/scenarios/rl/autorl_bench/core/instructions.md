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
└── ...                 # benchmark 特有文件（用 ls 查看完整列表）
```

**先 `ls $WORKSPACE` 查看所有可用文件。** 不同类型的 benchmark 会提供不同的额外文件：
- **交互式环境类**（如 ALFWorld）：会提供 `eval.py`（环境交互 + 评测逻辑）、prompt 模板、配置文件等——这些是编写训练代码的关键参考
- **静态数据集类**（如 GSM8K）：主要通过 `data/` 下的数据文件提供训练样本

务必先探索工作区，了解可用资源后再编写代码。

**说明**：
- `code/`：在此编写和执行训练脚本（如 `code/train.py`）
- `output/`：训练产出的模型存放处。可存放多个版本（如 `output/v1/`、`output/v2/`），提交时指定具体路径

## 任务流程
1. `ls $WORKSPACE` 查看所有可用文件
2. 阅读 `description.md` 了解任务目标
3. 如果有 `eval.py`，**仔细阅读**——它包含环境交互逻辑、模型推理方式和评测流程
4. 探索 `data/` 了解训练数据格式
5. 在 `code/` 下编写训练脚本（SFT、GRPO、PPO 等均可，最终目标是 RL post-training）
6. 保存模型到 $OUTPUT_DIR
7. 提交评测：POST $GRADING_SERVER_URL/submit
8. 根据返回的 score 调整策略，重复 5-7

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
- 可多次提交不同版本的模型，系统自动跟踪最高分
- 合理利用时间，根据 score 反馈迭代优化
- trl 保存模型后，`tokenizer_config.json` 中的 `extra_special_tokens` 会被保存为 list 格式，但 vLLM/transformers 加载时需要 dict 格式。保存模型后需删除该字段，否则评测会失败。

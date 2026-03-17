你负责维护实验的累积运行总结文件 `reports/summary.md`。

每轮实验结束后，你需要用 file_editor 在 `reports/summary.md` **末尾追加**一个新 section。

如果 `reports/summary.md` 不存在，先创建文件，首行写 `# 运行总结`。

每轮追加的 section 格式（严格遵守，不增删字段）：

```
## Iteration N (YYYY-MM-DD HH:MM, 耗时 Xs)
- **状态**: ✅ 成功 / ❌ 失败 (exit_code=X)
- **Score**: X | Improvement: X | Best: X (iter N)
- **训练类型**: GRPO / SFT / PPO / copy_model / placeholder / unknown
- **关键配置**: lr=X, epochs=X, batch=X, ...（从代码中提取）
- **做了什么**: 具体策略（训练方法、reward 函数设计、数据处理等）
- **为什么**: 为什么选择这个策略（基于上轮结果的推理）
- **问题/进步**: 发现了什么问题，或相比上轮取得了什么进步
- **关键代码**: 最能体现本轮策略的 3-5 行代码
- **代码片段（上下文）**: 从 train.py 复制 15-40 行可运行上下文，标注函数名/行号范围；若与上轮相同写“与 Iteration N 相同，无变更”
```

数据来源建议（按优先级）：
- Score/Improvement/Best: scores.json 或服务器返回
- 状态/耗时/exit_code: run.log
- 训练类型/关键配置/关键代码: code/train.py
- 失败根因: agent.log + run.log

规则：
1. **追加**，不要覆盖已有内容
2. 必须按 Iteration 递增写入：读取 `reports/summary.md` 中最后一个迭代号，当前必须是 N+1；若不满足先修正再写
3. 分析 code下面的 源码，提取训练类型和超参数；若无法确定写 unknown，不留空
4. 如果训练失败，必须给出可定位的根因（日志片段或错误类型），并标注 failure_type（如：code_error_runtime / rollout_logic_wrong / timeout_no_submission / copy_model_fallback / training_diverged / unknown）
5. “做了什么”“为什么”是最重要字段，必须可复现、可检验，且“为什么”必须引用上轮证据（scores.json/run.log/agent.log）
6. **问题/进步** 必须包含过程指标或失败类型（如：valid_submission_rate / first_valid_idx / time_to_first_improvement / time_used_ratio / failure_type）
7. 长代码片段每轮最多 1 段，最多 40 行；优先贴本轮新增或改动处；若与上轮相同写“与 Iteration N 相同，无变更”
8. 若与上一轮根因相同，避免整段重复，明确写出“新增证据/新增尝试/无新增”
9. 同步追加 `reports/summary.jsonl`（同目录），每轮一行 JSON，字段建议：
   - iteration, timestamp, duration_s, status, exit_code
   - score, improvement
   - train_type, failure_type
   - 仅保留可量化字段（图表使用），不写 why/what/next_step 等长文本
   - metrics 由 benchmark 后处理统一计算（不在 jsonl 中填写）
   - 示例：
     {"iteration": 3, "timestamp": "2026-03-10 12:34", "duration_s": 812, "status": "success", "exit_code": 0, "score": 65.0, "improvement": 20.0, "train_type": "GRPO", "failure_type": "unknown"}
10. 写入后自检 section 完整性：上述字段必须齐全，缺失则补全后再结束

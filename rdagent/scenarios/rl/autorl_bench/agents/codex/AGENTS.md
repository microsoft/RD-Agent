# AutoRL-Bench Agent Guidelines

## Summary Maintenance (MANDATORY)

You MUST maintain a file called `summary.md` in the workspace root. Update it **after every training attempt and every submission**, not just at the end.

### Format

```markdown
# 运行总结

## Attempt N (YYYY-MM-DD HH:MM)
- **状态**: ✅ 成功 / ❌ 失败
- **Score**: X.XX | Improvement: +Y.YY | Best: Z.ZZ
- **训练类型**: SFT / GRPO / PPO / DPO / ...
- **超参数**: lr=X, epochs=Y, batch_size=Z, ...
- **做了什么**: 简述本次尝试的策略和具体操作
- **为什么**: 为什么选择这个方法/这些超参数
- **问题/进步**: 遇到了什么问题，相比上次有什么改进
- **关键代码**: 关键改动的代码片段（如有）
- **下一步建议**: 基于本次结果，下一步打算怎么做
```

### Rules
1. **Append only** — never overwrite previous attempts
2. Analyze `code/train.py` source to extract training type and hyperparameters
3. If training fails, extract root cause from error output
4. "做了什么" and "为什么" are the most important fields — be thorough
5. Update summary.md IMMEDIATELY after each submission result comes back
6. Include the grading server response (score, improvement, best) verbatim

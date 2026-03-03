# HumanEval 任务

## 目标
训练模型在 HumanEval 的 Python 函数补全任务上获得更高 pass@1。

## 数据格式
```json
{
  "question": "函数签名与 docstring（prompt）",
  "answer": "参考实现（canonical_solution）",
  "task_id": "HumanEval/0",
  "entry_point": "目标函数名",
  "test": "用于校验实现正确性的测试代码"
}
```

## 评测指标
- pass@1（由 OpenCompass HumanEval 配置执行）

## 数据划分
- HumanEval 原始 `test` 共 164 条。
- 训练可见数据固定为前 82 条（`[:82]`）。
- 自动评测固定为后 82 条（`[82:]`），与训练集不重叠。

## 提示
- 生成可执行的 Python 函数实现，优先保证正确性。
- 注意函数名必须与 `entry_point` 一致。

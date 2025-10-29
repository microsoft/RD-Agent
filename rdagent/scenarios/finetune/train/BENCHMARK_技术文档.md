# Benchmark评估系统技术文档

## 1. 概述

FT场景的Benchmark评估系统使用 **lm-evaluation-harness** 对微调后的模型进行标准化评估。

### 核心特点

- **评估工具**: lm-evaluation-harness (EleutherAI开源项目)
- **后端引擎**: vLLM (高性能推理)
- **支持任务**: 200+ 预配置任务 (GSM8K、MMLU、HumanEval等)
- **结果格式**: JSON (结构化、多指标)
- **数据隔离**: 训练数据与评估数据分开存储

---

## 2. 系统架构

### 2.1 整体流程

```
训练完成 → 生成Adapter → Benchmark Docker → 评估结果
    ↓           ↓              ↓              ↓
LLaMA-Factory  LoRA权重    lm_eval CLI    JSON报告
```

### 2.2 目录结构

```
git_ignore_folder/finetune/
├── datasets/              # 训练数据 (LLaMA-Factory格式)
│   ├── dataset_info.json
│   └── alpaca-zh/
│
├── benchmarks/            # 评估数据 (HF自动缓存)
│   ├── gsm8k/
│   ├── mmlu/
│   └── hellaswag/
│
└── models/
    └── Qwen/
```

**设计原则**: 训练数据手动管理，评估数据自动下载，互不干扰。

---

## 3. 组件详解

### 3.1 Docker环境

**位置**: `rdagent/scenarios/finetune/docker/lm_eval/`

#### Dockerfile
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 安装lm-evaluation-harness
RUN pip install lm-eval[vllm]==0.4.9.1

# 安装LoRA支持
RUN pip install peft transformers accelerate

# 设置数据集缓存目录
ENV HF_DATASETS_CACHE=/workspace/benchmarks
```

**作用**: 提供独立的评估环境，包含lm_eval工具和vLLM推理引擎。

#### eval_entrypoint.sh
```bash
#!/bin/bash
# 读取环境变量
TASKS="${BENCHMARK_TASKS:-gsm8k}"
BASE_MODEL="${BASE_MODEL}"
ADAPTER_PATH="${ADAPTER_PATH:-/workspace/output}"

# 执行lm_eval命令
lm_eval --model vllm \
    --model_args "pretrained=${BASE_MODEL},lora_local_path=${ADAPTER_PATH},..." \
    --tasks ${TASKS} \
    --batch_size auto \
    --apply_chat_template \
    --output_path /workspace/benchmark_results/results.json \
    --log_samples
```

**作用**: 封装lm_eval CLI调用，通过环境变量配置参数。

---

### 3.2 Python评估器

**位置**: `rdagent/scenarios/finetune/train/benchmark.py`

#### 核心类: FTBenchmarkEvaluator

```python
class FTBenchmarkEvaluator(CoSTEEREvaluator):
    """Benchmark评估器"""
    
    def __init__(self, scen, tasks=None, limit=None):
        self.tasks = tasks or FT_RD_SETTING.benchmark_datasets
        self.limit = limit  # 样本数限制 (用于快速测试)
    
    def evaluate(self, target_task, implementation, ...):
        """执行评估流程"""
        # 1. 验证adapter文件
        validation = self._validate_adapter_files(output_path)
        
        # 2. 创建Docker环境
        env, env_vars = get_benchmark_env(...)
        
        # 3. 运行Docker评估
        result = env.run(entry="bash run_benchmark.sh", ...)
        
        # 4. 解析JSON结果
        scores = self._parse_results(results_path)
        
        # 5. 返回反馈
        return CoSTEERSingleFeedback(...)
```

#### 辅助函数: get_benchmark_env

```python
def get_benchmark_env(tasks, adapter_path, base_model, limit=None):
    """创建Benchmark Docker环境"""
    
    # Docker配置
    conf = BenchmarkDockerConf()
    env = BenchmarkDockerEnv(conf=conf)
    
    # 环境变量
    env_vars = {
        "BENCHMARK_TASKS": ",".join(tasks),
        "BASE_MODEL": base_model,
        "ADAPTER_PATH": adapter_path,
        "HF_DATASETS_CACHE": "/workspace/benchmarks",
    }
    
    if limit:
        env_vars["LIMIT"] = str(limit)
    
    return env, env_vars
```

**作用**: 
1. 验证训练输出的adapter文件
2. 配置Docker环境和参数
3. 执行评估并解析结果
4. 返回结构化反馈

---

### 3.3 Docker配置

**位置**: `rdagent/utils/env.py`

```python
class BenchmarkDockerConf(DockerConf):
    """Benchmark Docker配置"""
    
    dockerfile_folder_path = Path(...) / "docker" / "lm_eval"
    image = "rdagent-lm-eval:latest"
    default_entry = "bash /app/eval_entrypoint.sh"
    
    running_timeout_period = 3600  # 1小时超时
    mem_limit = "32g"
    enable_gpu = True
```

**作用**: 定义Docker镜像构建路径、资源限制、超时时间等配置。

---

## 4. 调用方式

### 4.1 自动评估 (推荐)

训练完成后自动触发评估：

```python
from rdagent.scenarios.finetune.train.runner import LLMFinetuneRunner

runner = LLMFinetuneRunner(scen=scen)
exp = runner.develop(exp)  # 训练 + 自动评估
```

**流程**:
1. `LLMFinetuneRunner` 包含 `FTBenchmarkEvaluator`
2. 训练完成后自动调用 `evaluate()`
3. 结果保存在 `workspace/benchmark_results/results.json`

### 4.2 手动评估

独立运行评估：

```python
from rdagent.scenarios.finetune.train.benchmark import FTBenchmarkEvaluator

# 创建评估器
evaluator = FTBenchmarkEvaluator(
    scen=scen,
    tasks=["gsm8k", "mmlu"],  # 指定任务
    limit=10,  # 限制样本数 (快速测试)
)

# 执行评估
feedback = evaluator.evaluate(
    target_task=task,
    implementation=workspace,
    gt_implementation=None,
)

# 查看结果
print(feedback.return_checking)  # 评估报告
print(feedback.code)             # 平均分数
```

### 4.3 配置方式

通过环境变量或 `.env` 文件配置：

```bash
# 评估任务 (逗号分隔)
FT_BENCHMARK_DATASETS=gsm8k,mmlu,hellaswag

# 超时时间 (秒)
FT_BENCHMARK_TIMEOUT=3600

# 数据集缓存目录
HF_DATASETS_CACHE=/path/to/finetune/benchmarks
```

---

## 5. 数据流转

### 5.1 输入数据

```
workspace/
└── output/                    # 训练输出
    ├── adapter_model.safetensors  # LoRA权重
    ├── adapter_config.json        # LoRA配置
    └── tokenizer files...
```

### 5.2 评估过程

```
1. 验证adapter文件
   ↓
2. 准备Docker环境
   - 挂载workspace目录
   - 设置环境变量
   ↓
3. Docker内执行
   - 下载benchmark数据集 (缓存到benchmarks/)
   - 加载base model + adapter
   - 运行lm_eval评估
   ↓
4. 生成结果文件
```

### 5.3 输出结果

```
workspace/
└── benchmark_results/
    ├── results.json              # 主结果文件
    └── samples_*.jsonl           # 详细样本 (可选)
```

**results.json 格式**:
```json
{
  "results": {
    "gsm8k": {
      "exact_match,flexible-extract": 0.6875,
      "exact_match,strict-match": 0.2969
    },
    "mmlu": {
      "acc": 0.4521,
      "acc_stderr": 0.0123
    }
  }
}
```

---

## 6. 测试脚本

### 6.1 快速测试

**位置**: `rdagent/scenarios/finetune/train/test_benchmark.py`

```bash
# 快速测试 (10个样本)
cd rdagent/scenarios/finetune/train
python test_benchmark.py
```

**功能**:
- 验证评估流程是否正常
- 使用限制样本数 (limit=10) 快速完成
- 打印详细的执行过程和结果

### 6.2 完整测试

```bash
# 完整评估 (所有样本)
python test_benchmark.py full
```

### 6.3 测试代码示例

```python
def test_benchmark_quick():
    """快速测试"""
    # 创建场景
    scen = LLMFinetuneScen()
    
    # 创建评估器 (限制10个样本)
    evaluator = FTBenchmarkEvaluator(
        scen=scen,
        tasks=["gsm8k"],
        limit=10,
    )
    
    # 创建workspace
    workspace = FBWorkspace(workspace_path)
    task = Task(name="test_benchmark")
    
    # 运行评估
    feedback = evaluator.evaluate(task, workspace, None)
    
    # 打印结果
    print("执行状态:", feedback.execution)
    print("评估结果:", feedback.return_checking)
    print("平均分数:", feedback.code)
```

---

## 7. 常见任务

### 7.1 支持的Benchmark任务

| 类别 | 任务名称 | 说明 |
|------|---------|------|
| 数学 | `gsm8k` | 小学数学应用题 |
| 数学 | `math` | 高中/大学数学 |
| 推理 | `hellaswag` | 常识推理 |
| 推理 | `arc_easy`, `arc_challenge` | AI2推理挑战 |
| 知识 | `mmlu` | 多任务语言理解 |
| 代码 | `humaneval` | Python代码生成 |
| 中文 | `cmmlu` | 中文多任务理解 |

完整列表: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks

### 7.2 快速测试流程

```python
# 1. 限制样本数
evaluator = FTBenchmarkEvaluator(scen=scen, tasks=["gsm8k"], limit=10)
feedback = evaluator.evaluate(...)

# 2. 查看结果是否合理
if feedback.final_decision:
    print("测试通过，可以运行完整评估")
    
# 3. 完整评估
evaluator_full = FTBenchmarkEvaluator(scen=scen, tasks=["gsm8k"])
feedback_full = evaluator_full.evaluate(...)
```

### 7.3 多任务评估

```python
# 同时评估多个任务
tasks = ["gsm8k", "mmlu", "hellaswag"]
evaluator = FTBenchmarkEvaluator(scen=scen, tasks=tasks)
feedback = evaluator.evaluate(...)

# 解析结果
import json
with open("workspace/benchmark_results/results.json") as f:
    data = json.load(f)
    for task, metrics in data["results"].items():
        print(f"{task}: {metrics}")
```

---

## 8. 故障排查

### 8.1 Adapter验证失败

**错误**: `Adapter validation failed: No adapter weight file`

**原因**: 训练未生成adapter文件

**解决**:
1. 检查训练是否成功完成
2. 确认 `workspace/output/` 目录存在
3. 确认存在 `adapter_model.safetensors` 或 `adapter_model.bin`

### 8.2 Docker构建失败

**错误**: `Docker build failed`

**解决**:
```bash
# 手动构建Docker镜像
cd rdagent/scenarios/finetune/docker/lm_eval
docker build -t rdagent-lm-eval:latest .
```

### 8.3 CUDA内存不足

**错误**: `CUDA out of memory`

**解决**: 修改 `benchmark.py` 中的批量大小
```python
env_vars["BATCH_SIZE"] = "4"  # 降低批量大小
```

### 8.4 任务未找到

**错误**: `Task 'xyz' not found`

**解决**: 
1. 检查任务名称拼写
2. 查看可用任务: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
3. 或运行: `lm_eval --tasks list`

### 8.5 评估超时

**错误**: `Benchmark execution failed (exit_code=124)`

**解决**: 增加超时时间
```bash
export FT_BENCHMARK_TIMEOUT=7200  # 2小时
```

---

## 9. 性能优化

### 9.1 调整批量大小

```python
# 在get_benchmark_env()中修改
env_vars["BATCH_SIZE"] = "16"  # 增大批量提升速度
```

### 9.2 使用多GPU

```python
env_vars["NUM_GPUS"] = "4"  # 使用4个GPU
```

### 9.3 限制样本数

```python
# 开发阶段使用小样本快速迭代
evaluator = FTBenchmarkEvaluator(scen=scen, tasks=["gsm8k"], limit=50)
```

---

## 10. 与OpenCompass的对比

| 维度 | OpenCompass (旧) | lm-evaluation-harness (新) |
|------|-----------------|---------------------------|
| 配置代码 | 241行Python | 60行Bash |
| 可用任务 | ~10个 | 200+ |
| 推理引擎 | HuggingFace | vLLM (快2-10倍) |
| 结果格式 | CSV | JSON (结构化) |
| 维护性 | 需自己维护 | 社区维护 |
| LoRA支持 | 需手动配置 | 原生支持 |

---

## 11. 核心原理总结

### 工作流程

```
1. 训练完成
   ↓
2. FTBenchmarkEvaluator.evaluate() 被调用
   ↓
3. 验证adapter文件完整性
   ↓
4. 创建BenchmarkDockerEnv
   ↓
5. 设置环境变量 (tasks, model, adapter_path)
   ↓
6. Docker内执行 eval_entrypoint.sh
   ↓
7. eval_entrypoint.sh 调用 lm_eval CLI
   ↓
8. lm_eval 下载数据集 (缓存到benchmarks/)
   ↓
9. lm_eval 加载模型+adapter，执行评估
   ↓
10. 生成results.json
   ↓
11. Python解析JSON，返回CoSTEERSingleFeedback
   ↓
12. 评估完成
```

### 关键设计

1. **数据隔离**: 训练数据 (`datasets/`) 与评估数据 (`benchmarks/`) 分开
2. **环境隔离**: 独立的Docker环境，不影响训练
3. **配置简化**: 环境变量配置，无需生成复杂配置文件
4. **结果结构化**: JSON格式，支持多指标
5. **社区生态**: 使用成熟工具，200+任务开箱即用

---

## 12. 快速参考

### 最小使用示例

```python
# 1. 导入
from rdagent.scenarios.finetune.train.benchmark import FTBenchmarkEvaluator
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen

# 2. 创建
scen = LLMFinetuneScen()
evaluator = FTBenchmarkEvaluator(scen=scen, tasks=["gsm8k"], limit=10)

# 3. 评估
feedback = evaluator.evaluate(task, workspace, None)

# 4. 查看
print(feedback.return_checking)
```

### 配置文件 (.env)

```bash
FT_BENCHMARK_DATASETS=gsm8k,mmlu
FT_BENCHMARK_TIMEOUT=3600
HF_DATASETS_CACHE=/path/to/benchmarks
```

### 测试命令

```bash
# 快速测试
python test_benchmark.py

# 完整测试
python test_benchmark.py full
```

---

## 附录: 文件清单

```
rdagent/scenarios/finetune/
├── docker/lm_eval/
│   ├── Dockerfile              # Docker环境定义
│   └── eval_entrypoint.sh      # 评估脚本
│
└── train/
    ├── benchmark.py            # 评估器实现
    ├── runner.py               # 训练+评估运行器
    ├── test_benchmark.py       # 测试脚本
    └── BENCHMARK_技术文档.md   # 本文档
```

---

**文档版本**: v2.0  
**更新日期**: 2025-10-29  
**适用场景**: FT (Fine-Tuning) Scenario


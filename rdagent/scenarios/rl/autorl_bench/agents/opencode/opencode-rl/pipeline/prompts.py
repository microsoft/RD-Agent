"""Pipeline prompt 构建逻辑 — 探索式自主 prompt。"""

import os
from pathlib import Path

from .types import IterationResult


def build_code_prompt(
    iteration: int,
    workspace: str,
    base_model: str,
    task_description: str,
    history: list[IterationResult],
    gpu_info: dict | None = None,
    task_type: str = "math",
    expose_files: tuple[str, ...] = (),
) -> str:
    """构建代码生成阶段的 prompt — 根据 task_type 自动切换引导内容。"""
    model_path = os.environ.get("MODEL_PATH", "")
    data_path = os.environ.get("DATA_PATH", "")
    output_dir = os.environ.get("OUTPUT_DIR", "")
    training_timeout = os.environ.get("TRAINING_TIMEOUT", "3600")

    gpu_section = ""
    if gpu_info:
        gpu_section = f"- 硬件：{gpu_info['num_gpus']}x {gpu_info['gpu_name']}"

    # expose_files 列表
    expose_section = ""
    if expose_files:
        files_list = "\n".join(f"  - {workspace}/{f}" for f in expose_files)
        expose_section = f"- 参考文件（已拷贝到工作空间）：\n{files_list}"

    # 根据 task_type 切换数据和训练方式说明
    if task_type == "interactive":
        data_section = f"- 训练数据：这是交互式环境任务，数据目录 {data_path}/ 仅供参考，核心训练信号来自环境交互"
        method_hints = """- 仔细阅读工作空间中的参考文件，理解环境 API、交互协议和评测逻辑
- 先读 description.md 了解任务目标和环境交互方式"""
    else:
        data_section = f"- 训练数据：{data_path}/train.jsonl"
        method_hints = """- 可以用 `head -5 {data_path}/train.jsonl` 查看数据
- TRL 的 GRPOTrainer 适合这类 RL 后训练任务""".format(data_path=data_path)

    history_section = ""
    if history:
        rows = []
        for h in history:
            score_s = f"{h.score:.2f}" if h.score is not None else "-"
            improvement_s = f"{h.improvement:+.2f}" if h.improvement is not None else "-"
            status = "OK" if h.exit_code == 0 else f"FAIL({h.exit_code})"
            rows.append(f"| {h.iteration} | {status} | {h.training_time:.0f}s | {score_s} | {improvement_s} |")

        history_section = "\n## 历史记录\n"
        history_section += "| 轮次 | 状态 | 耗时 | 评测分数 | vs Baseline |\n"
        history_section += "|------|------|------|---------|-------------|\n"
        history_section += "\n".join(rows) + "\n"

        # 注入上一轮的分析报告，让 agent 直接获得之前的诊断洞察
        last = history[-1]
        if last.analysis and last.analysis.strip():
            history_section += f"\n## 上一轮诊断报告（第 {last.iteration} 轮）\n"
            history_section += last.analysis.strip()[:3000] + "\n"

        history_section += f"""
## 上一轮文件
- 上一轮代码：{workspace}/code/train.py
- 上一轮训练日志：{workspace}/code/training_stdout.log

请根据上面的诊断报告和历史分数，针对性地改进代码。如果需要更多信息，可以自行读取日志和代码。
"""

    return f"""你是 RL 后训练工程师。目标：写一个训练脚本来提升模型在下面任务上的性能。

## 工作空间
- 代码目录：{workspace}/code/（在这里写 train.py）
{data_section}
- 基础模型：{model_path}（{base_model}）
- 输出目录：{output_dir}（训练后的模型保存在这里）
- 任务描述：{workspace}/description.md
{expose_section}
{gpu_section}

## 你的任务
1. 先探索：读 description.md（用 read limit=20）、了解任务要求、**采样**参考文件（head -3）
2. 再设计：选择训练方法、设计 reward 函数、确定超参数
3. 最后写代码：生成 {workspace}/code/train.py
⚠️ 探索时 **绝对不要** 一次性读取整个数据文件，只采样前 3-5 条了解格式

## 输出合约
- 文件：{workspace}/code/train.py
- 执行方式：pipeline 用 `accelerate launch train.py` 运行（自动多卡 DDP）
- 环境变量：MODEL_PATH, DATA_PATH, OUTPUT_DIR 在运行时可用
- 预装库：torch, transformers, trl, datasets, accelerate, peft（禁止 pip install）
- 训练超时：{training_timeout} 秒
- 训练完成后必须把**完整模型**保存到 $OUTPUT_DIR（如果用了 LoRA，必须 merge_and_unload() 后再保存，确保目录包含 config.json 和 model.safetensors）
- Pipeline 会用 vLLM 加载 $OUTPUT_DIR 里的模型评测，只能加载完整模型，不能加载单独的 LoRA adapter

## 任务描述
{task_description}

## 提示
- 可以用 `python3 -c "..."` 快速验证想法
{method_hints}
- 你只负责写代码，不要自己执行训练脚本。pipeline 会用 accelerate 自动运行
- 完成后调用 finish 工具结束

## CRITICAL: Context Size Limits（违反会导致 session 崩溃）
- **绝对禁止** 一次性读取整个数据文件（jsonl、csv、parquet 等），这些文件可能有数万行，
  单次读取会直接撑爆 LLM context window，导致 session 崩溃、timeout、无法恢复
- 查看数据格式：只用 `head -3 file` 或 `python3 -c "..."` 采样前 3-5 条
- read 工具：**必须设置 limit=20**，用 offset 翻页
- grep：加 `-m 10` 限制匹配数量，如 `grep -n 'pattern' file | head -10`
- 大日志文件：先 `wc -l` 看行数，再 `tail -30` 看末尾
- **错误示例（绝对禁止）**：`cat file.py`、`cat data.jsonl`、不带 limit 的 read、`sed -n '1,500p'`、不加限制的 grep
- **正确示例**：`head -3 train.jsonl`、`read file limit=20`、`grep -n 'error' log | head -10`

## 效率要求（严格遵守）
- **禁止用 inspect.getsource() 逐函数阅读库源码**，这会导致上下文爆炸、LLM 响应超时
- 了解库 API：用 `python3 -c "from trl import GRPOTrainer; help(GRPOTrainer)"` 查看文档
- 验证参数签名：用 `python3 -c "import inspect; from trl import GRPOConfig; print(inspect.signature(GRPOConfig))"`
- **不要花超过 3 步探索**，尽快开始写代码。你已经知道 TRL、transformers、PEFT 的基本用法
- 写完代码后立即调用 finish，不要反复检查和优化
- **DO NOT read entire data files** — 只采样前几条理解格式即可
{history_section}"""


def build_fix_prompt(
    code_path: str,
    error_log_path: str,
    data_path: str,
) -> str:
    """构造训练失败后的修复 prompt — 探索式，让 agent 自主读日志和代码诊断。"""

    return f"""训练脚本执行失败了。请诊断错误并修复代码。

## 工作空间
- 需要修复的代码：{code_path}
- 错误日志：{error_log_path}
- 训练数据：{data_path}（只读）

## 你的任务
1. 读错误日志（`tail -30` 看末尾），理解出了什么问题
2. 读当前代码（read limit=20，分段读），找到 bug
3. 如果需要，**采样**训练数据验证理解（`head -3`，禁止读取整个数据文件）
4. 可以跑小段测试代码验证修复
5. 修改 {code_path}
6. 不要运行完整训练——pipeline 会执行
7. 完成后调用 finish 工具结束

## CRITICAL: Context Size Limits（违反会导致 session 崩溃）
- **绝对禁止** 一次性读取整个数据文件或大日志——会撑爆 context window
- read 工具：**必须设置 limit=20**，用 offset 翻页
- grep：加 `-m 10`，如 `grep -n 'error' log | head -10`
- 大日志：先 `wc -l` 看行数，再 `tail -30` 看末尾
- **禁止用 inspect.getsource() 读库源码**
- **DO NOT read entire data files** — 只采样前几条
"""


def build_analysis_prompt(
    iteration: int,
    workspace: str,
    code_path: str,
    training_log_path: str,
    score: float | None,
    evaluation_summary: str = "",
) -> str:
    """构建自分析 prompt — 探索式，让 agent 自主查阅所有资料写诊断报告。"""

    evaluation_section = ""
    if evaluation_summary:
        evaluation_section = f"""
## Grading Server 评测结果
{evaluation_summary}
"""

    return f"""第 {iteration} 轮训练和评测已完成。请分析结果并写出诊断报告。

## 结果概览
- 评测分数：{score if score is not None else "无（评测失败或未运行）"}

## 可用资料（请自行查阅）
- 任务描述：{workspace}/description.md
- 训练代码：{code_path}
- 训练日志：{training_log_path}
{evaluation_section}
## 你的任务
分析训练过程：读代码（read limit=20 分段）、日志（tail -30 看末尾）。理解发生了什么、为什么。

将分析写入 {workspace}/code/analysis.md，包含：
- 做得好的和做得不好的地方
- 性能问题的根因
- 下一轮的具体改进建议（最多3条，按优先级排序）

用日志中的具体数据支撑你的分析（引用 loss 趋势等）。

## CRITICAL: Context Size Limits（违反会导致 session 崩溃）
- **绝对禁止** 一次性读取整个日志或数据文件——会撑爆 context window
- read 工具：**必须设置 limit=20**，用 offset 翻页
- grep：加 `-m 10`，如 `grep -n 'loss' log | head -10`
- 大日志：先 `wc -l` 看行数，再 `tail -30` 看末尾
- **DO NOT read entire data files or log files**

完成后调用 finish 工具结束。
"""

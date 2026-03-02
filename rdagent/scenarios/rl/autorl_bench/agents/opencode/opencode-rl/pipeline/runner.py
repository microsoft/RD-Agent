"""Pipeline 主循环：状态机 + 断点续跑 + Grading Server 评测。

阶段流程：
  每轮迭代:
    CODE_GEN → TRAINING → EVALUATION → ANALYSIS → COMPLETE
"""

import json
import os
import time
from pathlib import Path

from .phases import (
    phase_analysis,
    phase_code_generation,
    phase_evaluation,
    phase_fix_training,
    phase_training,
)
from .state import load_checkpoint, save_checkpoint
from .types import (
    IterationResult,
    IterationState,
    Phase,
    PhaseResult,
    PipelineState,
)
from .ui import (
    console,
    print_data_gpu_info,
    print_evaluation_report,
    print_iteration_header,
    print_iteration_summary,
    print_phase_header,
    print_phase_status,
    print_pipeline_footer,
    print_pipeline_header,
)
from .utils import get_data_stats, get_gpu_info


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_LOG_TRUNCATE = 20000
_ERR_TRUNCATE = 4000


# ---------------------------------------------------------------------------
# 共享工具函数
# ---------------------------------------------------------------------------
def _normalize_score(raw) -> float | None:
    """将 [0,1] 的原始分数归一化为 [0,100] 百分制。"""
    if raw is None or not isinstance(raw, (int, float)):
        return None
    return round(raw * 100, 2) if raw <= 1.05 else raw


# ---------------------------------------------------------------------------
# 迭代内各阶段
# ---------------------------------------------------------------------------
def _run_phase_code_gen(
    state: PipelineState,
    iter_state: IterationState,
    history: list[IterationResult],
    task_description: str,
    gpu_info: dict,
    opencode_model: str,
    opencode_url: str,
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
    task_type: str = "math",
    expose_files: tuple[str, ...] = (),
) -> PhaseResult:
    """CODE_GEN 阶段（含自动重试）。"""
    max_retries = max(1, state.max_retries)
    result: PhaseResult | None = None

    for attempt in range(max_retries):
        if attempt > 0:
            console.print(
                f"\n  [yellow]--- Code generation retry {attempt}/{max_retries - 1} ---[/]"
            )
        result = phase_code_generation(
            iteration=iter_state.iteration,
            workspace=state.workspace,
            base_model=state.base_model,
            task_description=task_description,
            history=history,
            max_agent_steps=state.max_agent_steps,
            gpu_info=gpu_info,
            opencode_model=opencode_model,
            opencode_url=opencode_url,
            stale_timeout=stale_timeout,
            http_timeout=http_timeout,
            task_type=task_type,
            expose_files=expose_files,
        )
        if result.success:
            break
        console.print(f"  [yellow]Code generation failed:[/] {result.error}")

    assert result is not None
    if result.success:
        iter_state.code_path = result.payload.get("code_path", "")
    else:
        iter_state.code_path = str(Path(state.workspace) / "code" / "train.py")
    return result


def _run_phase_training(
    state: PipelineState,
    iter_state: IterationState,
    opencode_model: str,
    opencode_url: str,
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """TRAINING 阶段（含重试循环）。"""
    code_path = iter_state.code_path
    workspace = state.workspace

    train_result = phase_training(
        workspace, code_path, timeout=state.training_timeout,
    )

    training_log_path = str(Path(workspace) / "code" / "training_stdout.log")
    stdout = train_result.payload.get("stdout", "")
    try:
        Path(training_log_path).write_text(stdout, encoding="utf-8")
    except OSError as e:
        console.print(f"  [bold yellow]WARNING:[/] Failed to write training log: {e}")

    total_time = train_result.payload.get("elapsed", 0.0)

    for retry in range(state.max_retries):
        if train_result.success:
            break
        console.print(f"\n  [yellow]--- Fix retry {retry + 1}/{state.max_retries} ---[/]")

        fix_result = phase_fix_training(
            code_path, training_log_path, state.data_path, workspace,
            iteration=iter_state.iteration,
            opencode_model=opencode_model,
            opencode_url=opencode_url,
            max_agent_steps=state.max_agent_steps,
            stale_timeout=stale_timeout,
            http_timeout=http_timeout,
        )
        if not fix_result.success:
            console.print(f"  [red]Fix agent failed:[/] {fix_result.error}")
            break

        console.print(f"  [dim]Agent fix attempt done, re-running training...[/]")
        train_result = phase_training(
            workspace, code_path, timeout=state.training_timeout,
        )
        extra_time = train_result.payload.get("elapsed", 0.0)
        total_time += extra_time
        stdout = train_result.payload.get("stdout", "")
        try:
            Path(training_log_path).write_text(stdout, encoding="utf-8")
        except OSError as e:
            console.print(f"  [bold yellow]WARNING:[/] Failed to write training log: {e}")

    exit_code = train_result.payload.get("exit_code", -1)
    iter_state.exit_code = exit_code
    iter_state.stdout = train_result.payload.get("stdout", "")
    iter_state.training_time = total_time

    return PhaseResult(
        success=train_result.success,
        phase="training",
        payload={
            "exit_code": exit_code,
            "elapsed": total_time,
        },
        error=train_result.error,
    )


def _run_phase_evaluation(
    state: PipelineState,
    iter_state: IterationState,
) -> PhaseResult:
    """EVALUATION 阶段：提交模型到 Grading Server。"""
    grading_url = os.environ.get("GRADING_SERVER_URL", "http://localhost:5000")
    result = phase_evaluation(
        workspace=state.workspace,
        grading_url=grading_url,
        output_dir=state.output_dir,
        timeout=state.eval_timeout,
    )
    if result.success:
        iter_state.score = result.payload.get("score")
        iter_state.improvement = result.payload.get("improvement")
        iter_state.best_score = result.payload.get("best", {}).get("score")
        iter_state.submission_id = result.payload.get("submission_id")
        iter_state.model_path = result.payload.get("model_path", "")
    return result


def _run_phase_analysis(
    state: PipelineState,
    iter_state: IterationState,
    opencode_model: str,
    opencode_url: str,
    stale_timeout: float = 180.0,
    http_timeout: float = 300.0,
) -> PhaseResult:
    """ANALYSIS 阶段（含自动重试）。"""
    workspace = state.workspace
    training_log_path = str(Path(workspace) / "code" / "training_stdout.log")

    # 构建评测摘要供 analysis prompt 使用
    evaluation_summary = ""
    if iter_state.score is not None:
        evaluation_summary = (
            f"Grading Server 评测分数: {iter_state.score}\n"
            f"vs Baseline 提升: {iter_state.improvement}\n"
            f"历史最佳: {iter_state.best_score}"
        )

    max_retries = max(1, state.max_retries)
    result: PhaseResult | None = None

    for attempt in range(max_retries):
        if attempt > 0:
            console.print(
                f"\n  [yellow]--- Analysis retry {attempt}/{max_retries - 1} ---[/]"
            )
        result = phase_analysis(
            iteration=iter_state.iteration,
            workspace=workspace,
            code_path=iter_state.code_path,
            training_log_path=training_log_path,
            score=iter_state.score,
            opencode_model=opencode_model,
            opencode_url=opencode_url,
            max_agent_steps=state.max_agent_steps,
            evaluation_summary=evaluation_summary,
            stale_timeout=stale_timeout,
            http_timeout=http_timeout,
        )
        if result.success:
            break
        console.print(f"  [yellow]Analysis failed:[/] {result.error}")

    assert result is not None
    if result.success:
        iter_state.analysis = result.payload.get("analysis", "")

    return result


# ---------------------------------------------------------------------------
# 主 Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    task: str,
    base_model: str,
    workspace: str,
    data_path: str = "",
    output_dir: str = "",
    max_iterations: int = 5,
    training_timeout: int = 3600,
    max_agent_steps: int = 25,
    max_retries: int = 3,
    fsm_config: dict | None = None,
    resume: bool = False,
    stale_timeout: int = 180,
    http_timeout: int = 300,
    eval_timeout: int = 600,
    task_type: str = "math",
    expose_files: tuple[str, ...] = (),
):
    """运行 Pipeline（状态机 + 断点续跑 + Grading Server 评测）

    阶段流程：
      每轮迭代：
        CODE_GEN → TRAINING → EVALUATION → ANALYSIS → COMPLETE
    """
    pipeline_start = time.time()
    fsm_config = fsm_config or {}

    if not data_path:
        data_path = os.environ.get("DATA_PATH", "")
    if not output_dir:
        output_dir = os.environ.get("OUTPUT_DIR", str(Path(workspace) / "output"))

    opencode_model = os.environ.get("OPENCODE_MODEL", "")
    opencode_url = os.environ.get("OPENCODE_URL", "")

    # ----- 尝试恢复 checkpoint -----
    state: PipelineState | None = None
    if resume:
        state = load_checkpoint(workspace)
        if state:
            console.print(f"  [green]Resuming from checkpoint:[/] iteration {state.current_iteration}")
        else:
            console.print(f"  [dim]No checkpoint found, starting fresh[/]")

    if state is None:
        state = PipelineState(
            task=task,
            base_model=base_model,
            workspace=workspace,
            data_path=data_path,
            output_dir=output_dir,
            max_iterations=max_iterations,
            training_timeout=training_timeout,
            max_agent_steps=max_agent_steps,
            max_retries=max_retries,
            eval_timeout=eval_timeout,
            pipeline_start_time=pipeline_start,
        )

    print_pipeline_header(
        task=task, base_model=base_model, workspace=workspace,
        data_path=data_path, output_dir=output_dir,
        max_iterations=max_iterations, training_timeout=training_timeout,
        max_agent_steps=max_agent_steps, opencode_model=opencode_model,
        resume=resume,
    )

    task_description = ""
    for fname in ["description.md", "instructions.md"]:
        fpath = Path(workspace) / fname
        if fpath.exists():
            task_description += fpath.read_text() + "\n\n"

    if not task_description.strip():
        task_description = f"Benchmark: {task}\nTrain a language model using GRPO reinforcement learning.\n"
        console.print(f"  [bold yellow]WARNING:[/] No description.md found, using fallback: {task}")

    data_stats = get_data_stats(data_path)
    gpu_info = get_gpu_info()
    print_data_gpu_info(data_stats["count"], gpu_info["num_gpus"], gpu_info["gpu_name"])

    # ----- 构建历史（从 state.iterations 转换）-----
    history: list[IterationResult] = []
    for it in state.iterations:
        if it.current_phase == Phase.COMPLETE.value:
            history.append(IterationResult(
                iteration=it.iteration,
                exit_code=it.exit_code,
                training_time=it.training_time,
                score=it.score,
                model_path=it.model_path,
                code_path=it.code_path,
                analysis=it.analysis,
                improvement=it.improvement,
            ))

    # ----- 确定起始迭代 -----
    start_iteration = state.current_iteration + 1 if state.current_iteration > 0 else 1

    # 如果 resume，检查是否有未完成的迭代
    resume_phase: str | None = None
    if resume and state.iterations:
        last_iter = state.iterations[-1]
        if last_iter.current_phase != Phase.COMPLETE.value:
            # 从中断的阶段继续
            start_iteration = last_iter.iteration
            resume_phase = last_iter.current_phase
            console.print(f"  [green]Resuming iteration {start_iteration} from phase: {resume_phase}[/]")

    best_score = state.best_score
    best_iteration = state.best_iteration

    for iteration in range(start_iteration, max_iterations + 1):
        iter_start = time.time()
        elapsed_total = iter_start - pipeline_start

        print_iteration_header(iteration, max_iterations, elapsed_total)

        # 获取或创建 IterationState
        if resume_phase and state.iterations and state.iterations[-1].iteration == iteration:
            iter_state = state.iterations[-1]
        else:
            iter_state = IterationState(iteration=iteration)
            state.iterations.append(iter_state)

        state.current_iteration = iteration

        # 确定起始阶段
        phases_order = [
            Phase.CODE_GEN, Phase.TRAINING, Phase.EVALUATION,
            Phase.ANALYSIS, Phase.COMPLETE,
        ]

        if resume_phase:
            try:
                start_phase_idx = [p.value for p in phases_order].index(resume_phase)
            except ValueError:
                start_phase_idx = 0
            resume_phase = None  # 仅首轮有效
        else:
            start_phase_idx = 0

        # ---- 状态机循环 ----
        for phase_idx in range(start_phase_idx, len(phases_order)):
            phase = phases_order[phase_idx]
            iter_state.current_phase = phase.value

            if phase == Phase.CODE_GEN:
                result = _run_phase_code_gen(
                    state, iter_state, history, task_description,
                    gpu_info, opencode_model, opencode_url,
                    stale_timeout=stale_timeout,
                    http_timeout=http_timeout,
                    task_type=task_type,
                    expose_files=expose_files,
                )
                iter_state.phase_results["code_gen"] = result.to_dict()
                save_checkpoint(state)

                if not result.success:
                    console.print(
                        f"  [red]Code generation failed after"
                        f" {state.max_retries} attempt(s):[/] {result.error}"
                    )
                    iter_state.current_phase = Phase.COMPLETE.value
                    save_checkpoint(state)
                    break

            elif phase == Phase.TRAINING:
                result = _run_phase_training(
                    state, iter_state, opencode_model, opencode_url,
                    stale_timeout=stale_timeout,
                    http_timeout=http_timeout,
                )
                iter_state.phase_results["training"] = result.to_dict()
                save_checkpoint(state)

                if not result.success:
                    console.print(f"  [red]Training failed after all retries[/]")
                    iter_state.current_phase = Phase.COMPLETE.value
                    save_checkpoint(state)
                    break

            elif phase == Phase.EVALUATION:
                result = _run_phase_evaluation(state, iter_state)
                iter_state.phase_results["evaluation"] = result.to_dict()

                if result.success and iter_state.score is not None:
                    eval_score = iter_state.score
                    print_evaluation_report(
                        score=eval_score,
                        improvement=iter_state.improvement,
                        best_score=iter_state.best_score,
                        submission_id=iter_state.submission_id,
                    )

                    if best_score is None or eval_score > best_score:
                        best_score = eval_score
                        best_iteration = iteration
                else:
                    if not result.success:
                        console.print(f"  [red]Evaluation failed:[/] {result.error}")
                    else:
                        print_phase_status("No score available", "dim")

                state.best_score = best_score
                state.best_iteration = best_iteration
                save_checkpoint(state)

            elif phase == Phase.ANALYSIS:
                if iteration < max_iterations:
                    result = _run_phase_analysis(
                        state, iter_state, opencode_model, opencode_url,
                        stale_timeout=stale_timeout,
                        http_timeout=http_timeout,
                    )
                    iter_state.phase_results["analysis"] = result.to_dict()
                    save_checkpoint(state)

            elif phase == Phase.COMPLETE:
                iter_state.current_phase = Phase.COMPLETE.value
                save_checkpoint(state)

        # ---- 迭代结束 ----
        # 追加到 history
        history.append(IterationResult(
            iteration=iter_state.iteration,
            exit_code=iter_state.exit_code,
            training_time=iter_state.training_time,
            score=iter_state.score,
            model_path=iter_state.model_path,
            code_path=iter_state.code_path,
            analysis=iter_state.analysis,
            improvement=iter_state.improvement,
        ))

        iter_elapsed = time.time() - iter_start
        print_iteration_summary(
            iteration=iteration,
            score=iter_state.score,
            improvement=iter_state.improvement,
            best_score=best_score,
            best_iteration=best_iteration,
            elapsed=iter_elapsed,
        )

    # ----- Pipeline 完成 -----
    total_time = time.time() - pipeline_start
    print_pipeline_footer(best_score, best_iteration,
                          len(state.iterations), total_time)

    summary = {
        "task": task,
        "base_model": base_model,
        "best_score": best_score,
        "best_iteration": best_iteration,
        "total_time": total_time,
        "iterations": [
            {
                "iteration": it.iteration,
                "exit_code": it.exit_code,
                "training_time": it.training_time,
                "score": it.score,
                "improvement": it.improvement,
            }
            for it in state.iterations
        ],
    }
    summary_path = Path(workspace) / "pipeline_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    console.print(f"  [dim]Results saved:[/] {summary_path}")

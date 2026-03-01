"""Pipeline 数据类型定义。

包含状态机阶段枚举、各阶段结果类型、
迭代状态和全局 Pipeline 状态。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Phase 枚举
# ---------------------------------------------------------------------------
class Phase(str, enum.Enum):
    """Pipeline 阶段枚举。"""
    CODE_GEN = "code_gen"              # Agent 编写 train.py
    TRAINING = "training"              # 执行训练
    EVALUATION = "evaluation"          # Grading Server 评测
    ANALYSIS = "analysis"              # Agent 自分析诊断
    COMPLETE = "complete"              # 本轮完成


# ---------------------------------------------------------------------------
# PhaseResult：所有阶段函数的统一返回类型
# ---------------------------------------------------------------------------
@dataclass
class PhaseResult:
    """阶段执行结果。"""
    success: bool
    phase: str = ""
    payload: dict = field(default_factory=dict)
    error: str = ""

    # 这些 payload 键不序列化到 checkpoint（体积太大）
    _PAYLOAD_EXCLUDE_KEYS = {"stdout"}

    def to_dict(self) -> dict:
        d: dict = {"success": self.success, "phase": self.phase}
        if self.payload:
            filtered = {k: v for k, v in self.payload.items()
                        if k not in self._PAYLOAD_EXCLUDE_KEYS}
            if filtered:
                d["payload"] = filtered
        if self.error:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PhaseResult:
        return cls(
            success=d.get("success", False),
            phase=d.get("phase", ""),
            payload=d.get("payload", {}),
            error=d.get("error", ""),
        )


# ---------------------------------------------------------------------------
# IterationState：每轮迭代的状态
# ---------------------------------------------------------------------------
@dataclass
class IterationState:
    """一轮迭代的完整状态。"""
    iteration: int
    current_phase: str = Phase.CODE_GEN.value
    score: float | None = None
    exit_code: int = -1
    stdout: str = ""
    training_time: float = 0.0
    model_path: str = ""
    code_path: str = ""
    analysis: str = ""
    phase_results: dict[str, dict] = field(default_factory=dict)
    improvement: float | None = None
    best_score: float | None = None
    submission_id: int | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "iteration": self.iteration,
            "current_phase": self.current_phase,
            "phase_results": self.phase_results,
        }
        if self.score is not None:
            d["score"] = self.score
        if self.exit_code != -1:
            d["exit_code"] = self.exit_code
        if self.training_time > 0:
            d["training_time"] = self.training_time
        if self.model_path:
            d["model_path"] = self.model_path
        if self.code_path:
            d["code_path"] = self.code_path
        if self.analysis:
            d["analysis"] = self.analysis
        if self.improvement is not None:
            d["improvement"] = self.improvement
        if self.best_score is not None:
            d["best_score"] = self.best_score
        if self.submission_id is not None:
            d["submission_id"] = self.submission_id
        # 注意：stdout 不序列化（可达几十 MB），已单独保存为 training_stdout.log
        return d

    @classmethod
    def from_dict(cls, d: dict) -> IterationState:
        return cls(
            iteration=d["iteration"],
            current_phase=d.get("current_phase", Phase.CODE_GEN.value),
            score=d.get("score"),
            exit_code=d.get("exit_code", -1),
            training_time=d.get("training_time", 0.0),
            model_path=d.get("model_path", ""),
            code_path=d.get("code_path", ""),
            analysis=d.get("analysis", ""),
            phase_results=d.get("phase_results", {}),
            improvement=d.get("improvement"),
            best_score=d.get("best_score"),
            submission_id=d.get("submission_id"),
        )


# ---------------------------------------------------------------------------
# PipelineState：全局状态，可完全序列化
# ---------------------------------------------------------------------------
@dataclass
class PipelineState:
    """Pipeline 全局状态，支持序列化/反序列化以实现断点续跑。"""
    # 配置（不可变）
    task: str = ""
    base_model: str = ""
    workspace: str = ""
    data_path: str = ""
    output_dir: str = ""
    max_iterations: int = 5
    training_timeout: int = 3600
    max_agent_steps: int = 25
    max_retries: int = 3  # 各阶段（code_gen / fix / analysis）失败后自动重试次数
    eval_timeout: int = 600  # Grading Server 评测请求超时秒数

    # 运行时状态
    current_iteration: int = 0
    best_score: float | None = None
    best_iteration: int = -1
    pipeline_start_time: float = 0.0

    # 迭代历史
    iterations: list[IterationState] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "base_model": self.base_model,
            "workspace": self.workspace,
            "data_path": self.data_path,
            "output_dir": self.output_dir,
            "max_iterations": self.max_iterations,
            "training_timeout": self.training_timeout,
            "max_agent_steps": self.max_agent_steps,
            "max_retries": self.max_retries,
            "eval_timeout": self.eval_timeout,
            "current_iteration": self.current_iteration,
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "iterations": [it.to_dict() for it in self.iterations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> PipelineState:
        state = cls(
            task=d.get("task", ""),
            base_model=d.get("base_model", ""),
            workspace=d.get("workspace", ""),
            data_path=d.get("data_path", ""),
            output_dir=d.get("output_dir", ""),
            max_iterations=d.get("max_iterations", 5),
            training_timeout=d.get("training_timeout", 3600),
            max_agent_steps=d.get("max_agent_steps", 25),
            max_retries=d.get("max_retries", d.get("max_fix_retries", 3)),
            eval_timeout=d.get("eval_timeout", 600),
            current_iteration=d.get("current_iteration", 0),
            best_score=d.get("best_score"),
            best_iteration=d.get("best_iteration", -1),
        )
        state.iterations = [
            IterationState.from_dict(it) for it in d.get("iterations", [])
        ]
        return state


# ---------------------------------------------------------------------------
# 向后兼容：IterationResult（供外部消费者使用）
# ---------------------------------------------------------------------------
@dataclass
class IterationResult:
    """一轮迭代的完整结果（向后兼容接口）。"""
    iteration: int
    exit_code: int = -1
    stdout: str = ""
    training_time: float = 0.0
    score: float | None = None
    model_path: str = ""
    code_path: str = ""
    analysis: str = ""
    improvement: float | None = None

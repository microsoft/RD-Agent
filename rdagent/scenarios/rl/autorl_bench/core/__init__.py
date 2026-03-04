"""
AutoRL-Bench Core Module

主干代码，定义统一的评测接口和服务。
开发新 benchmark 或 agent 时不需要修改此模块。

================================================================================
面向开发者的接口约定
================================================================================

评测器基类: BaseEvaluator (evaluator.py)
    所有 benchmark 评测器继承此类并实现 run_eval 方法。

    def run_eval(
        self,
        model_path: str,          # 训练后的模型路径（本地目录）
        workspace_path: str,       # 工作目录路径
        model_name: str = "",      # 模型名称（用于配置推理参数）
        gpu_count: int = 1,        # 可用 GPU 数量
        test_range: str = "[:]",   # 测试数据范围
        **kwargs,
    ) -> EvalResult

评测结果: EvalResult (evaluator.py)
    TypedDict，必须字段: benchmark, model_path, score, accuracy_summary

具体实现:
    - OpenCompassEvaluator (opencompass.py)  — 基于 OpenCompass 的评测
    - PerSampleEvaluator (benchmarks/smith/) — 逐样本评测

服务:
    - GradingServer (server.py)              — 评测服务器
    - create_grading_server (server.py)      — 创建服务上下文管理器
================================================================================
"""
from .evaluator import (
    BaseEvaluator,
    EvalResult,
)
from .opencompass import OpenCompassEvaluator
from .server import create_grading_server
from .utils import (
    ensure_symlink,
    download_model,
    download_data,
    get_baseline_score,
    submit_to_grading_server,
    set_baseline_to_server,
    setup_workspace,
    append_result,
    detect_driver_model,
    print_summary,
    kill_process_group,
)

__all__ = [
    # 数据结构
    "EvalResult",
    # 评测器
    "BaseEvaluator",
    "OpenCompassEvaluator",
    # 服务
    "create_grading_server",
    # 工具函数
    "ensure_symlink",
    "download_model",
    "download_data",
    "get_baseline_score",
    "submit_to_grading_server",
    "set_baseline_to_server",
    # workspace & results
    "setup_workspace",
    "append_result",
    "detect_driver_model",
    "print_summary",
    "kill_process_group",
]

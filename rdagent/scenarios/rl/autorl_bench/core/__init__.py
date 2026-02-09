"""
AutoRL-Bench Core Module

主干代码，定义统一的评测接口和服务。
开发新 benchmark 或 agent 时不需要修改此模块。
"""
from .evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
)
from .opencompass import OpenCompassEvaluator
from .utils import (
    download_model,
    download_data,
    get_baseline_score,
    submit_to_grading_server,
    set_baseline_to_server,
)

__all__ = [
    # 数据结构
    "EvalInput",
    "EvalResult",
    # 评测器
    "BaseEvaluator",
    "OpenCompassEvaluator",
    # 工具函数
    "download_model",
    "download_data",
    "get_baseline_score",
    "submit_to_grading_server",
    "set_baseline_to_server",
]

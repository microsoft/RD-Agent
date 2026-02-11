"""
AutoRL-Bench Core Module

主干代码，定义统一的评测接口和服务。
开发新 benchmark 或 agent 时不需要修改此模块。

================================================================================
面向开发者的接口约定（输入 / 输出 / 环节）
================================================================================
Put all the branches and complexity during initialization.

Config:
- eval_class
    - uri: rdagent.scenarios.rl.autorl_bench.core.OpenCompassEvaluator
    - kwargs:
        config: BenchmarkConfig   # 见 benchmarks/__init__.py
        ...

RESTful API  -- (...) -> def run_eval(...):

class BenchmarkBase:
    '''
    所有 benchmark 评测器的基类（实际代码见 evaluator.py 中的 BaseEvaluator）
    '''
    def run_eval(self, workspace_path: str, model_path: str, task_config) -> dict:
        '''
        输入:
            workspace_path: 工作目录路径
            model_path:     训练后的模型路径（本地目录）
            task_config:    任务配置（模型名称、GPU 数量、测试范围等）

        输出 (dict):
            benchmark:        str           # benchmark 名称
            model_path:       str           # 评测的模型路径
            score:            float         # 评测分数 (0-100)
            accuracy_summary: Dict[str,Any] # 详细指标

        副作用 (side-effects):
            - 在 workspace_path 下生成评测结果文件
            - 日志输出到 logger
        '''
        ...

class DRBenchmark(BenchmarkBase):
    '''具体实现示例（如 OpenCompassEvaluator、ALFWorldEvaluator）'''
    def run_eval(self, workspace_path: str, model_path: str, task_config) -> dict:
        '''调用具体 benchmark 的评测逻辑，返回统一格式的结果 dict'''
        ...

================================================================================
"""
from .evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalResult,
)
from .opencompass import OpenCompassEvaluator
from .utils import (
    ensure_symlink,
    download_model,
    download_data,
    get_baseline_score,
    submit_to_grading_server,
    set_baseline_to_server,
    create_grading_server,
    setup_workspace,
    append_result,
    detect_driver_model,
    print_summary,
)

__all__ = [
    # 数据结构
    "EvalInput",
    "EvalResult",
    # 评测器
    "BaseEvaluator",
    "OpenCompassEvaluator",
    # 工具函数
    "ensure_symlink",
    "download_model",
    "download_data",
    "get_baseline_score",
    "submit_to_grading_server",
    "set_baseline_to_server",
    "create_grading_server",
    # workspace & results
    "setup_workspace",
    "append_result",
    "detect_driver_model",
    "print_summary",
]

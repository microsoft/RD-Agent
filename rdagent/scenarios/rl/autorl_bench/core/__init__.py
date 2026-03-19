"""
AutoRL-Bench Core Module

Backbone code defines unified evaluation interfaces and services.
There is no need to modify this module when developing new benchmarks or agents.

================================================================================
Interface conventions for developers
================================================================================

Evaluator base class: BaseEvaluator (evaluator.py)
All benchmark evaluators inherit this class and implement the run_eval method.

    def run_eval(
        self,
model_path: str, # trained model path (local directory)
workspace_path: str, # Working directory path
model_name: str = "", # Model name (used to configure inference parameters)
gpu_count: int = 1, # Number of available GPUs
test_range: str = "[:]", # Test data range
        **kwargs,
    ) -> EvalResult

Evaluation results: EvalResult (evaluator.py)
TypedDict, required fields: benchmark, model_path, score, accuracy_summary

Specific implementation:
- OpenCompassEvaluator (opencompass.py) — OpenCompass-based evaluation
- PerSampleEvaluator (benchmarks/smith/) — sample-by-sample evaluation

Serve:
- GradingServer (server.py) — Grading server
- create_grading_server (server.py) — Create a service context manager
================================================================================
"""

from .evaluator import (
    BaseEvaluator,
    EvalResult,
)
from .metrics import run_workspace_metrics
from .opencompass import OpenCompassEvaluator
from .server import create_grading_server
from .utils import (
    append_result,
    detect_driver_model,
    download_data,
    download_model,
    ensure_symlink,
    get_baseline_score,
    init_run_meta,
    kill_process_group,
    print_summary,
    read_run_meta,
    set_baseline_to_server,
    setup_workspace,
    submit_to_grading_server,
    update_run_meta,
)

__all__ = [
# Data structure
    "EvalResult",
# evaluator
    "BaseEvaluator",
    "OpenCompassEvaluator",
# Serve
    "create_grading_server",
# Utility function
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
    "init_run_meta",
    "update_run_meta",
    "read_run_meta",
    "run_workspace_metrics",
]

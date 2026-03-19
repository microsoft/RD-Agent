"""
AutoRL-Bench Evaluator Base Class

The base class of all benchmark evaluators, defining a unified evaluation interface.

When developing a new benchmark, inherit BaseEvaluator and implement the run_eval method.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from typing_extensions import NotRequired, TypedDict

# ============================================================
#Data structure definition (Schema)
# ============================================================


class EvalResult(TypedDict):
    """
Evaluation output results

Required fields:
benchmark: benchmark name
model_path: model path for evaluation
score: evaluation score (0-100)
accuracy_summary: Detailed indicator dictionary

Optional fields:
eval_type: evaluation type ("opencompass" / "alfworld" / ...)
error: error message (when the evaluation fails)
raw_output: raw output log
    """

#Required fields
    benchmark: str
    model_path: str
    score: float
    accuracy_summary: Dict[str, Any]

# Optional fields
    eval_type: NotRequired[str]
    error: NotRequired[str]
    raw_output: NotRequired[str]


# ============================================================
#Abstract base class
# ============================================================


class BaseEvaluator(ABC):
    """
Benchmark evaluator base class

All custom benchmarks must inherit this class and implement the run_eval method.

    =====================================================
The simplest way: call benchmark’s own evaluation code
    =====================================================

Most benchmarks (such as HumanEval, MBPP, ALFWorld) have official evaluation scripts,
Just need:
1. Download benchmark repo
2. Call its evaluation function
3. Convert the result to EvalResult format

Example (wrapping an existing review):
        class MyBenchmarkEvaluator(BaseEvaluator):
            def __init__(self, config):
                self.config = config
                self.benchmark_id = config.id

            def run_eval(self, model_path, workspace_path, **kwargs) -> EvalResult:
                result = self.get_default_result(self.benchmark_id, model_path)

# 1. Call benchmark’s own evaluation
from some_benchmark import evaluate # benchmark official library
raw_result = evaluate(model_path) # Call the official evaluation

# 2. Convert to unified format
                result["score"] = raw_result["accuracy"] * 100
                result["accuracy_summary"] = raw_result
                return result

    =====================================================
Complete example: Custom evaluation logic
    =====================================================

If you need to fully customize the assessment (such as an interactive environment):

    Example:
        class InteractiveEvaluator(BaseEvaluator):
            def run_eval(self, model_path, workspace_path, **kwargs) -> EvalResult:
                result = self.get_default_result(self.benchmark_id, model_path)

# 1. Load model
                model = load_model(model_path)

# 2. Run the evaluation loop
                success = 0
                for task in tasks:
                    output = model.generate(task.prompt)
                    if task.check(output):
                        success += 1

# 3. Return results
                result["score"] = success / len(tasks) * 100
                result["accuracy_summary"] = {"success": success, "total": len(tasks)}
                return result
    """

    @abstractmethod
    def run_eval(
        self,
        model_path: str,
        workspace_path: str,
        model_name: str = "",
        gpu_count: int = 1,
        test_range: str = "[:]",
        **kwargs,
    ) -> EvalResult:
        """
Perform evaluation

        Args:
model_path: trained model path (local directory)
workspace_path: working directory path
model_name: model name (used to configure inference parameters)
gpu_count: Number of available GPUs
test_range: test data range, such as "[:]" or "[:100]"
**kwargs: other evaluation parameters

        Returns:
EvalResult: evaluation result
        """
        pass

    def validate_model(self, model_path: str) -> bool:
"""Verify whether the model path is valid"""
        return Path(model_path).exists()

    def get_default_result(self, benchmark_name: str, model_path: str) -> EvalResult:
"""Return to the default result structure"""
        return {
            "benchmark": benchmark_name,
            "model_path": model_path,
            "score": 0.0,
            "accuracy_summary": {},
        }

"""
Simple test script for benchmark evaluation.

This script demonstrates how to use the FTBenchmarkEvaluator independently.
"""

from pathlib import Path

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.finetune.train.benchmark import FTBenchmarkEvaluator


def test_benchmark_quick():
    """
    Quick test with limited samples.

    This is useful for verifying the evaluation pipeline works without
    running a full benchmark (which can take hours).
    """
    print("=" * 60)
    print("Quick Benchmark Test (Limited Samples)")
    print("=" * 60)

    # Create scenario
    scen = LLMFinetuneScen()

    # Create evaluator with limited samples
    evaluator = FTBenchmarkEvaluator(
        scen=scen,
        tasks=["gsm8k"],  # Single task for quick testing
        limit=10,  # Only 10 samples
    )

    # Create mock workspace with adapter files
    # In real usage, this would be the output from training
    workspace_path = Path("/path/to/your/workspace")
    workspace = FBWorkspace(workspace_path)

    # Create mock task
    task = Task(name="test_benchmark")

    print("\nRunning evaluation...")
    print(f"Tasks: {evaluator.tasks}")
    print(f"Limit: {evaluator.limit}")
    print(f"Base Model: {scen.base_model}")
    print()

    # Run evaluation
    feedback = evaluator.evaluate(
        target_task=task,
        implementation=workspace,
        gt_implementation=None,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Execution: {feedback.execution}")
    print(f"Return Checking:\n{feedback.return_checking}")
    print(f"Code: {feedback.code}")
    print(f"Final Decision: {feedback.final_decision}")
    print("=" * 60)


def test_benchmark_full():
    """
    Full benchmark test with multiple tasks.

    This runs complete evaluation on all configured tasks.
    Use this for final model evaluation.
    """
    print("=" * 60)
    print("Full Benchmark Test")
    print("=" * 60)

    # Create scenario
    scen = LLMFinetuneScen()

    # Create evaluator with default tasks from config
    evaluator = FTBenchmarkEvaluator(scen=scen)

    # Create workspace
    workspace_path = Path("/path/to/your/workspace")
    workspace = FBWorkspace(workspace_path)

    # Create task
    task = Task(name="test_benchmark_full")

    print("\nRunning full evaluation...")
    print(f"Tasks: {evaluator.tasks}")
    print(f"Base Model: {scen.base_model}")
    print()

    # Run evaluation
    feedback = evaluator.evaluate(
        target_task=task,
        implementation=workspace,
        gt_implementation=None,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Execution: {feedback.execution}")
    print(f"Return Checking:\n{feedback.return_checking}")
    print(f"Code: {feedback.code}")
    print(f"Final Decision: {feedback.final_decision}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "full":
        test_benchmark_full()
    else:
        print("Usage:")
        print("  python test_benchmark.py        # Quick test (10 samples)")
        print("  python test_benchmark.py full   # Full test (all samples)")
        print()
        test_benchmark_quick()

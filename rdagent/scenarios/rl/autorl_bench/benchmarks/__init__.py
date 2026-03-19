"""
AutoRL-Bench Benchmarks Registry

A registry that manages all available benchmark evaluators.
When adding a new benchmark, register it here.
"""

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type

from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator

BENCHMARKS_DIR = Path(__file__).parent


@dataclass
class BenchmarkConfig:
"""Benchmark configuration

The data download/processing logic of each benchmark is written in data.py in the respective directory.
It is not dealt with uniformly here. In this way, when adding a benchmark, you only need to implement it in your own directory.
    """

    id: str
evaluator_class: str # Full path to evaluator class
data_module: str = "" #Data module path (implement download_train_data function)
    description: str = ""
    eval_config: Optional[Dict[str, Any]] = field(default=None)
    expose_files: list = field(
        default_factory=list
) # Benchmark-specific additional files (description.md and instructions.md are mounted uniformly by run.py)
bench_dir: Optional[str] = None # Customize the benchmark directory path (default None, use BENCHMARKS_DIR/id)


#Benchmark Registry
BENCHMARKS: Dict[str, BenchmarkConfig] = {
    "gsm8k": BenchmarkConfig(
        id="gsm8k",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.core.opencompass.OpenCompassEvaluator",
        data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.gsm8k.data",
description="Grade School Math 8K - Elementary Mathematical Reasoning",
        eval_config={
            "dataset": "opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4",
        },
    ),
    "humaneval": BenchmarkConfig(
        id="humaneval",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.core.opencompass.OpenCompassEvaluator",
        data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.humaneval.data",
description="HumanEval - Python code generation",
        eval_config={
            "dataset": "opencompass.configs.datasets.humaneval.humaneval_gen",
            "test_range": "[82:]",
        },
    ),
    "alpacaeval": BenchmarkConfig(
        id="alpacaeval",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.benchmarks.alpacaeval.eval.AlpacaEvalEvaluator",
        data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.alpacaeval.data",
description="AlpacaEval 2.0 - Instruction Compliance and Preference Evaluation (LLM Judge)",
        eval_config={
            "reference_file": "alpaca_eval_gpt4_baseline.json",
            "annotators_config": "annotators_gpt52_fn",
            "max_model_len": 4096,
            "max_tokens": 512,
        },
        expose_files=["eval.py"],
    ),
    "alfworld": BenchmarkConfig(
        id="alfworld",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.benchmarks.alfworld.eval.ALFWorldEvaluator",
        data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.alfworld.data",
description="ALFWorld - text game interactive environment (ReAct agent, supports vLLM/API)",
        eval_config={
            "max_steps": 50,
"env_num": 134, # Complete evaluation set (valid_unseen), set to 1 during previous debugging
        },
        expose_files=["eval.py"],
    ),
    "webshop": BenchmarkConfig(
        id="webshop",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.benchmarks.webshop.eval.WebShopEvaluator",
        data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.webshop.data",
description="WebShop - Online shopping website interactive environment (ReAct agent, supports vLLM/API)",
        eval_config={
            "max_steps": 50,
            "num_instructions": 100,
            "webshop_port": 8080,
        },
        expose_files=["eval.py"],
    ),
    "deepsearchqa": BenchmarkConfig(
        id="deepsearchqa",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.benchmarks.deepsearchqa.eval.DeepSearchQAEvaluator",
        data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.deepsearchqa.data",
description="DeepSearchQA - Google DeepMind multi-step information retrieval benchmark (900 questions, 17 domains)",
        eval_config={
            "num_samples": 200,  # fixed held-out evaluation split after 100/200 train/eval partition
"max_steps": 6, # ReAct maximum search rounds
# api_key": "...", # Optional, if left blank, use DuckDuckGo
        },
        expose_files=["eval.py"],
    ),
}


from rdagent.scenarios.rl.autorl_bench.benchmarks.smith import discover_smith_benchmarks

BENCHMARKS.update(discover_smith_benchmarks())


def get_benchmark(benchmark_id: str) -> BenchmarkConfig:
"""Get benchmark configuration"""
    if benchmark_id not in BENCHMARKS:
        available = list(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {available}")
    return BENCHMARKS[benchmark_id]


def get_evaluator(benchmark_id: str) -> BaseEvaluator:
"""Get the evaluator instance of benchmark"""
    config = get_benchmark(benchmark_id)

# Dynamically import evaluator class
    module_path, class_name = config.evaluator_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    evaluator_class: Type[BaseEvaluator] = getattr(module, class_name)

    return evaluator_class(config)


def list_benchmarks() -> list[str]:
"""List all available benchmarks"""
    return list(BENCHMARKS.keys())

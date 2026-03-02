"""
AutoRL-Bench Benchmarks Registry

注册表，管理所有可用的 benchmark 评测器。
添加新 benchmark 时，在此注册。
"""
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type

from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator


BENCHMARKS_DIR = Path(__file__).parent


@dataclass
class BenchmarkConfig:
    """Benchmark 配置

    每个 benchmark 的数据下载/处理逻辑写在各自目录的 data.py 里，
    不在这里统一处理。这样新增 benchmark 时只需在自己目录下实现即可。
    """
    id: str
    evaluator_class: str  # 评测器类的完整路径
    data_module: str = ""  # 数据模块路径（实现 download_train_data 函数）
    description: str = ""
    eval_config: Optional[Dict[str, Any]] = field(default=None)
    expose_files: list = field(default_factory=list)  # benchmark 特有的额外文件（description.md 和 instructions.md 由 run.py 统一挂载）
    bench_dir: Optional[str] = None  # 自定义 benchmark 目录路径（默认 None 则用 BENCHMARKS_DIR / id）


# Benchmark 注册表
BENCHMARKS: Dict[str, BenchmarkConfig] = {
    "gsm8k": BenchmarkConfig(
        id="gsm8k",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.core.opencompass.OpenCompassEvaluator",
        data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.gsm8k.data",
        description="Grade School Math 8K - 小学数学推理",
        eval_config={
            "dataset": "opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4",
        },
    ),
    "alfworld": BenchmarkConfig(
        id="alfworld",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.benchmarks.alfworld.eval.ALFWorldEvaluator",
        data_module="rdagent.scenarios.rl.autorl_bench.benchmarks.alfworld.data",
        description="ALFWorld - 文本游戏交互环境（ReAct agent，支持 vLLM/API）",
        eval_config={
            "max_steps": 50,
            "env_num": 134,  # 完整评测集（valid_unseen），之前调试时设为 1
        },
        expose_files=["eval.py", "react_prompts.json"],
    ),
}


from rdagent.scenarios.rl.autorl_bench.benchmarks.smith import discover_smith_benchmarks
BENCHMARKS.update(discover_smith_benchmarks())


def get_benchmark(benchmark_id: str) -> BenchmarkConfig:
    """获取 benchmark 配置"""
    if benchmark_id not in BENCHMARKS:
        available = list(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {available}")
    return BENCHMARKS[benchmark_id]


def get_evaluator(benchmark_id: str) -> BaseEvaluator:
    """获取 benchmark 的评测器实例"""
    config = get_benchmark(benchmark_id)
    
    # 动态导入评测器类
    module_path, class_name = config.evaluator_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    evaluator_class: Type[BaseEvaluator] = getattr(module, class_name)
    
    return evaluator_class(config)


def list_benchmarks() -> list[str]:
    """列出所有可用的 benchmark"""
    return list(BENCHMARKS.keys())

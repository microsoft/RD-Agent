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
    """Benchmark 配置"""
    id: str
    evaluator_class: str  # 评测器类的完整路径
    data_source: str  # 数据来源（HuggingFace dataset 或 Git repo）
    description: str = ""
    eval_config: Optional[Dict[str, Any]] = field(default=None)
    expose_files: list = field(default_factory=list)  # benchmark 特有的额外文件（description.md 和 instructions.md 由 run.py 统一挂载）
    use_docker: bool = False  # 是否使用 Docker 环境评测
    docker_image: str = ""  # Docker 镜像名


# Benchmark 注册表
BENCHMARKS: Dict[str, BenchmarkConfig] = {
    # ============================================================
    # OpenCompass 类 benchmark（只需配置，不需要 eval.py）
    # ============================================================
    "gsm8k": BenchmarkConfig(
        id="gsm8k",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.core.opencompass.OpenCompassEvaluator",
        data_source="openai/gsm8k",
        description="Grade School Math 8K - 小学数学推理",
        eval_config={
            "dataset": "opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4",
        },
        # description.md 和 instructions.md 由 run.py 统一挂载，无需在此声明
    ),
    # "math": BenchmarkConfig(
    #     id="math",
    #     evaluator_class="rdagent.scenarios.rl.autorl_bench.core.opencompass.OpenCompassEvaluator",
    #     data_source="lighteval/MATH",
    #     description="MATH - 竞赛级数学题",
    #     eval_config={
    #         "dataset": "opencompass.configs.datasets.math.math_0shot_gen_393424",
    #     },
    # ),
    
    # ============================================================
    # 自定义评测 benchmark（需要 eval.py）
    # ============================================================
    "alfworld": BenchmarkConfig(
        id="alfworld",
        evaluator_class="rdagent.scenarios.rl.autorl_bench.benchmarks.alfworld.eval.ALFWorldEvaluator",
        data_source="https://github.com/alfworld/alfworld.git",
        description="ALFWorld - 文本游戏交互环境（ReAct agent，支持 vLLM/API）",
        eval_config={
            "max_steps": 50,
            "env_num": 1,  # 默认评测 1 局（完整评测改为 134）
        },
        expose_files=["eval.py", "react_prompts.json"],  # benchmark 特有文件
        use_docker=False,  # 本地已安装 alfworld；生产环境可改为 True 用 Docker
        docker_image="autorl-bench/alfworld:latest",
    ),
}


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

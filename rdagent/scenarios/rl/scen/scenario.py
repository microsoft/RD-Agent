"""
RL Post-training Scenario

参考 SFT: rdagent/scenarios/finetune/scen/scenario.py
"""

import json
import shutil
from pathlib import Path

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.conf import get_autorl_bench_dir
from rdagent.scenarios.rl.env.conf import get_rl_benchmark_env
from rdagent.scenarios.rl.experiment.workspace import RLWorkspace
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env


class RLPostTrainingScen(Scenario):
    """RL Post-training Scenario
    
    初始化时自动运行 baseline 评测（有缓存则跳过）
    """

    def __init__(self) -> None:
        """Initialize RL post-training scenario"""
        logger.info("Initializing RL Post-training scenario")

        # 基础属性
        self.base_model = RL_RD_SETTING.base_model
        self.benchmark = RL_RD_SETTING.benchmark

        if not self.base_model:
            raise ValueError("base_model is required. Use --base-model or set RL_RD_SETTING.base_model")
        if not self.benchmark:
            raise ValueError("benchmark is required. Use --benchmark or set RL_RD_SETTING.benchmark")

        logger.info(f"  Base model: {self.base_model}")
        logger.info(f"  Benchmark: {self.benchmark}")

        # 下载数据和模型（内部有存在检查）
        from rdagent.scenarios.rl.autorl_bench.utils.download import download_data, download_model
        from rdagent.scenarios.rl.env.conf import RL_DATA_DIR, RL_MODELS_DIR
        logger.info("Checking and downloading resources...")
        download_data(self.benchmark, str(RL_DATA_DIR))
        download_model(self.base_model, str(RL_MODELS_DIR))

        # 获取 GPU 信息
        try:
            env = get_rl_benchmark_env()
            self.device_info = get_runtime_environment_by_env(env)
            self.gpu_count = json.loads(self.device_info).get("gpu_count", 1)
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}, using default gpu_count=1")
            self.device_info = '{"gpu_count": 1}'
            self.gpu_count = 1

        logger.info(f"  GPU count: {self.gpu_count}")

        # 运行 baseline 评测（有缓存则跳过）
        logger.info("Running baseline evaluation...")
        baseline_result = self.run_baseline_model_evaluation(
            model_name=self.base_model,
            benchmark_name=self.benchmark,
        )
        self.baseline_benchmark_score = baseline_result.get("benchmark", {})
        logger.info(f"  Baseline score: {self.baseline_benchmark_score}")

        # 读取任务描述
        task_desc_file = get_autorl_bench_dir() / "tasks" / self.benchmark / "description.md"
        if task_desc_file.exists():
            self.task_description = task_desc_file.read_text()
            logger.info(f"  Loaded task description from {task_desc_file}")
        else:
            self.task_description = ""
            logger.warning(f"  Task description not found: {task_desc_file}")

    def benchmark_hash(self, model_name: str, benchmark_name: str) -> str:
        """缓存 key"""
        return f"rl_baseline_eval_{model_name}_{benchmark_name}"

    @cache_with_pickle(benchmark_hash)
    def run_baseline_model_evaluation(self, model_name: str, benchmark_name: str) -> dict:
        """
        运行 baseline 评测
        
        1. 创建临时 workspace
        2. 复制基础模型到 workspace 内
        3. 调用 run_benchmark() 评测
        4. 返回结果（会被缓存）
        """
        from rdagent.scenarios.rl.autorl_bench.benchmark import run_benchmark

        # 创建 workspace
        ws = RLWorkspace()
        ws.prepare()

        # 复制模型到 workspace 内（模型已在 __init__ 中下载）
        src_model_path = RL_RD_SETTING.file_path / "models" / model_name
        dst_model_path = ws.workspace_path / "models" / model_name

        logger.info(f"Copying model to workspace: {src_model_path} -> {dst_model_path}")
        dst_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_model_path, dst_model_path, dirs_exist_ok=True)

        # 运行全量评测
        logger.info(f"Running benchmark '{benchmark_name}' on baseline model (full dataset)...")
        result = run_benchmark(
            workspace_path=str(ws.workspace_path),
            model_path=str(dst_model_path),
            model_name=model_name,
            benchmark_name=benchmark_name,
            gpu_count=self.gpu_count,
            test_range="[:]",  # 全量评测
        )

        return {"benchmark": result}

    @property
    def background(self) -> str:
        """Background information for the agent"""
        background = f"""RL Post-training Scenario

Base Model: {self.base_model}
Benchmark: {self.benchmark}
Baseline Score: {self.baseline_benchmark_score}

Goal: Improve model performance on {self.benchmark} through RL post-training.
"""
        if self.task_description:
            background += f"\n## Task Description\n{self.task_description}"
        return background

    def get_runtime_environment(self) -> str:
        """Get runtime environment info"""
        return self.device_info

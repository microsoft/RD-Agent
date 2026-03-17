"""
RL Post-training Scenario

作为 autorl_bench 的 agent 运行时，run.py 已经完成了：
- 资源下载（模型、数据）
- workspace 创建 + 软链接
- Grading Server 启动 + baseline 评测
- 环境变量传递

本 Scenario 只需读取这些信息，不重复操作。
"""

import os
from pathlib import Path

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger


class RLPostTrainingScen(Scenario):
    """RL Post-training Scenario

    从 run.py 传递的环境变量中读取配置，不重复下载资源或评测 baseline。
    """

    def __init__(self) -> None:
        logger.info("Initializing RL Post-training scenario")

        # 从 env var 读取（run.py 已设置），CLI 参数作为 fallback
        self.base_model = os.environ.get("BASE_MODEL") or RL_RD_SETTING.base_model or ""
        self.benchmark = os.environ.get("TASK") or RL_RD_SETTING.benchmark or ""
        self.workspace = os.environ.get("WORKSPACE", "")
        self.model_path = os.environ.get("MODEL_PATH", "")
        self.data_path = os.environ.get("DATA_PATH", "")
        self.output_dir = os.environ.get("OUTPUT_DIR", "")
        self.grading_server_url = os.environ.get("GRADING_SERVER_URL", "")

        if not self.base_model:
            raise ValueError("BASE_MODEL env var or --base-model required")
        if not self.benchmark:
            raise ValueError("TASK env var or --benchmark required")

        logger.info(f"  Benchmark: {self.benchmark}")
        logger.info(f"  Base model: {self.base_model}")
        logger.info(f"  Workspace: {self.workspace}")
        logger.info(f"  Grading Server: {self.grading_server_url}")

        # 读取任务描述（workspace 里的 description.md，已由 run.py 软链接）
        desc_file = Path(self.workspace) / "description.md" if self.workspace else None
        if desc_file and desc_file.exists():
            self.task_description = desc_file.read_text()
            logger.info(f"  Loaded task description from {desc_file}")
        else:
            self.task_description = ""
            logger.warning("  Task description not found in workspace")

    @property
    def background(self) -> str:
        """Background information for LLM prompts"""
        bg = f"""RL Post-training Scenario

Benchmark: {self.benchmark}
Base Model: {self.base_model}
Model Path: {self.model_path}
Data Path: {self.data_path}
Output Dir: {self.output_dir}
Grading Server: {self.grading_server_url}

Goal: Improve model performance on {self.benchmark} through RL post-training.
Submit trained model via POST {self.grading_server_url}/submit for evaluation.
"""
        if self.task_description:
            bg += f"\n## Task Description\n{self.task_description}"
        return bg

    def get_runtime_environment(self) -> str:
        """Get runtime environment info"""
        return f'{{"workspace": "{self.workspace}", "grading_server": "{self.grading_server_url}"}}'

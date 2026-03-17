"""
RL Training Environment Configuration

autorl_bench 模式下，run.py 已完成环境搭建，不需要 Docker。
保留基础路径配置供其他模块引用。
"""

import os
from pathlib import Path

from rdagent.app.rl.conf import RL_RD_SETTING

# RL 资源路径（从 env var 优先，fallback 到 RL_RD_SETTING）
RL_MODELS_DIR = Path(os.environ.get("MODEL_PATH", str(RL_RD_SETTING.file_path / "models")))
RL_DATA_DIR = Path(os.environ.get("DATA_PATH", str(RL_RD_SETTING.file_path / "datasets")))

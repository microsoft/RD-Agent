"""
RL Runner - 执行训练代码并提交 Grading Server 评测

作为 autorl_bench agent 运行：
- 训练代码在本地执行（$WORKSPACE/code/ 下）
- 评测通过 HTTP POST $GRADING_SERVER_URL/submit
"""

import json
import os
import subprocess
import time
from pathlib import Path

import requests

from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger


class RLPostTrainingRunner(Developer):
    """RL Runner - 本地执行训练 + HTTP API 评测"""

    def __init__(self, scen: Scenario, timeout: int = 360000) -> None:
        self.scen = scen
        self.timeout = timeout

    def develop(self, exp: Experiment) -> Experiment:
        """
        执行训练代码并提交评测

        流程：
        1. 将生成的代码写入 $WORKSPACE/code/
        2. 本地执行 main.py
        3. POST $GRADING_SERVER_URL/submit 提交评测
        """
        workspace = exp.experiment_workspace
        if workspace is None or "main.py" not in workspace.file_dict:
            logger.warning("No main.py in experiment workspace, skipping")
            exp.result = {"exit_code": -1, "stdout": "No main.py generated"}
            return exp

        # 从 env var 读取路径（run.py 已设置）
        ws_dir = os.environ.get("WORKSPACE", "")
        output_dir = os.environ.get("OUTPUT_DIR", "")
        grading_url = os.environ.get("GRADING_SERVER_URL", "")

        if not ws_dir:
            logger.error("WORKSPACE env var not set")
            exp.result = {"exit_code": -1, "stdout": "WORKSPACE not set"}
            return exp

        code_dir = Path(ws_dir) / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        # 1. 将生成的代码写入 code/
        for filename, content in workspace.file_dict.items():
            dst = code_dir / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content)
            logger.info(f"  Wrote {dst}")

        # 2. 本地执行 main.py
        main_py = code_dir / "main.py"
        logger.info(f"=== Executing {main_py} ===")
        start_time = time.time()

        try:
            proc = subprocess.run(
                ["python", str(main_py)],
                cwd=str(code_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            exit_code = proc.returncode
            stdout = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as e:
            exit_code = -1
            stdout = f"Timeout after {self.timeout}s\n{e.stdout or ''}"
            logger.warning(f"Training timed out after {self.timeout}s")

        elapsed = time.time() - start_time
        logger.info(f"Training finished: exit_code={exit_code}, time={elapsed:.1f}s")

        if exit_code != 0:
            logger.warning(f"Training failed:\n{stdout[:2000]}")

        exp.result = {
            "exit_code": exit_code,
            "stdout": stdout,
            "running_time": elapsed,
            "benchmark": None,
        }

        # 3. 提交 Grading Server 评测
        if exit_code != 0 or not grading_url or not output_dir:
            return exp

        output_path = Path(output_dir)
        if not output_path.exists() or not any(output_path.iterdir()):
            logger.info("No model output found, skipping evaluation")
            return exp

        # 找到 output/ 下最新的模型目录（可能有 v1/, v2/ 等子目录）
        model_path = self._find_latest_model(output_path)
        logger.info(f"=== Submitting to Grading Server: {model_path} ===")

        try:
            resp = requests.post(
                f"{grading_url}/submit",
                json={"model_path": str(model_path)},
                timeout=600,
            )
            result = resp.json()
            exp.result["benchmark"] = result
            logger.info(
                f"  Score: {result.get('score')}, "
                f"Improvement: {result.get('improvement')}, "
                f"Best: {result.get('best', {}).get('score')}"
            )
        except Exception as e:
            logger.error(f"Grading server submission failed: {e}")

        return exp

    @staticmethod
    def _find_latest_model(output_dir: Path) -> Path:
        """找到 output/ 下的模型路径。

        如果有子目录（v1/, v2/ 等），返回最新修改的那个；
        否则返回 output/ 本身。
        """
        subdirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if subdirs:
            return max(subdirs, key=lambda d: d.stat().st_mtime)
        return output_dir

"""
RL Runner - Execute RL training code in Docker
"""

import hashlib
from pathlib import Path

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.env.conf import get_rl_env, RL_MODELS_DIR
from rdagent.scenarios.rl.autorl_bench.utils.grading import submit_to_grading_server


def _file_hash(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """计算文件 MD5（只读前 chunk_size 字节，快速判断）"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(chunk_size))
    return h.hexdigest()


class RLPostTrainingRunner(Developer):
    """RL Runner - 在 Docker 中执行训练代码"""

    def __init__(self, scen: Scenario, timeout: int = 3600) -> None:
        self.scen = scen
        self.timeout = timeout

    def develop(self, exp: Experiment) -> Experiment:
        """
        执行 RL 训练代码
        
        流程：
        1. 获取 Docker 环境
        2. 调用 workspace.run() 执行 main.py
        3. 验证训练是否真的发生
        4. 评测训练后模型
        """
        workspace = exp.experiment_workspace
        
        if workspace is None:
            logger.warning("No workspace found in experiment")
            return exp
            
        if "main.py" not in workspace.file_dict:
            logger.warning("No main.py found in workspace")
            return exp
        
        # 获取 Docker 环境（根据 benchmark 自动选择镜像）
        env = get_rl_env(benchmark=RL_RD_SETTING.benchmark, timeout=self.timeout)
        
        # 执行训练
        logger.info("=== Starting RL Training in Docker ===")
        result = workspace.run(env, "python main.py")
        
        # 记录结果
        logger.info(f"Training exit code: {result.exit_code}")
        logger.info(f"Training time: {result.running_time:.2f}s")
        
        if result.exit_code != 0:
            logger.warning(f"Training failed:\n{result.stdout[:1000] if result.stdout else 'No output'}")
        else:
            logger.info("Training completed successfully")
        
        # 存储结果到 experiment
        exp.result = {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "running_time": result.running_time,
        }

        # 评测
        benchmark_name = RL_RD_SETTING.benchmark or getattr(exp.sub_tasks[0], "benchmark", "") if exp.sub_tasks else ""
        exp.result["benchmark"] = None
        
        if not benchmark_name or result.exit_code != 0:
            return exp
        
        output_model = Path(workspace.workspace_path) / "output" / "model.safetensors"
        original_model = RL_MODELS_DIR / RL_RD_SETTING.base_model / "model.safetensors"
        
        if not output_model.exists():
            logger.info("No model output, skip benchmark")
            return exp
        
        if original_model.exists() and _file_hash(output_model) == _file_hash(original_model):
            logger.warning("Model unchanged from baseline, skip benchmark")
            return exp
        
        logger.info(f"=== Benchmark: {benchmark_name} ===")
        
        # 优先使用 grading server（如果有的话）
        grading_result = submit_to_grading_server(str(output_model.parent))
        if grading_result:
            exp.result["benchmark"] = grading_result
        else:
            # 本地评测
            from rdagent.scenarios.rl.autorl_bench.benchmark import run_benchmark
            exp.result["benchmark"] = run_benchmark(
                workspace_path=str(workspace.workspace_path),
                model_path=str(output_model.parent),
                model_name=RL_RD_SETTING.base_model,
                benchmark_name=benchmark_name,
            )
        return exp

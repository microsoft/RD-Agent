"""
OpenCompass Evaluator

用于所有使用 OpenCompass 评测的 benchmark（gsm8k, math 等）。
"""
import subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from rdagent.components.benchmark import BENCHMARK_CONFIGS_DIR
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator
from rdagent.utils.agent.tpl import T


class OpenCompassEvaluator(BaseEvaluator):
    """
    OpenCompass 通用评测器
    
    适用于所有使用 OpenCompass 评测的 benchmark。
    """
    
    def __init__(self, config):
        self.config = config
        self.benchmark_id = config.id
        self.eval_config = config.eval_config or {}
    
    def run_eval(
        self,
        model_path: str,
        workspace_path: str,
        model_name: str = "",
        gpu_count: int = 1,
        test_range: str = "[:]",
        **kwargs
    ) -> Dict[str, Any]:
        """使用 OpenCompass 评测"""
        result = self.get_default_result(self.benchmark_id, model_path)
        result["eval_type"] = "opencompass"
        
        if not self.validate_model(model_path):
            result["error"] = f"Model not found: {model_path}"
            return result
        
        workspace = Path(workspace_path)
        model_path = str(Path(model_path).resolve())
        work_dir = workspace / "benchmark_results"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取评测配置
        dataset_import = self.eval_config.get(
            "dataset", 
            f"opencompass.configs.datasets.{self.benchmark_id}"
        )
        # 允许 benchmark 在配置中声明默认评测切片（例如 HumanEval 仅评后半）
        effective_test_range = test_range
        if test_range == "[:]" and self.eval_config.get("test_range"):
            effective_test_range = self.eval_config["test_range"]
        
        # 从 models.yaml 获取模型推理配置
        inference_config = self._get_model_inference_config(model_name, gpu_count)
        
        # Resolve explicit dataset variable names to avoid `import *`,
        # which leaks non-serializable objects (e.g. `os`, `f` from BBH)
        # and breaks mmengine's config dump+reload in the CLI.
        dataset_imports_explicit = []
        for mod_path in [dataset_import]:
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                names = [a for a in dir(mod) if (a == "datasets" or a.endswith("_datasets")) and isinstance(getattr(mod, a), list)]
                dataset_imports_explicit.append({"module": mod_path, "names": names})
            except Exception:
                dataset_imports_explicit.append({"module": mod_path, "names": []})

        # 生成 OpenCompass 配置
        template_vars = {
            "model_abbr": f"rl-{self.benchmark_id}",
            "model_path": model_path,
            "dataset_imports": dataset_imports_explicit,
            "test_range": effective_test_range,
            "num_runs": 1,
            "pass_k": None,
            "work_dir": str(work_dir),
            "is_lora": False,
            "lora_path": "",
            **inference_config,
        }
        
        config_content = T("rdagent.components.benchmark.configs.opencompass_template:template").r(**template_vars)
        config_path = workspace / "opencompass_config.py"
        config_path.write_text(config_content)
        
        logger.info(f"Running OpenCompass benchmark: {self.benchmark_id}")
        logger.info(f"Model: {model_path}")
        
        # 运行 OpenCompass
        cmd = ["opencompass", str(config_path), "--work-dir", str(work_dir)]
        
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        except subprocess.TimeoutExpired:
            result["error"] = "OpenCompass timeout (7200s)"
            return result
        
        if proc.returncode != 0:
            error_msg = proc.stderr[:1000] if proc.stderr else proc.stdout[:1000] if proc.stdout else "No output"
            logger.warning(f"OpenCompass failed: {error_msg}")
            result["error"] = f"OpenCompass exit code: {proc.returncode}"
            result["raw_output"] = error_msg
            return result
        
        # 解析结果
        result = self._parse_results(work_dir, result)
        logger.info(f"Benchmark score: {result['score']}")
        return result
    
    def _get_model_inference_config(self, model_name: str, gpu_count: int) -> dict:
        """从 models.yaml 加载模型推理配置"""
        config_data = yaml.safe_load(open(BENCHMARK_CONFIGS_DIR / "models.yaml", "r"))
        
        default_config = config_data.get("default", {})
        models_config = config_data.get("models", {})
        
        model_specific = models_config.get(model_name, {})
        if not model_specific:
            best_match_len = 5
            for configured_model in models_config:
                if model_name.startswith(configured_model) and len(configured_model) > best_match_len:
                    model_specific = models_config[configured_model]
                    best_match_len = len(configured_model)
        
        final_config = {**default_config, **model_specific}
        
        # 处理 auto tensor_parallel_size
        if final_config.get("tensor_parallel_size") == "auto":
            if gpu_count <= 0:
                final_config["tensor_parallel_size"] = 1
            else:
                power = 0
                while (1 << (power + 1)) <= gpu_count:
                    power += 1
                final_config["tensor_parallel_size"] = 1 << power
        
        return final_config
    
    def _parse_results(self, work_dir: Path, result: dict) -> dict:
        """解析 OpenCompass 输出结果"""
        timestamped_dirs = sorted(
            [d for d in work_dir.glob("202*_*") if d.is_dir()], 
            reverse=True
        )
        
        if not timestamped_dirs:
            result["error"] = "No results directory found"
            return result
        
        summary_dir = timestamped_dirs[0] / "summary"
        csv_files = list(summary_dir.rglob("*.csv"))
        
        if not csv_files:
            result["error"] = "No results CSV found"
            return result
        
        df = pd.read_csv(csv_files[0])
        score_col = [c for c in df.columns if c not in ["dataset", "version", "metric", "mode"]]
        
        if score_col:
            numeric = []
            for raw in df[score_col[0]].dropna().values:
                try:
                    numeric.append(float(raw))
                except (ValueError, TypeError):
                    logger.warning(f"OpenCompass returned non-numeric score: {raw!r}, skipping")
            if numeric:
                result["score"] = sum(numeric) / len(numeric)
                result["accuracy_summary"] = {"accuracy": result["score"], "num_subdatasets": len(numeric)}
        
        return result

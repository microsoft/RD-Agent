"""
AutoRL-Bench Benchmark: 评测逻辑

使用 OpenCompass 评测静态任务（如 gsm8k, math）
交互式任务（如 alfworld）使用各自的 eval 模块
"""
import importlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from rdagent.components.benchmark import BENCHMARK_CONFIGS_DIR
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.tasks import get_task
from rdagent.utils.agent.tpl import T


def get_model_inference_config(model_name: str, gpu_count: int) -> dict:
    """从 models.yaml 加载模型推理配置"""
    config_data = yaml.safe_load(open(BENCHMARK_CONFIGS_DIR / "models.yaml", "r"))
    
    default_config = config_data.get("default", {})
    models_config = config_data.get("models", {})
    
    # 精确匹配
    if model_name in models_config:
        model_specific = models_config[model_name]
    else:
        # 前缀匹配
        model_specific = {}
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


def run_benchmark(
    workspace_path: str,
    model_path: str,
    model_name: str,
    benchmark_name: str,
    gpu_count: int = 1,
    test_range: Optional[str] = "[:]",
    num_runs: int = 1,
) -> Dict[str, Any]:
    """运行评测
    
    Args:
        workspace_path: 工作目录
        model_path: 训练后模型路径
        model_name: 基础模型名称
        benchmark_name: 评测任务名称
        gpu_count: GPU 数量
        test_range: 测试数据范围
        num_runs: 运行次数
    
    Returns:
        评测结果字典，包含 accuracy_summary, score 等
    """
    task_config = get_task(benchmark_name)

    """
    # only keep the following code
    eval_module = importlib.import_module(
        f"rdagent.scenarios.rl.autorl_bench.tasks.{benchmark_name}.eval"
    )
    return eval_module.run_eval(
        workspace_path=workspace_path,
        model_path=model_path,
        task_config=task_config,
    )
    # f"rdagent.scenarios.rl.autorl_bench.tasks.opencompass_adaptor.eval"
    """
    
    if task_config.eval_type == "opencompass":
        return _run_opencompass_eval(
            workspace_path=workspace_path,
            model_path=model_path,
            model_name=model_name,
            task_config=task_config,
            gpu_count=gpu_count,
            test_range=test_range,
        )
    else:
        # 动态导入各 benchmark 的 eval 模块
        eval_module = importlib.import_module(
            f"rdagent.scenarios.rl.autorl_bench.tasks.{benchmark_name}.eval"
        )
        return eval_module.run_eval(
            workspace_path=workspace_path,
            model_path=model_path,
            task_config=task_config,
        )


def _run_opencompass_eval(
    workspace_path: str,
    model_path: str,
    model_name: str,
    task_config,
    gpu_count: int = 1,
    test_range: Optional[str] = "[:]",
) -> Dict[str, Any]:
    """使用 OpenCompass 评测（本地运行）"""
    result = {
        "benchmark": task_config.id,
        "model_path": model_path,
        "eval_type": "opencompass",
        "accuracy_summary": {},
        "score": 0.0,
    }
    
    if not Path(model_path).exists():
        result["error"] = f"Model not found: {model_path}"
        return result
    
    workspace_path = Path(workspace_path)
    model_path = str(Path(model_path).resolve())
    work_dir = workspace_path / "benchmark_results"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取评测配置
    eval_config = task_config.eval_config or {}
    dataset_import = eval_config.get("dataset", f"opencompass.configs.datasets.{task_config.id}")
    
    # 从 models.yaml 获取模型推理配置
    inference_config = get_model_inference_config(model_name, gpu_count)
    
    # 生成 OpenCompass 配置（使用本地路径）
    template_vars = {
        "model_abbr": f"rl-{task_config.id}",
        "model_path": model_path,
        "dataset_imports": [dataset_import],
        "test_range": test_range,
        "num_runs": 1,
        "pass_k": None,
        "work_dir": str(work_dir),
        "is_lora": False,
        "lora_path": "",
        **inference_config,
    }
    
    config_content = T("rdagent.components.benchmark.configs.opencompass_template:template").r(**template_vars)
    config_path = workspace_path / "opencompass_config.py"
    config_path.write_text(config_content)
    
    logger.info(f"Running OpenCompass benchmark: {task_config.id}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Work dir: {work_dir}")
    
    # 本地运行 OpenCompass
    cmd = ["opencompass", str(config_path), "--work-dir", str(work_dir)]
    logger.info(f"Command: {' '.join(cmd)}")
    
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    if proc.returncode != 0:
        error_msg = proc.stderr[:1000] if proc.stderr else proc.stdout[:1000] if proc.stdout else "No output"
        logger.warning(f"OpenCompass failed: {error_msg}")
        result["error"] = f"OpenCompass exit code: {proc.returncode}"
        result["raw_output"] = error_msg
        return result
    
    # 解析结果
    timestamped_dirs = sorted([d for d in work_dir.glob("202*_*") if d.is_dir()], reverse=True)
    
    if not timestamped_dirs:
        result["error"] = "No results directory found"
        return result
    
    summary_dir = timestamped_dirs[0] / "summary"
    csv_files = list(summary_dir.rglob("*.csv"))
    
    if not csv_files:
        result["error"] = "No results CSV found"
        return result
    
    # 读取 CSV 获取分数
    df = pd.read_csv(csv_files[0])
    score_col = [c for c in df.columns if c not in ["dataset", "version", "metric", "mode"]]
    
    if score_col:
        scores = df[score_col[0]].dropna().values
        if len(scores) > 0:
            result["score"] = float(scores[0])
            result["accuracy_summary"] = {"accuracy": result["score"]}
    
    logger.info(f"Benchmark score: {result['score']}")
    return result



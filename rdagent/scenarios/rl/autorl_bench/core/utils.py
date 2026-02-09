"""
AutoRL-Bench Core Utilities

统一的工具函数：下载、baseline、grading client
"""
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from datasets import load_dataset
from huggingface_hub import snapshot_download
from loguru import logger

from rdagent.scenarios.rl.autorl_bench.conf import (
    get_models_dir,
    get_data_dir,
    get_baseline_cache_dir,
)


# ============================================================
# 下载相关
# ============================================================

def download_model(model_name: str, model_dir: Optional[str] = None) -> str:
    """下载模型（已存在则跳过）"""
    base_dir = Path(model_dir) if model_dir else get_models_dir()
    target_dir = base_dir / model_name
    
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Model exists: {target_dir}")
        return str(target_dir)
    
    logger.info(f"Downloading model: {model_name}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir=str(target_dir), local_dir_use_symlinks=False)
    logger.info(f"Model downloaded to {target_dir}")
    return str(target_dir)


def download_data(task: str, data_dir: Optional[str] = None) -> str:
    """下载数据（已存在则跳过）"""
    # 延迟导入避免循环依赖
    from rdagent.scenarios.rl.autorl_bench.benchmarks import get_benchmark
    
    config = get_benchmark(task)
    base_dir = Path(data_dir) if data_dir else get_data_dir()
    target_dir = base_dir / task
    
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Data exists: {target_dir}")
        return str(target_dir)
    
    logger.info(f"Downloading data: {task}...")
    
    if config.data_source.startswith("http"):
        # Git repo
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        _clone_repo(config.data_source, target_dir)
    else:
        # HuggingFace dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        _download_hf_dataset(config.data_source, target_dir)
    
    logger.info(f"Data downloaded to {target_dir}")
    return str(target_dir)


def _clone_repo(url: str, target_dir: Path) -> None:
    """克隆 git repo"""
    logger.info(f"Cloning {url} to {target_dir}...")
    subprocess.run(["git", "clone", "--depth", "1", url, str(target_dir)], check=True)


def _download_hf_dataset(source: str, target_dir: Path, split: str = "train") -> None:
    """下载 HuggingFace 数据集"""
    dataset = load_dataset(source, split=split)
    output_file = target_dir / f"{split}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(dataset)} samples to {output_file}")


# ============================================================
# Baseline 相关
# ============================================================

def _safe_model_name(model_name: str) -> str:
    """将模型名转为安全的文件名"""
    return re.sub(r"[/\\:*?\"<>|]", "_", model_name)


def get_baseline_score(
    task: str,
    model_name: str,
    model_path: str,
    workspace_path: str,
    gpu_count: int = 1,
    test_range: str = "[:100]",
    force_rerun: bool = False,
) -> float:
    """
    获取 baseline score（有缓存则读缓存，没有则评测）
    """
    safe_name = _safe_model_name(model_name)
    cache_file = get_baseline_cache_dir() / f"{task}_{safe_name}.json"
    
    # 检查缓存
    if not force_rerun and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            score = data.get("score", 0.0)
            logger.info(f"Baseline cache hit: {cache_file.name}, score={score}")
            return score
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
    
    # 执行评测（延迟导入避免循环依赖）
    logger.info(f"Running baseline evaluation: task={task}, model={model_name}")
    from rdagent.scenarios.rl.autorl_bench.benchmarks import get_evaluator
    
    evaluator = get_evaluator(task)
    result = evaluator.run_eval(
        model_path=model_path,
        workspace_path=workspace_path,
        model_name=model_name,
        gpu_count=gpu_count,
        test_range=test_range,
    )
    
    score = result.get("score", 0.0)
    logger.info(f"Baseline score: {score}")
    
    # 保存缓存
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "task": task,
        "model_name": model_name,
        "score": score,
        "test_range": test_range,
        "timestamp": datetime.now().isoformat(),
    }
    cache_file.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False))
    
    return score


# ============================================================
# Grading Server Client
# ============================================================

def submit_to_grading_server(
    model_path: str,
    grading_url: Optional[str] = None,
    timeout: int = 600,
) -> dict | None:
    """提交模型到 grading server 评测"""
    url = grading_url or os.environ.get("GRADING_SERVER_URL")
    if not url:
        return None
    
    try:
        logger.info(f"Submitting to grading server: {url}/submit")
        resp = requests.post(f"{url}/submit", json={"model_path": model_path}, timeout=timeout)
        if resp.status_code == 200:
            result = resp.json()
            logger.info(f"Grading result: score={result.get('score')}")
            return result
        else:
            logger.warning(f"Grading server returned {resp.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Grading server error: {e}")
        return None


def set_baseline_to_server(score: float, grading_url: Optional[str] = None) -> bool:
    """设置 baseline score 到 grading server"""
    url = grading_url or os.environ.get("GRADING_SERVER_URL")
    if not url:
        return False
    
    try:
        resp = requests.post(f"{url}/set_baseline", json={"score": score}, timeout=30)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Failed to set baseline: {e}")
        return False

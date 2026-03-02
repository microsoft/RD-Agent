"""
AutoRL-Bench Core Utilities

统一的工具函数：下载、baseline、grading client、workspace、results
"""
import csv
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import snapshot_download

from rdagent.log import rdagent_logger as logger

from rdagent.scenarios.rl.autorl_bench.conf import (
    AUTORL_BENCH_SETTING,
    get_baseline_cache_dir,
    get_data_dir,
    get_models_dir,
)
from werkzeug.serving import make_server

from rdagent.scenarios.rl.autorl_bench.core.server import app, init_server


# ============================================================
# 文件工具
# ============================================================

def ensure_symlink(src: Path, dst: Path):
    """创建软链接（已存在则跳过，并发安全）"""
    if not src.exists():
        return
    try:
        dst.symlink_to(src)
    except FileExistsError:
        pass


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
    """下载训练数据（agent 可见部分）

    支持两种模式：
    1. data_module 模式（传统）：调用 data.py 中的 download_train_data()
    2. download_data.py 脚本模式（smith benchmarks）：直接运行脚本
    """
    import importlib
    import shutil
    import sys
    from rdagent.scenarios.rl.autorl_bench.benchmarks import get_benchmark, BENCHMARKS_DIR

    config = get_benchmark(task)
    base_dir = Path(data_dir) if data_dir else get_data_dir()
    target_dir = base_dir / task

    if config.data_module:
        # 传统方式（gsm8k、alfworld 等）
        module = importlib.import_module(config.data_module)
        module.download_train_data(target_dir)
    else:
        # 脚本方式（所有 smith benchmarks）
        bench_dir = Path(config.bench_dir) if config.bench_dir else BENCHMARKS_DIR / task
        script = bench_dir / "download_data.py"
        if script.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, str(script)],
                cwd=str(bench_dir),
                check=True,
            )
            # 脚本输出到 bench_dir/data/train.jsonl，拷贝到 target_dir
            src = bench_dir / "data" / "train.jsonl"
            dst = target_dir / "train.jsonl"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        else:
            logger.warning(f"Benchmark {task} has no data_module or download_data.py, skipping.")
            target_dir.mkdir(parents=True, exist_ok=True)

    return str(target_dir)


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
    test_range: str = "[:]",
    force_rerun: bool = False,
) -> float:
    """获取 baseline score（有缓存则读缓存，没有则评测）"""
    safe_name = _safe_model_name(model_name)
    cache_file = get_baseline_cache_dir() / f"{task}_{safe_name}.json"
    
    # 检查缓存
    if not force_rerun and cache_file.exists():
        data = json.loads(cache_file.read_text())
        score = data.get("score", 0.0)
        logger.info(f"Baseline cache hit: {cache_file.name}, score={score}")
        return score
    
    # 执行评测
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
    
    logger.info(f"Submitting to grading server: {url}/submit")
    resp = requests.post(f"{url}/submit", json={"model_path": model_path}, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    logger.info(f"Grading result: score={result.get('score')}")
    return result


def set_baseline_to_server(score: float, grading_url: Optional[str] = None) -> bool:
    """设置 baseline score 到 grading server"""
    url = grading_url or os.environ.get("GRADING_SERVER_URL")
    if not url:
        return False
    
    resp = requests.post(f"{url}/set_baseline", json={"score": score}, timeout=30)
    resp.raise_for_status()
    return True


# ============================================================
# Grading Server 上下文管理器
# ============================================================

class GradingServerContext:
    """Grading Server 基类"""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def get_baseline(self, task: str, model_name: str, model_path: str, workspace_path: str) -> float:
        raise NotImplementedError
    
    def load_scores(self) -> list:
        raise NotImplementedError


class LocalServerContext(GradingServerContext):
    """本地 Flask Server"""

    def __init__(self, task: str, base_model: str, workspace: str, port: int):
        self.task = task
        self.base_model = base_model
        self.workspace = workspace
        self.port = port
        self.server = None
        self._http_server = None
        self._thread = None

    def __enter__(self):
        logger.info(f"[Local Mode] Starting evaluation server on port {self.port}...")
        self.server = init_server(self.task, self.base_model, self.workspace)

        self._http_server = make_server("0.0.0.0", self.port, app, threaded=True)
        self._thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._thread.start()

        # Poll /health for up to 15 seconds instead of blind sleep(2)
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                resp = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Grading server failed to start on port {self.port}")

        return self

    def __exit__(self, *args):
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None
    
    def get_baseline(self, task: str, model_name: str, model_path: str, workspace_path: str) -> float:
        baseline = get_baseline_score(task, model_name, model_path, workspace_path)
        self.server.set_baseline(baseline)
        return baseline
    
    def load_scores(self) -> list:
        return self.server.load_scores() if self.server else []


def create_grading_server(benchmark, workspace: Path, port: int, base_model: str) -> GradingServerContext:
    """创建 Grading Server 上下文"""
    return LocalServerContext(
        task=benchmark.id,
        base_model=base_model,
        workspace=str(workspace),
        port=port,
    )


# ============================================================
# Workspace 搭建
# ============================================================

def setup_workspace(
    run_id: str,
    agent_id: str,
    task: str,
    base_model: str,
    model_path: str,
    data_path: str,
    benchmark,
) -> Path:
    """创建隔离的 workspace 目录并挂载资源文件，返回 workspace 路径。"""
    from rdagent.scenarios.rl.autorl_bench.benchmarks import BENCHMARKS_DIR
    from rdagent.scenarios.rl.autorl_bench.conf import get_instructions_file, get_workspace_dir

    workspace = get_workspace_dir() / task / f"{run_id}_{agent_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "code").mkdir(exist_ok=True)
    (workspace / "output").mkdir(exist_ok=True)

    # 模型 & 数据 symlink
    model_link = workspace / "models" / base_model
    data_link = workspace / "data"
    model_link.parent.mkdir(parents=True, exist_ok=True)

    ensure_symlink(Path(model_path), model_link)
    ensure_symlink(Path(data_path), data_link)

    # 挂载文件：任务描述 + 通用说明 + benchmark 特有文件
    bench_dir = Path(benchmark.bench_dir) if benchmark.bench_dir else BENCHMARKS_DIR / task
    ensure_symlink(bench_dir / "description.md", workspace / "description.md")
    ensure_symlink(get_instructions_file(), workspace / "instructions.md")

    for fname in benchmark.expose_files:
        ensure_symlink(bench_dir / fname, workspace / fname)

    return workspace


# ============================================================
# Results CSV 记录
# ============================================================

RESULTS_CSV_COLUMNS = [
    "run_id", "timestamp", "task", "agent", "driver_model", "base_model",
    "baseline", "best_score", "improvement", "submissions",
    "duration_s", "success", "workspace",
]


def detect_driver_model(env: dict) -> str:
    """从环境变量检测驱动 agent 的 LLM 模型名。"""
    return (
        env.get("LLM_MODEL")
        or os.environ.get("CHAT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "unknown"
    )


def append_result(row: dict) -> Path:
    """追加一行到全局 results.csv，返回文件路径。"""
    from rdagent.scenarios.rl.autorl_bench.conf import get_autorl_bench_dir

    results_csv = get_autorl_bench_dir() / "results.csv"
    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return results_csv


# ============================================================
# 运行摘要
# ============================================================

def print_summary(
    baseline: float,
    best: dict | None,
    scores: list,
    workspace,
) -> None:
    """打印运行摘要。"""
    logger.info("=" * 60)
    logger.info(f"Baseline: {baseline}")
    if best:
        logger.info(f"Best Score: {best.get('score', 0)}")
        logger.info(f"Improvement: {best.get('improvement')}")
    logger.info(f"Total Submissions: {len(scores)}")
    logger.info(f"Workspace: {workspace}")
    logger.info("=" * 60)

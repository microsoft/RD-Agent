"""
AutoRL-Bench Core Utilities

统一的工具函数：下载、baseline、grading client
"""
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
from datasets import load_dataset
from huggingface_hub import snapshot_download
from loguru import logger

from rdagent.scenarios.rl.autorl_bench.conf import (
    AUTORL_BENCH_SETTING,
    get_baseline_cache_dir,
    get_data_dir,
    get_models_dir,
)
from rdagent.scenarios.rl.autorl_bench.core.server import app, init_server


# ============================================================
# 文件工具
# ============================================================

def ensure_symlink(src: Path, dst: Path):
    """创建软链接（已存在则跳过）"""
    if src.exists() and not (dst.is_symlink() or dst.exists()):
        dst.symlink_to(src)


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
    
    def __enter__(self):
        logger.info(f"[Local Mode] Starting evaluation server on port {self.port}...")
        self.server = init_server(self.task, self.base_model, self.workspace)
        
        server_thread = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=self.port, debug=False, threaded=False),
            daemon=True
        )
        server_thread.start()
        time.sleep(2)
        return self
    
    def get_baseline(self, task: str, model_name: str, model_path: str, workspace_path: str) -> float:
        baseline = get_baseline_score(task, model_name, model_path, workspace_path)
        self.server.set_baseline(baseline)
        return baseline
    
    def load_scores(self) -> list:
        return self.server.load_scores() if self.server else []


class DockerServerContext(GradingServerContext):
    """Docker 容器 Server"""
    
    def __init__(self, benchmark_id: str, docker_image: str, dockerfile_dir: Path, workspace: Path, port: int):
        self.benchmark_id = benchmark_id
        self.docker_image = docker_image
        self.dockerfile_dir = dockerfile_dir
        self.workspace = workspace
        self.port = port
        self.container_name = None
    
    def __enter__(self):
        logger.info(f"[Docker Mode] Starting evaluation server...")
        self.container_name = start_docker_server(
            self.benchmark_id, self.docker_image, self.dockerfile_dir, self.workspace, self.port
        )
        time.sleep(5)
        return self
    
    def __exit__(self, *args):
        if self.container_name:
            stop_docker_server(self.container_name)
    
    def get_baseline(self, task: str, model_name: str, model_path: str, workspace_path: str) -> float:
        # Docker 模式也需要评测 baseline（在主机上调用 grading server）
        baseline = get_baseline_score(task, model_name, model_path, workspace_path)
        # 通知 Docker 容器内的 server 设置 baseline
        set_baseline_to_server(f"http://localhost:{self.port}", baseline)
        return baseline
    
    def load_scores(self) -> list:
        # 从 Docker 容器内的 grading server 获取分数
        resp = requests.get(f"http://localhost:{self.port}/scores", timeout=30)
        resp.raise_for_status()
        return resp.json()


def create_grading_server(benchmark, workspace: Path, port: int, base_model: str) -> GradingServerContext:
    """工厂函数：根据 benchmark 配置创建对应的 Server 上下文"""
    from rdagent.scenarios.rl.autorl_bench.benchmarks import BENCHMARKS_DIR
    if benchmark.use_docker:
        return DockerServerContext(
            benchmark_id=benchmark.id,
            docker_image=benchmark.docker_image,
            dockerfile_dir=BENCHMARKS_DIR / benchmark.id,
            workspace=workspace,
            port=port,
        )
    else:
        return LocalServerContext(
            task=benchmark.id,
            base_model=base_model,
            workspace=str(workspace),
            port=port,
        )


# ============================================================
# Docker 相关
# ============================================================

def start_docker_server(benchmark_id: str, docker_image: str, dockerfile_dir: Path, workspace: Path, port: int) -> str:
    """启动 Docker 评测环境，返回容器名"""
    from rdagent.scenarios.rl.autorl_bench.conf import get_data_dir
    rdagent_root = AUTORL_BENCH_SETTING.rdagent_root
    data_dir = get_data_dir() / benchmark_id
    
    # 构建镜像
    dockerfile_path = dockerfile_dir / "Dockerfile"
    if dockerfile_path.exists():
        logger.info(f"[Docker] Building image: {docker_image}")
        subprocess.run(["docker", "build", "-t", docker_image, str(dockerfile_dir)], check=True)
    
    # 启动容器
    container_name = f"autorl-bench-{benchmark_id}-{port}"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    
    # 构建 docker run 命令
    docker_cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", f"{port}:5000",
        "-v", f"{workspace}:/workspace",
        "-v", f"{rdagent_root}:/app/rdagent",
        "-e", f"TASK={benchmark_id}",
        "-e", "PYTHONPATH=/app/rdagent",
    ]
    
    # 挂载数据目录（如果存在）
    if data_dir.exists():
        docker_cmd.extend(["-v", f"{data_dir}:/data/{benchmark_id}"])
    
    docker_cmd.extend([
        docker_image,
        "python", "-m", "rdagent.scenarios.rl.autorl_bench.core.server",
        "--task", benchmark_id,
        "--workspace", "/workspace",
        "--port", "5000",
    ])
    
    logger.info(f"[Docker] Starting container: {container_name}")
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Docker start failed: {result.stderr}")
    
    logger.info(f"[Docker] Container started: {result.stdout.strip()[:12]}")
    return container_name


def stop_docker_server(container_name: str):
    """停止 Docker 容器"""
    logger.info(f"[Docker] Stopping container: {container_name}")
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

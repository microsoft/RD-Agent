"""Pipeline 工具函数：数据统计、GPU 信息、模型解析。"""

import json
import os
import subprocess
from pathlib import Path


def resolve_model_path(model_name: str) -> str:
    """将 HuggingFace 模型名解析为本地路径。

    优先级：
    1. 如果 model_name 已经是本地目录 → 直接返回
    2. 查 HuggingFace 缓存 → 返回 snapshot 路径
    3. 下载模型到缓存 → 返回 snapshot 路径
    """
    # 已是本地路径
    if Path(model_name).is_dir():
        return str(Path(model_name).resolve())

    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
    except ImportError:
        print(f"  huggingface_hub not installed, using model name as-is: {model_name}")
        return model_name

    # 尝试从缓存查找
    try:
        cached = try_to_load_from_cache(model_name, "config.json")
        if isinstance(cached, str):
            # 返回 snapshot 目录（config.json 的父目录）
            snapshot_dir = str(Path(cached).parent)
            print(f"  Model found in cache: {snapshot_dir}")
            return snapshot_dir
    except Exception:
        pass

    # 缓存未命中，下载
    print(f"  Downloading model: {model_name} ...")
    try:
        path = snapshot_download(model_name)
        print(f"  Model downloaded to: {path}")
        return path
    except Exception as e:
        print(f"  WARNING: Failed to download {model_name}: {e}")
        print(f"  Falling back to model name (from_pretrained will handle it)")
        return model_name


def get_data_stats(data_path: str) -> dict:
    """读取 train.jsonl，返回样本数和长度统计。"""
    samples = []
    path = Path(data_path)
    jsonl = path / "train.jsonl" if path.is_dir() else path

    if not jsonl.exists():
        return {"count": 0, "avg_prompt_len": 0, "avg_answer_len": 0}

    try:
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    except Exception:
        return {"count": 0, "avg_prompt_len": 0, "avg_answer_len": 0}

    if not samples:
        return {"count": 0, "avg_prompt_len": 0, "avg_answer_len": 0}

    prompt_lens = []
    answer_lens = []
    for s in samples:
        prompt = s.get("prompt") or s.get("question") or s.get("instruction") or ""
        answer = s.get("answer") or s.get("response") or s.get("output") or ""
        prompt_lens.append(len(prompt))
        answer_lens.append(len(answer))

    return {
        "count": len(samples),
        "avg_prompt_len": sum(prompt_lens) // max(len(prompt_lens), 1),
        "avg_answer_len": sum(answer_lens) // max(len(answer_lens), 1),
    }


def get_gpu_info() -> dict:
    """获取 GPU 数量和型号。"""
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    gpu_name = "Unknown"
    nvidia_gpu_count = 0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            names = [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
            nvidia_gpu_count = len(names)
            if names:
                gpu_name = names[0]
    except Exception:
        pass

    # 如果设置了 CUDA_VISIBLE_DEVICES，以它为准；否则取 nvidia-smi 的数量
    if cuda_devices:
        num_gpus = len(cuda_devices.split(","))
    else:
        num_gpus = nvidia_gpu_count

    return {"num_gpus": num_gpus, "gpu_name": gpu_name, "cuda_devices": cuda_devices}

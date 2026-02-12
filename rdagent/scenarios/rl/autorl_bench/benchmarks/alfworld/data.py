"""
ALFWorld 数据下载

Agent 只能看到训练游戏数据。
评估用 eval_out_of_distribution split，由 eval.py 通过 alfworld-download 自己管理。
"""
import subprocess
from pathlib import Path

from loguru import logger


def download_train_data(target_dir: Path) -> None:
    """下载 ALFWorld 训练数据（agent 可见）

    ALFWorld 游戏数据通过 alfworld-download 下载到 ~/.cache/alfworld/。
    这里只把训练部分的游戏文件链接/复制到 target_dir，agent 看不到评估数据。
    """
    marker = target_dir / ".downloaded"
    if marker.exists():
        logger.info(f"ALFWorld train data exists: {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # 确保完整游戏数据已下载
    cache_dir = Path.home() / ".cache" / "alfworld"
    if not (cache_dir / "json_2.1.1").exists():
        logger.info("Downloading ALFWorld game data (~2.1GB, first time only)...")
        subprocess.run(["alfworld-download"], check=True)
        logger.info(f"ALFWorld data downloaded to {cache_dir}")

    # 只暴露训练数据给 agent
    train_src = cache_dir / "json_2.1.1" / "train"
    if train_src.exists():
        train_dst = target_dir / "train"
        if not train_dst.exists():
            train_dst.symlink_to(train_src)
        logger.info(f"ALFWorld train data linked: {train_dst} -> {train_src}")
    else:
        logger.warning(f"ALFWorld train data not found at {train_src}")

    # 标记已完成
    marker.touch()

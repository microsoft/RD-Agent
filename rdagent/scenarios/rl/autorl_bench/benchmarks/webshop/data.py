"""
WebShop 数据准备

注意：WebShop PyPI 包不完整（缺少 web_agent_site 模块），需要从 GitHub 克隆完整仓库。
为避免 setup.sh 破坏当前环境依赖，我们手动下载数据。
"""
import subprocess
import sys
from pathlib import Path

from loguru import logger

WEBSHOP_CACHE_DIR = Path.home() / ".cache" / "webshop"
WEBSHOP_REPO_DIR = WEBSHOP_CACHE_DIR / "repo"


def _clone_webshop_repo() -> Path:
    """克隆 WebShop 仓库到缓存目录"""
    if WEBSHOP_REPO_DIR.exists() and (WEBSHOP_REPO_DIR / ".git").exists():
        logger.info(f"WebShop repo exists: {WEBSHOP_REPO_DIR}")
        return WEBSHOP_REPO_DIR

    WEBSHOP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning WebShop repository...")

    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/princeton-nlp/webshop.git",
        str(WEBSHOP_REPO_DIR)
    ], check=True)

    logger.info(f"WebShop repo cloned to: {WEBSHOP_REPO_DIR}")
    return WEBSHOP_REPO_DIR


def _ensure_repo_in_path():
    """确保 webshop 仓库在 Python 路径中（优先于 PyPI 包）"""
    repo_str = str(WEBSHOP_REPO_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _download_webshop_data():
    """下载 WebShop 数据（手动下载，避免 setup.sh 破坏环境依赖）"""
    data_dir = WEBSHOP_REPO_DIR / "data"
    marker = data_dir / ".download_complete"

    if marker.exists():
        logger.info(f"WebShop data already downloaded: {data_dir}")
        return

    logger.info("Downloading WebShop data (~500MB, first time only)...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 使用 gdown 下载 Google Drive 文件（small 数据集，1000个产品）
    files = [
        ("1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib", "items_shuffle_1000.json"),
        ("1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu", "items_ins_v2_1000.json"),
        ("14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O", "items_human_ins.json"),
    ]

    for file_id, filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            try:
                subprocess.run(
                    ["gdown", file_id, "-O", str(filepath)],
                    check=True, timeout=120
                )
                logger.info(f"Downloaded {filename}")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Failed to download {filename}: {e}")

    # 构建搜索引擎索引
    _build_search_index()

    marker.touch()
    logger.info(f"WebShop data ready: {data_dir}")


def _build_search_index():
    """构建 WebShop 搜索引擎索引"""
    search_engine_dir = WEBSHOP_REPO_DIR / "search_engine"
    marker = search_engine_dir / ".index_built"

    if marker.exists():
        return

    logger.info("Building WebShop search index...")

    # 创建必要的目录
    resources_dir = search_engine_dir / "resources_1k"
    resources_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir = search_engine_dir / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 转换产品文件格式
        convert_script = search_engine_dir / "convert_product_file_format.py"
        if convert_script.exists():
            subprocess.run(
                [sys.executable, str(convert_script)],
                cwd=search_engine_dir, check=True, timeout=60
            )

        # 构建索引
        index_script = search_engine_dir / "run_indexing.sh"
        if index_script.exists():
            subprocess.run(
                ["bash", str(index_script)],
                cwd=search_engine_dir, check=True, timeout=120
            )

        marker.touch()
        logger.info("Search index built successfully")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to build search index: {e}")


def download_train_data(target_dir: Path) -> None:
    """准备 WebShop 训练数据（agent 可见）

    流程：
    1. 克隆 WebShop 仓库（如果不存在）
    2. 下载产品数据（手动方式，避免 setup.sh 破坏依赖）
    3. 将训练数据链接到 target_dir
    """
    marker = target_dir / ".downloaded"
    if marker.exists():
        logger.info(f"WebShop train data exists: {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    _clone_webshop_repo()
    _ensure_repo_in_path()
    _download_webshop_data()

    # 链接训练数据给 agent
    human_traj_src = WEBSHOP_REPO_DIR / "data" / "human_trajectories"
    if human_traj_src.exists():
        human_traj_dst = target_dir / "human_trajectories"
        if not human_traj_dst.exists():
            human_traj_dst.symlink_to(human_traj_src)
        logger.info(f"Linked human_trajectories: {human_traj_dst}")

    marker.touch()

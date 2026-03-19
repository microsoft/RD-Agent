"""
WebShop data preparation

Note: The WebShop PyPI package is incomplete (missing the web_agent_site module), and the complete repository needs to be cloned from GitHub.
To avoid setup.sh destroying current environment dependencies, we download the data manually.
"""

import subprocess
import sys
from pathlib import Path

from loguru import logger

WEBSHOP_CACHE_DIR = Path.home() / ".cache" / "webshop"
WEBSHOP_REPO_DIR = WEBSHOP_CACHE_DIR / "repo"


def _clone_webshop_repo() -> Path:
"""Clone the WebShop repository to the cache directory"""
    if WEBSHOP_REPO_DIR.exists() and (WEBSHOP_REPO_DIR / ".git").exists():
        logger.info(f"WebShop repo exists: {WEBSHOP_REPO_DIR}")
        return WEBSHOP_REPO_DIR

    WEBSHOP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning WebShop repository...")

    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/princeton-nlp/webshop.git", str(WEBSHOP_REPO_DIR)],
        check=True,
    )

    logger.info(f"WebShop repo cloned to: {WEBSHOP_REPO_DIR}")
    return WEBSHOP_REPO_DIR


def _ensure_repo_in_path():
"""Make sure the webshop repository is in the Python path (preferring the PyPI package).

Also write webshop.pth to venv site-packages to enable any child processes (accelerate launch, etc.)
You can import web_agent_site directly without manually setting sys.path.
    """
    import site

    repo_str = str(WEBSHOP_REPO_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # Write a .pth file so subprocesses inherit the path without extra setup.
    pth_content = repo_str + "\n"
    for sp in site.getsitepackages():
        pth_file = Path(sp) / "webshop.pth"
        try:
            if not pth_file.exists() or pth_file.read_text() != pth_content:
                pth_file.write_text(pth_content)
                logger.info(f"Registered webshop path via {pth_file}")
            break
        except OSError:
            continue


def _download_webshop_data():
"""Download WebShop data (manual download to avoid setup.sh damaging environment dependencies)"""
    data_dir = WEBSHOP_REPO_DIR / "data"
    marker = data_dir / ".download_complete"

    if marker.exists():
        logger.info(f"WebShop data already downloaded: {data_dir}")
        return

    logger.info("Downloading WebShop data (~500MB, first time only)...")
    data_dir.mkdir(parents=True, exist_ok=True)

# Use gdown to download Google Drive files (small data set, 1000 products)
    files = [
        ("1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib", "items_shuffle_1000.json"),
        ("1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu", "items_ins_v2_1000.json"),
        ("14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O", "items_human_ins.json"),
    ]

    for file_id, filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            try:
                subprocess.run(["gdown", file_id, "-O", str(filepath)], check=True, timeout=120)
                logger.info(f"Downloaded {filename}")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Failed to download {filename}: {e}")

# Build search engine index
    _build_search_index()

    marker.touch()
    logger.info(f"WebShop data ready: {data_dir}")


def _build_search_index():
"""Build WebShop search engine index"""
    search_engine_dir = WEBSHOP_REPO_DIR / "search_engine"
    marker = search_engine_dir / ".index_built"

    if marker.exists():
        return

    logger.info("Building WebShop search index...")

#Create all directories needed by convert_product_file_format.py
    for d in ["resources_100", "resources", "resources_1k", "resources_100k", "indexes"]:
        (search_engine_dir / d).mkdir(parents=True, exist_ok=True)

    try:
# Convert product file format
        convert_script = search_engine_dir / "convert_product_file_format.py"
        if convert_script.exists():
            subprocess.run([sys.executable, str(convert_script)], cwd=search_engine_dir, check=True, timeout=60)

# Build index
        index_script = search_engine_dir / "run_indexing.sh"
        if index_script.exists():
            subprocess.run(["bash", str(index_script)], cwd=search_engine_dir, check=True, timeout=120)

        marker.touch()
        logger.info("Search index built successfully")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Failed to build search index: {e}") from e


def download_train_data(target_dir: Path) -> None:
"""Prepare WebShop training data (visible to agent)

process:
1. Clone the WebShop repository (if it does not exist)
2. Download product data (manual method to avoid setup.sh destroying dependencies)
3. Link the training data to target_dir
    """
    marker = target_dir / ".downloaded"
    if marker.exists():
        logger.info(f"WebShop train data exists: {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    _clone_webshop_repo()
    _ensure_repo_in_path()
    _download_webshop_data()

# Link training data to agent
    human_traj_src = WEBSHOP_REPO_DIR / "data" / "human_trajectories"
    if human_traj_src.exists():
        human_traj_dst = target_dir / "human_trajectories"
        if not human_traj_dst.exists():
            human_traj_dst.symlink_to(human_traj_src)
        logger.info(f"Linked human_trajectories: {human_traj_dst}")

    marker.touch()

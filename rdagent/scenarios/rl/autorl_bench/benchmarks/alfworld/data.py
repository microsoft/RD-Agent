"""
ALFWorld data preparation

Official alfworld-download downloads all data at once (json + pddl + game.tw-pddl + logic)
Go to ~/.cache/alfworld/ and symlink only the training data to the agent.
"""

import sys
from pathlib import Path

from loguru import logger


def _run_alfworld_download() -> None:
"""Call alfworld-download, compatible with conda env PATH issue"""
    import subprocess

    bin_dir = Path(sys.executable).parent
    script = bin_dir / "alfworld-download"
    if script.exists():
        subprocess.run([sys.executable, str(script)], check=True)
    else:
        subprocess.run(["alfworld-download"], check=True)


def _ensure_alfworld_data() -> Path:
"""Make sure alfworld complete data has been downloaded and return to the data root directory

alfworld-download downloads three zips to ~/.cache/alfworld/:
      - json_2.1.1_json.zip  -> traj_data.json
      - json_2.1.1_pddl.zip  -> initial_state.pddl
      - json_2.1.3_tw-pddl.zip -> game.tw-pddl
      + logic/alfred.pddl, logic/alfred.twl2
    """
    cache_dir = Path.home() / ".cache" / "alfworld"
    json_dir = cache_dir / "json_2.1.1"

    tw_pddl_ok = json_dir.exists() and any(json_dir.rglob("game.tw-pddl"))
    pddl_ok = json_dir.exists() and any(json_dir.rglob("initial_state.pddl"))
    logic_ok = (cache_dir / "logic" / "alfred.pddl").exists()

    if tw_pddl_ok and pddl_ok and logic_ok:
        logger.info(f"ALFWorld data already complete: {cache_dir}")
        return cache_dir

    logger.info("Running alfworld-download (downloads ~2GB, first time only)...")
    _run_alfworld_download()

    if not any(json_dir.rglob("game.tw-pddl")):
        raise RuntimeError(
            f"alfworld-download finished but game.tw-pddl not found in {json_dir}. "
            "Check network connectivity to GitHub releases."
        )
    logger.info(f"ALFWorld data ready: {cache_dir}")
    return cache_dir


def download_train_data(target_dir: Path) -> None:
"""Prepare ALFWorld training data (visible to agent)"""
    marker = target_dir / ".downloaded"
    if marker.exists():
        logger.info(f"ALFWorld train data exists: {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = _ensure_alfworld_data()
    train_src = cache_dir / "json_2.1.1" / "train"

    if not train_src.exists():
        raise FileNotFoundError(f"ALFWorld train data not found: {train_src}")

    train_dst = target_dir / "train"
    if not train_dst.exists():
        train_dst.symlink_to(train_src)
    logger.info(f"ALFWorld train data linked: {train_dst} -> {train_src}")

    marker.touch()

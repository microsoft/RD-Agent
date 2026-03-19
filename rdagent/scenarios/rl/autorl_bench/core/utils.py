"""
AutoRL-Bench Core Utilities

Unified tool functions: download, baseline, grading client, workspace, results
"""

import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import snapshot_download

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.conf import (
    get_baseline_cache_dir,
    get_data_dir,
    get_models_dir,
)


def kill_process_group(proc: "subprocess.Popen") -> None:
"""Try to kill the process group: SIGTERM → SIGKILL → proc.kill()"""
    import signal as _signal

    if proc.poll() is not None:
        return
    for sig in (_signal.SIGTERM, _signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=10)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue
        except OSError:
            break
    proc.kill()
    proc.wait()


# ============================================================
#File tool
# ============================================================


def ensure_symlink(src: Path, dst: Path):
"""Create a soft link (skip if it already exists, safe for concurrency)"""
    if not src.exists():
        return
    try:
        dst.symlink_to(src)
    except FileExistsError:
        pass


# ============================================================
# Download related
# ============================================================


def download_model(model_name: str, model_dir: Optional[str] = None) -> str:
"""Download the model (skip if it already exists)"""
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
"""Download training data (visible part to agent)

Two modes are supported:
1. data_module mode (traditional): call download_train_data() in data.py
2. download_data.py script mode (smith benchmarks): run the script directly
    """
    import importlib
    import shutil
    import sys

    from rdagent.scenarios.rl.autorl_bench.benchmarks import (
        BENCHMARKS_DIR,
        get_benchmark,
    )

    config = get_benchmark(task)
    base_dir = Path(data_dir) if data_dir else get_data_dir()
    target_dir = base_dir / task

    if config.data_module:
# Traditional method (gsm8k, alfworld, etc.)
        module = importlib.import_module(config.data_module)
        module.download_train_data(target_dir)
    else:
# Script mode (all smith benchmarks)
        bench_dir = Path(config.bench_dir) if config.bench_dir else BENCHMARKS_DIR / task
        script = bench_dir / "download_data.py"
        if script.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, str(script)],
                cwd=str(bench_dir),
                check=True,
            )
# Script output to bench_dir/data/train.jsonl, copy to target_dir
            src = bench_dir / "data" / "train.jsonl"
            dst = target_dir / "train.jsonl"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        else:
            # No download script — copy pre-existing data from bench_dir/data/
            target_dir.mkdir(parents=True, exist_ok=True)
            src = bench_dir / "data" / "train.jsonl"
            dst = target_dir / "train.jsonl"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                logger.info(f"Copied {src} → {dst}")
            elif not src.exists():
                logger.warning(f"Benchmark {task} has no data_module, download_data.py, or train.jsonl")

    return str(target_dir)


# ============================================================
#Baseline related
# ============================================================


def _safe_model_name(model_name: str) -> str:
"""Convert the model name to a safe file name"""
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
"""Get the baseline score (read the cache if there is a cache, and evaluate if there is no cache)"""
    safe_name = _safe_model_name(model_name)
    cache_file = get_baseline_cache_dir() / f"{task}_{safe_name}.json"

# Check cache
    if not force_rerun and cache_file.exists():
        data = json.loads(cache_file.read_text())
        score = data.get("score", 0.0)
        logger.info(f"Baseline cache hit: {cache_file.name}, score={score}")
        return score

# Execute evaluation
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
    error = result.get("error")
    logger.info(f"Baseline score: {score}")

    # Only cache successful evaluations — failed ones should be retried next time
    if not error:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "task": task,
            "model_name": model_name,
            "score": score,
            "test_range": test_range,
            "timestamp": datetime.now().isoformat(),
        }
        cache_file.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False))
    else:
        logger.warning(f"Baseline evaluation failed ({error}), result NOT cached")

    return score


# ============================================================
# Grading Server Client
# ============================================================


def submit_to_grading_server(
    model_path: str,
    grading_url: Optional[str] = None,
    timeout: int = 600,
) -> dict | None:
"""Submit the model to the grading server for evaluation"""
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
"""Set baseline score to grading server"""
    url = grading_url or os.environ.get("GRADING_SERVER_URL")
    if not url:
        return False

    resp = requests.post(f"{url}/set_baseline", json={"score": score}, timeout=30)
    resp.raise_for_status()
    return True


# ============================================================
# Workspace setup
# ============================================================


def init_run_meta(workspace: Path, timeout_s: int) -> Path:
"""Initialize run_meta.json (single source of truth)."""
    run_meta = workspace / "run_meta.json"
    payload = {
        "start_time": int(datetime.now().timestamp()),
        "timeout_s": int(timeout_s),
        "last_submit_time": None,
        "end_time": None,
    }
    run_meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return run_meta


def update_run_meta(workspace: Path, **fields) -> Path:
"""Update some fields of run_meta.json."""
    run_meta = workspace / "run_meta.json"
    data = json.loads(run_meta.read_text()) if run_meta.exists() else {}
    data.update(fields)
    run_meta.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return run_meta


def read_run_meta(workspace: Path) -> dict:
"""Read run_meta.json."""
    run_meta = workspace / "run_meta.json"
    return json.loads(run_meta.read_text()) if run_meta.exists() else {}


def setup_workspace(
    run_id: str,
    agent_id: str,
    task: str,
    base_model: str,
    model_path: str,
    data_path: str,
    benchmark,
) -> Path:
"""Create an isolated workspace directory and mount resource files, returning the workspace path."""
    from rdagent.scenarios.rl.autorl_bench.benchmarks import BENCHMARKS_DIR
    from rdagent.scenarios.rl.autorl_bench.conf import (
        get_instructions_file,
        get_workspace_dir,
    )

    workspace = get_workspace_dir() / task / f"{run_id}_{agent_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "code").mkdir(exist_ok=True)
    (workspace / "output").mkdir(exist_ok=True)
    (workspace / "reports").mkdir(exist_ok=True)

# model & data symlink
    model_link = workspace / "models" / base_model
    data_link = workspace / "data"
    model_link.parent.mkdir(parents=True, exist_ok=True)

    ensure_symlink(Path(model_path), model_link)
    ensure_symlink(Path(data_path), data_link)

#Mount files: task description + general instructions + benchmark-specific files
    bench_dir = Path(benchmark.bench_dir) if benchmark.bench_dir else BENCHMARKS_DIR / task
    ensure_symlink(bench_dir / "description.md", workspace / "description.md")
    ensure_symlink(get_instructions_file(), workspace / "instructions.md")

    for fname in benchmark.expose_files:
        ensure_symlink(bench_dir / fname, workspace / fname)

    return workspace


# ============================================================
# Results CSV record
# ============================================================

RESULTS_CSV_COLUMNS = [
    "run_id",
    "timestamp",
    "task",
    "agent",
    "driver_model",
    "base_model",
    "baseline",
    "best_score",
    "improvement",
    "submissions",
    "duration_s",
    "success",
    "workspace",
]


def detect_driver_model(env: dict) -> str:
"""Detect the LLM model name of the driver agent from environment variables."""
    return env.get("LLM_MODEL") or os.environ.get("CHAT_MODEL") or os.environ.get("OPENAI_MODEL") or "unknown"


def append_result(row: dict) -> Path:
"""Append a line to global results.csv and return the file path."""
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
# Run summary
# ============================================================


def print_summary(
    baseline: float,
    best: dict | None,
    scores: list,
    workspace,
) -> None:
"""Print the run summary."""
    logger.info("=" * 60)
    logger.info(f"Baseline: {baseline}")
    if best:
        logger.info(f"Best Score: {best.get('score', 0)}")
        logger.info(f"Improvement: {best.get('improvement')}")
    logger.info(f"Total Submissions: {len(scores)}")
    logger.info(f"Workspace: {workspace}")
    logger.info("=" * 60)

#!/usr/bin/env python
"""
AutoRL-Bench Runner

入口脚本，使用 core/benchmarks/agents 结构。

Usage:
    python run.py --agent openhands --task gsm8k --model Qwen/Qwen2.5-0.5B
"""
import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from rdagent.scenarios.rl.autorl_bench.agents import get_agent
from rdagent.scenarios.rl.autorl_bench.benchmarks import get_benchmark, BENCHMARKS_DIR
from rdagent.scenarios.rl.autorl_bench.conf import (
    get_autorl_bench_dir,
    get_workspace_dir,
    get_instructions_file,
)
from rdagent.scenarios.rl.autorl_bench.core import (
    ensure_symlink,
    download_model,
    download_data,
    create_grading_server,
)

# 全局实验记录表
RESULTS_CSV_COLUMNS = [
    "run_id", "timestamp", "task", "agent", "base_model",
    "baseline", "best_score", "improvement", "submissions",
    "duration_s", "success", "workspace",
]


def run(
    agent_id: str,
    task: str,
    base_model: str,
    timeout: int = 3600,
    port: int = 5000,
) -> dict:
    """运行 Agent 评测"""
    start_time = datetime.now()
    run_id = start_time.strftime("%Y%m%dT%H%M%S")
    benchmark = get_benchmark(task)

    # 1. 准备资源（已有则跳过下载）
    print("Preparing resources...")
    model_path = download_model(base_model)
    data_path = download_data(task)

    # 2. 设置 workspace（每次运行隔离）
    workspace = get_workspace_dir() / task / f"{run_id}_{agent_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "code").mkdir(exist_ok=True)
    (workspace / "output").mkdir(exist_ok=True)

    model_link = workspace / "models" / base_model
    data_link = workspace / "data"
    model_link.parent.mkdir(parents=True, exist_ok=True)

    ensure_symlink(Path(model_path), model_link)
    ensure_symlink(Path(data_path), data_link)

    # 3. 挂载文件到 workspace（让 agent 能看到任务描述和评测逻辑）
    bench_dir = BENCHMARKS_DIR / task

    # 3a. 通用文件：所有 benchmark 都需要
    ensure_symlink(bench_dir / "description.md", workspace / "description.md")
    ensure_symlink(get_instructions_file(), workspace / "instructions.md")

    # 3b. benchmark 特有的额外文件（如 eval.py、react_prompts.json）
    for fname in benchmark.expose_files:
        ensure_symlink(bench_dir / fname, workspace / fname)

    # 4. 启动 Grading Server
    with create_grading_server(benchmark, workspace, port, base_model) as grading:
        # 5. 评测 baseline
        print("Evaluating baseline...")
        baseline = grading.get_baseline(task, base_model, str(model_link), str(workspace))
        print(f"  Baseline Score: {baseline}")

        # 6. 运行 Agent
        agent = get_agent(agent_id)
        print(f"Running agent: {agent.name}")

        env = {
            **os.environ,
            "TASK": task,
            "BASE_MODEL": base_model,
            "WORKSPACE": str(workspace),
            "MODEL_PATH": str(model_link),
            "DATA_PATH": str(data_link),
            "OUTPUT_DIR": str(workspace / "output"),
            "GRADING_SERVER_URL": f"http://localhost:{port}",
            **agent.env_vars,
        }

        try:
            proc = subprocess.run(
                ["bash", str(agent.start)],
                env=env,
                timeout=timeout,
            )
            success = proc.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"Agent timed out after {timeout}s, collecting results...")
            success = False

        # 7. 收集结果
        scores = grading.load_scores()

    # 8. 保存结果
    end_time = datetime.now()
    best = max(scores, key=lambda x: x.get("score", 0)) if scores else None

    result = {
        "success": success,
        "agent_id": agent_id,
        "task": task,
        "base_model": base_model,
        "baseline_score": baseline,
        "best": best,
        "total_submissions": len(scores),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "docker_mode": benchmark.use_docker,
    }

    # 追加到全局 results.csv
    best_score = best.get("score", 0) if best else 0
    improvement = best.get("improvement") if best else None
    results_csv = get_autorl_bench_dir() / "results.csv"
    write_header = not results_csv.exists()

    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "agent": agent_id,
            "base_model": base_model,
            "baseline": baseline,
            "best_score": best_score,
            "improvement": improvement,
            "submissions": len(scores),
            "duration_s": round((end_time - start_time).total_seconds()),
            "success": success,
            "workspace": str(workspace),
        })

    print("\n" + "=" * 60)
    print(f"Mode: {'Docker' if benchmark.use_docker else 'Local'}")
    print(f"Baseline: {baseline}")
    if best:
        print(f"Best Score: {best_score}")
        print(f"Improvement: {improvement}")
    print(f"Total Submissions: {len(scores)}")
    print(f"Workspace: {workspace}")
    print(f"Results: {results_csv}")
    print("=" * 60)

    return result


def main():
    load_dotenv(".env")

    parser = argparse.ArgumentParser(description="AutoRL-Bench Runner")
    parser.add_argument("--agent", "-a", required=True, help="Agent ID (openhands, rdagent)")
    parser.add_argument("--task", "-t", required=True, help="Task name (gsm8k, math, alfworld)")
    parser.add_argument("--model", "-m", required=True, help="Base model name")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--port", type=int, default=5000, help="Grading server port")
    args = parser.parse_args()

    result = run(
        agent_id=args.agent,
        task=args.task,
        base_model=args.model,
        timeout=args.timeout,
        port=args.port,
    )

    exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
AutoRL-Bench Runner

入口脚本，使用 core/benchmarks/agents 结构。

Usage:
    python run.py --agent openhands --task gsm8k --model Qwen/Qwen2.5-0.5B
"""
import argparse
import os
import subprocess
from datetime import datetime

from dotenv import load_dotenv

from rdagent.scenarios.rl.autorl_bench.agents import get_agent
from rdagent.scenarios.rl.autorl_bench.benchmarks import get_benchmark
from rdagent.scenarios.rl.autorl_bench.core import (
    download_model,
    download_data,
    create_grading_server,
    setup_workspace,
    append_result,
    detect_driver_model,
    print_summary,
)


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

    # 2. 搭建 workspace（目录创建 + symlink 挂载）
    workspace = setup_workspace(
        run_id, agent_id, task, base_model, model_path, data_path, benchmark,
    )

    # 3. 启动 Grading Server + 运行 Agent
    with create_grading_server(benchmark, workspace, port, base_model) as grading:
        print("Evaluating baseline...")
        baseline = grading.get_baseline(
            task, base_model, str(workspace / "models" / base_model), str(workspace),
        )
        print(f"  Baseline Score: {baseline}")

        agent = get_agent(agent_id)
        print(f"Running agent: {agent.name}")

        env = {
            **os.environ,
            "TASK": task,
            "BASE_MODEL": base_model,
            "WORKSPACE": str(workspace),
            "MODEL_PATH": str(workspace / "models" / base_model),
            "DATA_PATH": str(workspace / "data"),
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

        scores = grading.load_scores()

    # 4. 保存结果
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
    append_result({
        "run_id": run_id,
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "agent": agent_id,
        "driver_model": detect_driver_model(env),
        "base_model": base_model,
        "baseline": baseline,
        "best_score": best.get("score", 0) if best else 0,
        "improvement": best.get("improvement") if best else None,
        "submissions": len(scores),
        "duration_s": round((end_time - start_time).total_seconds()),
        "success": success,
        "workspace": str(workspace),
    })

    print_summary(baseline, best, scores, workspace, benchmark.use_docker)

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

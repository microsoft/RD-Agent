#!/usr/bin/env python
"""
AutoRL-Bench Runner

入口脚本，使用 core/benchmarks/agents 结构。

Usage:
    python run.py --agent openhands --task gsm8k --model Qwen/Qwen2.5-0.5B
"""
import argparse
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from rdagent.scenarios.rl.autorl_bench.agents import get_agent
from rdagent.scenarios.rl.autorl_bench.conf import get_workspace_dir, get_results_dir
from rdagent.scenarios.rl.autorl_bench.core import (
    download_model,
    download_data,
    get_baseline_score,
)
from rdagent.scenarios.rl.autorl_bench.core.server import init_server, app


def run(
    agent_id: str,
    task: str,
    base_model: str,
    timeout: int = 3600,
    port: int = 5000,
) -> dict:
    """运行 Agent 评测"""
    start_time = datetime.now()
    
    # 1. 准备资源
    print(f"Preparing resources...")
    model_path = download_model(base_model)
    data_path = download_data(task)
    
    # 2. 设置 workspace
    workspace = get_workspace_dir() / task
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "output").mkdir(exist_ok=True)
    
    # 创建软链接（已存在则跳过）
    model_link = workspace / "models" / base_model
    data_link = workspace / "data"
    model_link.parent.mkdir(parents=True, exist_ok=True)
    
    if not (model_link.is_symlink() or model_link.exists()):
        model_link.symlink_to(model_path)
    if not (data_link.is_symlink() or data_link.exists()):
        data_link.symlink_to(data_path)
    
    # 3. 启动 grading server
    print(f"Starting grading server on port {port}...")
    server = init_server(task, base_model, str(workspace))
    
    # 在后台线程启动 Flask
    server_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, threaded=False),
        daemon=True
    )
    server_thread.start()
    time.sleep(2)  # 等待服务器启动
    
    # 4. 评测 baseline
    print(f"Evaluating baseline...")
    baseline = get_baseline_score(
        task=task,
        model_name=base_model,
        model_path=str(model_link),
        workspace_path=str(workspace),
    )
    server.set_baseline(baseline)
    print(f"  Baseline Score: {baseline}")
    
    # 5. 运行 Agent
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
        success = False
    
    # 6. 收集结果
    end_time = datetime.now()
    scores = server.load_scores()
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
    }
    
    # 保存结果
    result_dir = get_results_dir() / f"{start_time.strftime('%Y-%m-%dT%H-%M-%S')}_{task}_{agent_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    
    print("\n" + "=" * 60)
    if best:
        print(f"Best Score: {best.get('score')}")
        print(f"Improvement: {best.get('improvement')}")
    print(f"Total Submissions: {len(scores)}")
    print(f"Result saved: {result_dir}")
    print("=" * 60)
    
    return result


def main():
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

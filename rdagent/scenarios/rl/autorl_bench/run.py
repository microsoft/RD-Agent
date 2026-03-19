#!/usr/bin/env python
"""
AutoRL-Bench Runner

Entry script.

Usage:
    python -m rdagent.scenarios.rl.autorl_bench.run \
        --agent example_agent --task gsm8k --model Qwen/Qwen2.5-0.5B
"""

import argparse
import os
import signal
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger as loguru_logger

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.agents import get_agent
from rdagent.scenarios.rl.autorl_bench.benchmarks import get_benchmark
from rdagent.scenarios.rl.autorl_bench.core import (
    append_result,
    create_grading_server,
    detect_driver_model,
    download_data,
    download_model,
    init_run_meta,
    kill_process_group,
    print_summary,
    run_workspace_metrics,
    setup_workspace,
    update_run_meta,
)


def run(
    agent_id: str,
    task: str,
    base_model: str,
    timeout: int = 3600,
    port: int = 5000,
) -> dict:
    """Run Agent Evaluation"""
    from rdagent.scenarios.rl.autorl_bench.conf import get_workspace_dir

    start_time = datetime.now()
    run_id = start_time.strftime("%Y%m%dT%H%M%S")
    if port != 5000:
        run_id = f"{run_id}_p{port}"
    benchmark = get_benchmark(task)

    # Independent workspace + independent log file for each run
    workspace = get_workspace_dir() / task / f"{run_id}_{agent_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    log_file = workspace / "run.log"
    _sink_id = loguru_logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG")

    # Use a mutable container to allow closures to access subsequently assigned agent subprocesses
    _agent_proc = [None]

    # When receiving SIGTERM/SIGINT, kill the entire process tree and exit.
    def _on_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, terminating...")
        proc = _agent_proc[0]
        if proc is not None:
            kill_process_group(proc)
        logger.info(f"Run interrupted by {sig_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        loguru_logger.remove(_sink_id)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    logger.info(f"=== AutoRL-Bench ===")
    logger.info(f"Agent: {agent_id}, Task: {task}, Model: {base_model}")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Prepare resources (skip downloading if you already have them)
    logger.info("Preparing resources...")
    model_path = download_model(base_model)
    data_path = download_data(task)

    # 2. Build workspace (supplement symlink mounting)
    workspace = setup_workspace(
        run_id,
        agent_id,
        task,
        base_model,
        model_path,
        data_path,
        benchmark,
    )
    init_run_meta(workspace, timeout)

    # 3. Start Grading Server + run Agent
    with create_grading_server(benchmark, workspace, port, base_model) as grading:
        logger.info("Evaluating baseline...")
        baseline = grading.get_baseline(
            task,
            base_model,
            str(workspace / "models" / base_model),
            str(workspace),
        )
        logger.info(f"Baseline Score: {baseline}")

        agent = get_agent(agent_id)
        logger.info(f"Running agent: {agent.name}")

        env = {
            **agent.env_vars,
            **os.environ,
            "TASK": task,
            "BASE_MODEL": base_model,
            "WORKSPACE": str(workspace),
            "MODEL_PATH": str(workspace / "models" / base_model),
            "DATA_PATH": str(workspace / "data"),
            "OUTPUT_DIR": str(workspace / "output"),
            "GRADING_SERVER_URL": f"http://localhost:{port}",
        }

        agent_log = workspace / "agent.log"
        success = False
        with open(agent_log, "w", encoding="utf-8") as af:
            proc = subprocess.Popen(
                ["bash", str(agent.start)],
                env=env,
                stdout=af,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _agent_proc[0] = proc
            try:
                proc.wait(timeout=timeout)
                success = proc.returncode == 0
                logger.info(f"Agent finished, exit_code={proc.returncode}, log: {agent_log}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Agent timed out after {timeout}s, killing process group...")
                kill_process_group(proc)

        scores = grading.load_scores()

    # 4. Save results
    end_time = datetime.now()
    update_run_meta(workspace, end_time=int(end_time.timestamp()))
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

    # Append to global results.csv
    append_result(
        {
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
        }
    )

    try:
        run_workspace_metrics(
            workspace=workspace,
            baseline=baseline,
            base_model_path=str(workspace / "models" / base_model),
        )
    except Exception:
        logger.exception("Failed to write workspace metrics")

    print_summary(baseline, best, scores, workspace)

    logger.info(f"Log saved to: {log_file}")

    # Remove the file sink added by this run (to avoid exceptions causing the process to exit)
    if _sink_id is not None:
        try:
            loguru_logger.remove(_sink_id)
        except Exception:
            logger.exception(f"Failed to remove log sink id={_sink_id}")

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

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()

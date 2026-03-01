#!/usr/bin/env python3
"""
OpenCode RL Post-training Pipeline (Fixed-Stage)

每轮迭代：代码生成 → 训练执行 → Grading Server 评测 → 反馈注入。
"""

import argparse
import atexit
import os
import signal
import shutil
import sys
import time
from pathlib import Path

from benchmarks.registry import get_benchmark, list_benchmarks
from pipeline.runner import run_pipeline
from pipeline.utils import resolve_model_path


def _cleanup_active_clients():
    """Close all active OpenCodeClient instances to prevent orphan processes."""
    try:
        from runner_fsm.opencode.client import _active_clients
    except ImportError:
        return
    for client in list(_active_clients):
        try:
            client.close()
        except Exception:
            pass


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM: clean up child processes and exit."""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n  Received {sig_name}, cleaning up...", flush=True)
    _cleanup_active_clients()
    sys.exit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_cleanup_active_clients)

    parser = argparse.ArgumentParser(description="OpenCode RL Pipeline (Fixed-Stage)")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        help="Benchmark 名称（对应 benchmarks/ 下的子目录）")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--run-dir", type=str, default="",
                        help="指定运行目录（默认自动生成 runs/{benchmark}_{timestamp}）")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--training-timeout", type=int, default=3600)
    parser.add_argument("--max-agent-steps", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=3,
                        help="各阶段（code_gen/fix/analysis）失败后自动重试次数（默认 3）")
    parser.add_argument("--resume", action="store_true",
                        help="从上次 checkpoint 断点续跑（需指定 --run-dir）")
    parser.add_argument("--stale-timeout", type=int, default=180,
                        help="LLM 无响应超时秒数，超过后自动重试（默认 180）")
    parser.add_argument("--http-timeout", type=int, default=300,
                        help="OpenCode HTTP 请求超时秒数（默认 300）")
    parser.add_argument("--eval-timeout", type=int, default=600,
                        help="Grading Server 评测请求超时秒数（默认 600）")

    parser.add_argument("--list-benchmarks", action="store_true",
                        help="列出所有可用 benchmark 并退出")

    args = parser.parse_args()

    if args.list_benchmarks:
        names = list_benchmarks()
        if not names:
            print("No benchmarks found in benchmarks/ directory.")
        else:
            print(f"Available benchmarks ({len(names)}):")
            for n in names:
                b = get_benchmark(n)
                print(f"  {n:<20s} [{b.task_type}]  {b.description}")
        sys.exit(0)

    bench = get_benchmark(args.benchmark)

    # 数据不存在时自动下载（如果框架已通过 DATA_PATH 提供数据则跳过）
    if not os.environ.get("DATA_PATH") and not bench.train_jsonl.exists():
        print(f"  数据文件不存在，自动下载 {args.benchmark} ...")
        from benchmarks.download import download_benchmark
        if not download_benchmark(bench.root):
            print(f"  ERROR: 数据下载失败，请手动运行: python benchmarks/download.py {args.benchmark}")
            sys.exit(1)

    data_dir = str(bench.data_dir.resolve())

    model_path = os.environ.get("MODEL_PATH") or resolve_model_path(args.base_model)
    os.environ["MODEL_PATH"] = model_path
    print(f"  Model: {args.base_model} → {model_path}")

    if args.run_dir:
        run_dir = str(Path(args.run_dir).resolve())
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = str((Path("runs") / f"{args.benchmark}_{ts}").resolve())

    output_dir = os.environ.get("OUTPUT_DIR") or str(Path(run_dir) / "output")
    data_path = os.environ.get("DATA_PATH") or data_dir
    code_dir = str(Path(run_dir) / "code")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(code_dir).mkdir(parents=True, exist_ok=True)

    os.environ["DATA_PATH"] = data_path
    os.environ["OUTPUT_DIR"] = output_dir
    os.environ["TRAINING_TIMEOUT"] = str(args.training_timeout)

    data_link = Path(run_dir) / "data"
    if not data_link.exists():
        data_link.symlink_to(Path(data_path).resolve())

    for desc_name in ["description.md", "instructions.md"]:
        src_desc = bench.root / desc_name
        dst_desc = Path(run_dir) / desc_name
        if src_desc.exists() and not dst_desc.exists():
            shutil.copy2(src_desc, dst_desc)

    # Benchmark 特有文件（如 eval.py, react_prompts.json）拷贝到 workspace
    for fname in bench.expose_files:
        src = bench.root / fname
        dst = Path(run_dir) / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  Exposed: {fname}")

    fsm_config = {
        "target_repo": run_dir,
        "opencode_url": os.environ.get("OPENCODE_URL", ""),
        "opencode_model": os.environ.get("OPENCODE_MODEL", ""),
    }

    run_pipeline(
        task=args.benchmark,
        base_model=args.base_model,
        workspace=run_dir,
        data_path=data_path,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        training_timeout=args.training_timeout,
        max_agent_steps=args.max_agent_steps,
        max_retries=args.max_retries,
        fsm_config=fsm_config,
        resume=args.resume,
        stale_timeout=args.stale_timeout,
        http_timeout=args.http_timeout,
        eval_timeout=args.eval_timeout,
        task_type=bench.task_type,
        expose_files=bench.expose_files,
    )


if __name__ == "__main__":
    main()

"""
测试 benchmark 评测功能

用法:
    python -m rdagent.scenarios.rl.autorl_bench.test.test_benchmark \
        --model-path /path/to/model \
        --task gsm8k
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="本地模型路径")
    parser.add_argument("--model-name", default=None, help="模型名称（默认从路径推断）")
    parser.add_argument("--task", default="gsm8k", help="评测任务")
    parser.add_argument("--port", type=int, default=15000, help="grading server 端口")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    model_name = args.model_name or model_path.name
    grading_url = f"http://localhost:{args.port}"

    print(f"Model Path: {model_path}")
    print(f"Model Name: {model_name}")
    print(f"Task: {args.task}")
    print(f"Grading URL: {grading_url}")
    print("-" * 50)

    # 使用固定 workspace
    from rdagent.scenarios.rl.autorl_bench.conf import get_workspace_dir

    workspace = get_workspace_dir() / args.task
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"Workspace: {workspace}")

    # 启动 grading_server
    import threading

    from rdagent.scenarios.rl.autorl_bench.core.server import app, init_server

    server = init_server(args.task, model_name, str(workspace))

    print(f"Starting grading server on port {args.port}...")
    server_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=args.port, debug=False, threaded=False), daemon=True
    )
    server_thread.start()

    # 等待 server 启动
    for i in range(10):
        time.sleep(0.5)
        try:
            resp = requests.get(f"{grading_url}/health", timeout=2)
            if resp.status_code == 200:
                print(f"Grading server started.")
                break
        except:
            pass
    else:
        print("[ERROR] Grading server failed to start")
        return 1

    # 提交评测
    print("-" * 50)
    print("Submitting model for evaluation...")
    print(f"POST {grading_url}/submit")

    start_time = time.time()
    resp = requests.post(
        f"{grading_url}/submit",
        json={"model_path": str(model_path)},
        timeout=3600,
    )
    elapsed = time.time() - start_time

    print("-" * 50)
    print(f"Response status: {resp.status_code}")
    print(f"Elapsed: {elapsed:.2f}s")
    print("Result:")

    if resp.status_code == 200:
        result = resp.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        score = result.get("score", 0)
        print("-" * 50)
        if score > 0:
            print(f"[SUCCESS] Score: {score}")
        else:
            print(f"[FAILED] Score: {score}")
    else:
        print(f"Error response: {resp.text}")
        print("-" * 50)
        print(f"[ERROR] Server returned {resp.status_code}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

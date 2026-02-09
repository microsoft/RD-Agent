"""
Grading Server

提供 HTTP API 供 agent 提交模型进行评测。
- /submit: 提交模型评测
- /best: 获取最高分
- /history: 获取历史记录
- /health: 健康检查

日志文件：
- scores.json: 评测结果（含 submission_id、baseline_score、improvement）
- interactions.jsonl: 所有 HTTP 交互记录
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from loguru import logger

app = Flask(__name__)

# 配置
TASK = os.environ.get("TASK", "")
BASE_MODEL = os.environ.get("BASE_MODEL", "")
WORKSPACE = Path(os.environ.get("WORKSPACE", "."))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", WORKSPACE / "output"))
SCORES_FILE = WORKSPACE / "scores.json"
INTERACTIONS_FILE = WORKSPACE / "interactions.jsonl"
BASELINE_SCORE_FILE = WORKSPACE / "baseline_score.json"

# 配置 loguru
logger.add(
    WORKSPACE / "grading_server.log",
    rotation="10 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)


def log_interaction(endpoint: str, method: str, request_data: dict, response_data: dict, elapsed_seconds: float):
    """记录 HTTP 交互到 interactions.jsonl"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "method": method,
        "request": request_data,
        "response": response_data,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    with open(INTERACTIONS_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_scores() -> list[dict]:
    if SCORES_FILE.exists():
        return json.loads(SCORES_FILE.read_text())
    return []


def save_scores(scores: list[dict]):
    SCORES_FILE.write_text(json.dumps(scores, indent=2, ensure_ascii=False))


def get_baseline_score() -> float | None:
    """获取 baseline score（首次评测时缓存）"""
    if BASELINE_SCORE_FILE.exists():
        data = json.loads(BASELINE_SCORE_FILE.read_text())
        return data.get("score")
    return None


def set_baseline_score(score: float):
    """设置 baseline score"""
    BASELINE_SCORE_FILE.write_text(json.dumps({"score": score, "timestamp": datetime.now().isoformat()}))


@app.route("/health", methods=["GET"])
def health():
    start_time = time.time()
    response = {"status": "ok", "task": TASK, "workspace": str(WORKSPACE)}
    
    log_interaction("/health", "GET", {}, response, time.time() - start_time)
    return jsonify(response)


@app.route("/validate", methods=["POST"])
def validate():
    start_time = time.time()
    data = request.get_json() or {}
    model_path = Path(data.get("model_path", str(OUTPUT_DIR)))
    
    logger.info(f"[VALIDATE] model_path={model_path}")
    
    if not model_path.exists():
        response = {"valid": False, "error": f"Path not found: {model_path}"}
        log_interaction("/validate", "POST", data, response, time.time() - start_time)
        return jsonify(response)
    
    files = [f.name for f in model_path.iterdir()] if model_path.is_dir() else []
    model_files = ["config.json", "pytorch_model.bin", "model.safetensors", "adapter_config.json"]
    has_model = any(f in files for f in model_files)
    
    response = {"valid": has_model, "files": files}
    log_interaction("/validate", "POST", data, response, time.time() - start_time)
    logger.info(f"[VALIDATE] valid={has_model}, files={len(files)}")
    
    return jsonify(response)


@app.route("/submit", methods=["POST"])
def submit():
    from rdagent.scenarios.rl.autorl_bench.benchmark import run_benchmark
    
    start_time = time.time()
    data = request.get_json() or {}
    model_path = data.get("model_path", str(OUTPUT_DIR))
    
    # 获取 submission_id
    scores = load_scores()
    submission_id = len(scores) + 1
    
    logger.info(f"[SUBMIT #{submission_id}] Started | model_path={model_path}")
    
    # 运行评测（全量）
    result = run_benchmark(
        workspace_path=str(WORKSPACE),
        model_path=model_path,
        model_name=BASE_MODEL,
        benchmark_name=TASK,
        test_range="[:]",
    )
    
    elapsed_seconds = time.time() - start_time
    
    # 解析分数
    score = 0.0
    if "accuracy_summary" in result:
        acc = result["accuracy_summary"]
        score = acc.get("accuracy") or acc.get("score") or 0.0
    else:
        score = result.get("score") or result.get("accuracy") or 0.0
    
    # baseline score 和 improvement
    baseline_score = get_baseline_score()
    improvement = None
    if baseline_score is not None and score is not None:
        improvement = round(score - baseline_score, 6)
    
    # 构建结果
    entry = {
        "submission_id": submission_id,
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "score": score,
        "baseline_score": baseline_score,
        "improvement": improvement,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "metrics": result,
    }
    
    scores.append(entry)
    save_scores(scores)
    
    # 记录交互日志
    log_interaction("/submit", "POST", data, {
        "submission_id": submission_id,
        "score": score,
        "improvement": improvement,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }, elapsed_seconds)
    
    logger.info(f"[SUBMIT #{submission_id}] Done | score={score}, improvement={improvement}, elapsed={elapsed_seconds:.2f}s")
    
    return jsonify(entry)


@app.route("/best", methods=["GET"])
def best():
    # TODO:
    # It is not general enough
    # _function_name
    start_time = time.time()
    scores = load_scores()
    
    if not scores:
        response = {"error": "No submissions"}
        log_interaction("/best", "GET", {}, response, time.time() - start_time)
        return jsonify(response), 404
    
    best_entry = max(scores, key=lambda x: x.get("score") or float("-inf"))
    response = {"best": best_entry, "total_submissions": len(scores)}
    
    log_interaction("/best", "GET", {}, {"best_score": best_entry.get("score"), "total": len(scores)}, time.time() - start_time)
    return jsonify(response)


@app.route("/history", methods=["GET"])
def history():
    start_time = time.time()
    scores = load_scores()
    
    log_interaction("/history", "GET", {}, {"count": len(scores)}, time.time() - start_time)
    return jsonify(scores)


@app.route("/set_baseline", methods=["POST"])
def set_baseline():
    """设置 baseline score（通常在 agent 开始前由 run_agent.py 调用）"""
    start_time = time.time()
    data = request.get_json() or {}
    score = data.get("score")
    
    if score is None:
        response = {"error": "Missing 'score' field"}
        log_interaction("/set_baseline", "POST", data, response, time.time() - start_time)
        return jsonify(response), 400
    
    set_baseline_score(float(score))
    response = {"baseline_score": score, "status": "set"}
    
    log_interaction("/set_baseline", "POST", data, response, time.time() - start_time)
    logger.info(f"[SET_BASELINE] baseline_score={score}")
    
    return jsonify(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    TASK = args.task or TASK
    
    logger.info(f"Grading Server Starting | task={TASK} | {args.host}:{args.port}")
    logger.info(f"  WORKSPACE: {WORKSPACE}")
    logger.info(f"  SCORES_FILE: {SCORES_FILE}")
    logger.info(f"  INTERACTIONS_FILE: {INTERACTIONS_FILE}")
    
    print(f"Grading Server | Task: {TASK} | {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)

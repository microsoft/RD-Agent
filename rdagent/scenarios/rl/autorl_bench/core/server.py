"""
AutoRL-Bench Grading Server (Simplified)

精简的评测服务，主要提供 submit 接口。
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request
from loguru import logger

app = Flask(__name__)


class GradingServer:
    """评测服务器"""
    
    def __init__(
        self,
        task: str,
        base_model: str,
        workspace: Path,
    ):
        self.task = task
        self.base_model = base_model
        self.workspace = Path(workspace)
        self.scores_file = self.workspace / "scores.json"
        self.baseline_score: Optional[float] = None
        
        # 配置日志
        logger.add(
            self.workspace / "grading_server.log",
            rotation="10 MB",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO"
        )
    
    def load_scores(self) -> list[dict]:
        if self.scores_file.exists():
            return json.loads(self.scores_file.read_text())
        return []
    
    def save_scores(self, scores: list[dict]):
        self.scores_file.write_text(json.dumps(scores, indent=2, ensure_ascii=False))
    
    def get_evaluator(self):
        """获取当前 task 的评测器"""
        from rdagent.scenarios.rl.autorl_bench.benchmarks import get_evaluator
        return get_evaluator(self.task)
    
    def submit(self, model_path: str) -> dict:
        """
        提交模型评测
        
        Returns:
            包含 score、best、improvement 等完整信息的结果
        """
        start_time = time.time()
        scores = self.load_scores()
        submission_id = len(scores) + 1
        
        logger.info(f"[SUBMIT #{submission_id}] Started | model_path={model_path}")
        
        # 运行评测
        evaluator = self.get_evaluator()
        result = evaluator.run_eval(
            model_path=model_path,
            workspace_path=str(self.workspace),
            model_name=self.base_model,
        )
        
        elapsed_seconds = time.time() - start_time
        
        # 解析分数
        score = result.get("score", 0.0)
        
        # 计算 improvement
        improvement = None
        if self.baseline_score is not None:
            improvement = round(score - self.baseline_score, 6)
        
        # 构建结果
        entry = {
            "submission_id": submission_id,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "score": score,
            "baseline_score": self.baseline_score,
            "improvement": improvement,
            "elapsed_seconds": round(elapsed_seconds, 2),
        }
        
        scores.append(entry)
        self.save_scores(scores)
        
        # 查找最高分
        best_entry = max(scores, key=lambda x: x.get("score", 0))
        
        logger.info(f"[SUBMIT #{submission_id}] Done | score={score}, best={best_entry['score']}")
        
        return {
            **entry,
            "best": best_entry,
            "total_submissions": len(scores),
        }
    
    def set_baseline(self, score: float):
        """设置 baseline 分数"""
        self.baseline_score = score
        logger.info(f"[BASELINE] Set to {score}")


# 全局服务器实例
_server: Optional[GradingServer] = None


def get_server() -> GradingServer:
    global _server
    if _server is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return _server


def init_server(task: str, base_model: str, workspace: str) -> GradingServer:
    """初始化服务器"""
    global _server
    _server = GradingServer(task, base_model, Path(workspace))
    return _server


# Flask 路由
@app.route("/submit", methods=["POST"])
def submit():
    """
    提交模型评测
    
    Request:
        {"model_path": "/path/to/model"}
        
    Response:
        {
            "submission_id": 1,
            "score": 85.0,
            "improvement": 5.0,
            "best": {...},
            "total_submissions": 10
        }
    """
    data = request.get_json() or {}
    model_path = data.get("model_path")
    
    if not model_path:
        return jsonify({"error": "Missing model_path"}), 400
    
    server = get_server()
    result = server.submit(model_path)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    server = get_server()
    return jsonify({
        "status": "ok",
        "task": server.task,
        "workspace": str(server.workspace),
    })


@app.route("/set_baseline", methods=["POST"])
def set_baseline():
    """设置 baseline 分数"""
    data = request.get_json() or {}
    score = data.get("score")
    
    if score is None:
        return jsonify({"error": "Missing score"}), 400
    
    server = get_server()
    server.set_baseline(float(score))
    return jsonify({"baseline_score": score, "status": "set"})


def run_server(task: str, base_model: str, workspace: str, host: str = "0.0.0.0", port: int = 5000):
    """启动服务器"""
    init_server(task, base_model, workspace)
    logger.info(f"Grading Server | task={task} | {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    run_server(args.task, args.base_model, args.workspace, args.host, args.port)

"""
AutoRL-Bench Grading Server (Simplified)

精简的评测服务，主要提供 submit 接口。
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

from flask import Flask, jsonify, request

from rdagent.log import rdagent_logger as logger

app = Flask(__name__)


def _get_available_gpus() -> Set[str]:
    """从 CUDA_VISIBLE_DEVICES 获取可用 GPU 集合"""
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_env.strip():
        return set()
    return {g.strip() for g in cuda_env.split(",") if g.strip()}


def _validate_gpu(gpu: str, available: Set[str]) -> Optional[str]:
    """校验 gpu 参数，返回错误信息或 None（合法）"""
    requested = {g.strip() for g in gpu.split(",") if g.strip()}
    if not requested:
        return "gpu parameter is empty"
    invalid = requested - available
    if invalid:
        return f"GPU {invalid} not in available GPUs {sorted(available)} (from CUDA_VISIBLE_DEVICES)"
    return None


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
        self.available_gpus: Set[str] = _get_available_gpus()
    
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
    
    def submit(self, model_path: str, gpu: Optional[str] = None) -> dict:
        """
        提交模型评测
        
        Args:
            model_path: 模型路径
            gpu: 指定 GPU（如 "0", "1", "0,1"），必须是 CUDA_VISIBLE_DEVICES 中的子集。
                 None 则使用 CUDA_VISIBLE_DEVICES 中的第一个 GPU。
            
        Returns:
            包含 score、best、improvement 等完整信息的结果
            
        Raises:
            ValueError: gpu 不在 CUDA_VISIBLE_DEVICES 范围内
        """
        if self.available_gpus:
            if gpu is None:
                gpu = sorted(self.available_gpus, key=int)[0]
            else:
                err = _validate_gpu(gpu, self.available_gpus)
                if err:
                    raise ValueError(err)
        
        start_time = time.time()
        scores = self.load_scores()
        submission_id = len(scores) + 1
        
        logger.info(f"[SUBMIT #{submission_id}] Started | model_path={model_path} | gpu={gpu}")
        
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        try:
            evaluator = self.get_evaluator()
            result = evaluator.run_eval(
                model_path=model_path,
                workspace_path=str(self.workspace),
                model_name=self.base_model,
            )
        finally:
            if old_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        
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
    gpu = data.get("gpu")
    
    if not model_path:
        return jsonify({"error": "Missing model_path"}), 400
    
    server = get_server()
    
    if gpu is not None:
        gpu = str(gpu)
        err = _validate_gpu(gpu, server.available_gpus)
        if err:
            return jsonify({
                "error": err,
                "available_gpus": sorted(server.available_gpus, key=int),
            }), 400
    
    result = server.submit(model_path, gpu=gpu)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    server = get_server()
    return jsonify({
        "status": "ok",
        "task": server.task,
        "workspace": str(server.workspace),
        "available_gpus": sorted(server.available_gpus, key=int) if server.available_gpus else [],
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

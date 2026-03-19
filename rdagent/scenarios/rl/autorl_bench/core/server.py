"""
AutoRL-Bench Grading Server (Simplified)

A streamlined evaluation service that mainly provides the submit interface.
"""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

import requests
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.core.utils import read_run_meta, update_run_meta

app = Flask(__name__)


def _get_available_gpus() -> Set[str]:
"""Get the available GPU collection from CUDA_VISIBLE_DEVICES"""
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_env.strip():
        return set()
    return {g.strip() for g in cuda_env.split(",") if g.strip()}


def _validate_gpu(gpu: str, available: Set[str]) -> Optional[str]:
"""Verify gpu parameters, return error message or None (legal)"""
    requested = {g.strip() for g in gpu.split(",") if g.strip()}
    if not requested:
        return "gpu parameter is empty"
    invalid = requested - available
    if invalid:
        return f"GPU {invalid} not in available GPUs {sorted(available)} (from CUDA_VISIBLE_DEVICES)"
    return None


class GradingServer:
"""Evaluation server"""

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
        self._eval_lock = threading.Lock()
        self._eval_cache: dict[str, dict] = {}

    @staticmethod
    def _make_cache_key(resolved_path: Path) -> str:
"""Use the path + the latest mtime combination of the safetensors/bin file as the cache key.
If the mtime changes after the model is overwritten, the cache will automatically expire. """
        mtime = 0.0
        if resolved_path.is_dir():
            for f in resolved_path.rglob("*"):
                if f.suffix in (".safetensors", ".bin", ".json") and f.is_file():
                    mt = f.stat().st_mtime
                    if mt > mtime:
                        mtime = mt
        elif resolved_path.is_file():
            mtime = resolved_path.stat().st_mtime
        return f"{resolved_path}@{mtime}"

    def load_scores(self) -> list[dict]:
        if self.scores_file.exists():
            return json.loads(self.scores_file.read_text())
        return []

    def save_scores(self, scores: list[dict]):
        self.scores_file.write_text(json.dumps(scores, indent=2, ensure_ascii=False))

    def get_evaluator(self):
"""Get the evaluator of the current task"""
        from rdagent.scenarios.rl.autorl_bench.benchmarks import get_evaluator

        return get_evaluator(self.task)

    def resolve_model_path(self, model_path: str) -> Path:
"""Constrain the model path to the workspace to prevent access to arbitrary file system paths."""
        if "\x00" in model_path:
            raise ValueError("Invalid model_path")

        workspace_root = self.workspace.expanduser().resolve()
        normalized = os.path.normpath(model_path)
        if os.path.splitdrive(normalized)[0]:
            raise ValueError("Invalid model_path")

        if os.path.isabs(normalized):
            candidate = normalized
        else:
            candidate = os.path.join(str(workspace_root), normalized)

        resolved_path = Path(candidate).expanduser().resolve(strict=False)
        try:
            resolved_path.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError("Invalid model_path") from exc
        return resolved_path

    def submit(self, model_path: str, gpu: Optional[str] = None) -> dict:
        """
Submit model review

        Args:
model_path: model path
gpu: Specifies the GPU (such as "0", "1", "0,1"), which must be a subset of CUDA_VISIBLE_DEVICES.
None uses the first GPU in CUDA_VISIBLE_DEVICES.

        Returns:
Results containing complete information such as score, best, improvement, etc.

        Raises:
ValueError: gpu is not in the range of CUDA_VISIBLE_DEVICES, or model_path is illegal
        """
        if self.available_gpus:
            if gpu is None:
                gpu = sorted(self.available_gpus, key=int)[0]
            else:
                err = _validate_gpu(gpu, self.available_gpus)
                if err:
                    raise ValueError(err)

# B3 fix: Same model_path + same content, deduplication, return cached results directly
# Use the path + the latest mtime of the model file as the cache key. The model file will automatically expire after being overwritten.
        resolved_path = self.resolve_model_path(model_path)
        cache_key = self._make_cache_key(resolved_path)
        if cache_key in self._eval_cache:
            cached = self._eval_cache[cache_key]
            logger.info(f"[SUBMIT] Cache hit for {model_path}, score={cached.get('score')}")
            return cached

        start_time = time.time()

# B2 fix: Serialized evaluation to prevent multiple vLLM instances from grabbing the GPU at the same time
        with self._eval_lock:
# Double-check: The review may have been completed by other threads while waiting for the lock.
            cache_key = self._make_cache_key(resolved_path)
            if cache_key in self._eval_cache:
                cached = self._eval_cache[cache_key]
                logger.info(f"[SUBMIT] Cache hit (after lock) for {model_path}, score={cached.get('score')}")
                return cached

            scores = self.load_scores()
            submission_id = len(scores) + 1

            logger.info(f"[SUBMIT #{submission_id}] Started | model_path={model_path} | gpu={gpu}")

            old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            if gpu is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

            try:
                evaluator = self.get_evaluator()
                gpu_count = len(self.available_gpus) if self.available_gpus else 1
                result = evaluator.run_eval(
                    model_path=str(resolved_path),
                    workspace_path=str(self.workspace),
                    model_name=self.base_model,
                    gpu_count=gpu_count,
                )
            finally:
                if old_cuda is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

        elapsed_seconds = time.time() - start_time

# Parse scores
        score = result.get("score", 0.0)
        error = result.get("error")

# Calculate improvement
        improvement = None
        if self.baseline_score is not None:
            improvement = round(score - self.baseline_score, 6)

# Build results
        entry = {
            "submission_id": submission_id,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "score": score,
            "baseline_score": self.baseline_score,
            "improvement": improvement,
            "elapsed_seconds": round(elapsed_seconds, 2),
        }
# B4 fix: transparent transmission of error field
        if error:
            entry["error"] = error

        scores.append(entry)
        self.save_scores(scores)
        update_run_meta(self.workspace, last_submit_time=int(time.time()))

# Find the highest score
        best_entry = max(scores, key=lambda x: x.get("score", 0))

        logger.info(f"[SUBMIT #{submission_id}] Done | score={score}, best={best_entry['score']}")

        response = {
            **entry,
            "best": best_entry,
            "total_submissions": len(scores),
        }

# Only cache successful evaluation results (failed ones are not cached and retry is allowed)
        if not error:
            self._eval_cache[self._make_cache_key(resolved_path)] = response

        return response

    def set_baseline(self, score: float):
"""Set baseline score"""
        self.baseline_score = score
        logger.info(f"[BASELINE] Set to {score}")


# Global server instance
_server: Optional[GradingServer] = None


def get_server() -> GradingServer:
    global _server
    if _server is None:
        raise RuntimeError("Server not initialized. Call init_server() first.")
    return _server


def init_server(task: str, base_model: str, workspace: str) -> GradingServer:
"""Initialize server"""
    global _server
    _server = GradingServer(task, base_model, Path(workspace))
    return _server


# Flask routing
@app.route("/submit", methods=["POST"])
def submit():
    """
Submit model review

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
            return (
                jsonify(
                    {
                        "error": err,
                        "available_gpus": sorted(server.available_gpus, key=int),
                    }
                ),
                400,
            )

    try:
        result = server.submit(model_path, gpu=gpu)
        return jsonify(result)
    except ValueError:
        logger.warning("[SUBMIT] Invalid request", exc_info=True)
        return jsonify({"error": "Invalid request"}), 400
    except (RuntimeError, OSError):
        logger.exception("[SUBMIT] Internal server error")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health", methods=["GET"])
def health():
"""Health Check"""
    server = get_server()
    return jsonify(
        {
            "status": "ok",
            "task": server.task,
            "workspace": str(server.workspace),
            "available_gpus": sorted(server.available_gpus, key=int) if server.available_gpus else [],
        }
    )


@app.route("/time", methods=["GET"])
def time_status():
"""Time and budget signals"""
    server = get_server()
    meta = read_run_meta(server.workspace)
    now = int(time.time())
    timeout_s = meta.get("timeout_s")
    start_time = meta.get("start_time")
    remaining = None
    if isinstance(timeout_s, int) and isinstance(start_time, int):
        remaining = max(timeout_s - (now - start_time), 0)
    return jsonify(
        {
            **meta,
            "now": now,
            "remaining": remaining,
        }
    )


@app.route("/set_baseline", methods=["POST"])
def set_baseline():
"""Set baseline score"""
    data = request.get_json() or {}
    score = data.get("score")

    if score is None:
        return jsonify({"error": "Missing score"}), 400

    server = get_server()
    server.set_baseline(float(score))
    return jsonify({"baseline_score": score, "status": "set"})


def run_server(task: str, base_model: str, workspace: str, host: str = "0.0.0.0", port: int = 5000):
"""Start the server"""
    init_server(task, base_model, workspace)
    logger.info(f"Grading Server | task={task} | {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


# ============================================================
# Grading Server context manager
# ============================================================


class GradingServerContext:
"""Grading Server base class"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_baseline(self, task: str, model_name: str, model_path: str, workspace_path: str) -> float:
        raise NotImplementedError

    def load_scores(self) -> list:
        raise NotImplementedError


class LocalServerContext(GradingServerContext):
"""Local Flask Server"""

    def __init__(self, task: str, base_model: str, workspace: str, port: int):
        self.task = task
        self.base_model = base_model
        self.workspace = workspace
        self.port = port
        self.server = None
        self._http_server = None
        self._thread = None

    def __enter__(self):
        logger.info(f"[Local Mode] Starting evaluation server on port {self.port}...")
        self.server = init_server(self.task, self.base_model, self.workspace)

        self._http_server = make_server("0.0.0.0", self.port, app, threaded=True)
        self._thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._thread.start()

        # Poll /health for up to 15 seconds instead of blind sleep(2)
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                resp = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Grading server failed to start on port {self.port}")

        return self

    def __exit__(self, *args):
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None

    def get_baseline(self, task: str, model_name: str, model_path: str, workspace_path: str) -> float:
        from rdagent.scenarios.rl.autorl_bench.core.utils import get_baseline_score

        baseline = get_baseline_score(task, model_name, model_path, workspace_path)
        self.server.set_baseline(baseline)
        return baseline

    def load_scores(self) -> list:
        return self.server.load_scores() if self.server else []


def create_grading_server(benchmark, workspace: Path, port: int, base_model: str) -> GradingServerContext:
"""Create Grading Server context"""
    return LocalServerContext(
        task=benchmark.id,
        base_model=base_model,
        workspace=str(workspace),
        port=port,
    )


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

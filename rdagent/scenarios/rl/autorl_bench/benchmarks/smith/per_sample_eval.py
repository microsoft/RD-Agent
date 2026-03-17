"""Per-sample evaluator for smith benchmarks (arc_agi, zero_shot_cot).

Loads a model via vLLM, runs inference on each test sample, then uses the
benchmark's eval.py to score each prediction individually.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator


class PerSampleEvaluator(BaseEvaluator):
    """Evaluator that scores each sample individually using benchmark-specific eval.py."""

    def __init__(self, config):
        self.config = config
        self.benchmark_id = config.id
        self.eval_config = config.eval_config or {}

    def run_eval(
        self,
        model_path: str,
        workspace_path: str,
        model_name: str = "",
        gpu_count: int = 1,
        test_range: str = "[:]",
        **kwargs,
    ) -> Dict[str, Any]:
        result = self.get_default_result(self.benchmark_id, model_path)
        result["eval_type"] = "per_sample"

        if not self.validate_model(model_path):
            result["error"] = f"Model not found: {model_path}"
            return result

        # Load the benchmark-specific eval module
        eval_script = self.eval_config.get("eval_script", "")
        eval_module_path = self.eval_config.get("eval_module", "")
        if not eval_script and not eval_module_path:
            result["error"] = "No eval_script or eval_module configured"
            return result

        try:
            if eval_script:
                spec = importlib.util.spec_from_file_location("eval", eval_script)
                eval_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(eval_mod)
            else:
                eval_mod = importlib.import_module(eval_module_path)
        except Exception as e:
            result["error"] = f"Cannot load eval module: {e}"
            return result

        # Load test data
        workspace = Path(workspace_path)
        data_dir = workspace / "data"
        test_file = data_dir / "train.jsonl"
        if not test_file.exists():
            result["error"] = f"Test data not found: {test_file}"
            return result

        test_data = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    test_data.append(json.loads(line))

        # Apply test_range slicing
        test_data = _apply_range(test_data, test_range)

        if not test_data:
            result["error"] = "No test data after applying range"
            return result

        logger.info(f"[{self.benchmark_id}] Running per-sample eval on {len(test_data)} samples")

        # Load model and run inference via vLLM
        try:
            import vllm
            from vllm import SamplingParams

            llm = vllm.LLM(model=model_path, tensor_parallel_size=gpu_count)
            sampling_params = SamplingParams(temperature=0, max_tokens=2048)

            prompts = []
            for item in test_data:
                q = item.get("question", "")
                if isinstance(q, dict):
                    # For arc_agi: question is a JSON object, stringify it
                    q = json.dumps(q)
                prompts.append(q)

            outputs = llm.generate(prompts, sampling_params)
        except Exception as e:
            # Clean up vLLM GPU memory even on failure
            if "llm" in locals():
                _cleanup_vllm(llm)
            result["error"] = f"vLLM inference failed: {e}"
            return result

        # Release vLLM GPU memory to avoid OOM for subsequent evaluations
        _cleanup_vllm(llm)

        # Score each sample
        total = 0
        correct = 0.0
        for item, output in zip(test_data, outputs):
            model_answer = output.outputs[0].text
            question = item.get("question", "")
            reference = item.get("answer", "")

            # Pass extra kwargs from the item (e.g. answer_type for zero_shot_cot)
            extra = {k: v for k, v in item.items() if k not in ("question", "answer")}
            try:
                score = eval_mod.evaluate(question, model_answer, reference, **extra)
            except Exception as e:
                logger.warning(f"Eval error on sample: {e}")
                score = 0.0

            correct += score
            total += 1

        accuracy = (correct / total) * 100 if total > 0 else 0.0
        result["score"] = accuracy
        result["accuracy_summary"] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }

        logger.info(f"[{self.benchmark_id}] Score: {accuracy:.2f}% ({correct}/{total})")
        return result


def _cleanup_vllm(llm) -> None:
    """Release vLLM GPU memory without initializing CUDA in the main process.

    We delete the LLM object and run torch.cuda.empty_cache() inside a
    *spawned* subprocess so that the main process never touches CUDA directly.
    This avoids the 'Cannot re-initialize CUDA in forked subprocess' error
    that OpenCompass would hit later when it forks inference workers.
    """
    import multiprocessing as mp

    def _gpu_cleanup():
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        del llm
    except Exception:
        pass

    try:
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_gpu_cleanup)
        p.start()
        p.join(timeout=30)
    except Exception:
        pass


def _apply_range(data: list, test_range: str) -> list:
    """Apply a Python-style slice string like '[:]' or '[:100]' to a list."""
    test_range = test_range.strip()
    if not test_range or test_range == "[:]":
        return data
    try:
        # Parse "[start:stop]" or "[:stop]" etc.
        inner = test_range.strip("[]")
        parts = inner.split(":")
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        return data[start:stop]
    except Exception:
        return data

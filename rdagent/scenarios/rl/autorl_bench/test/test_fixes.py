"""
测试 B1-B4 修复

验证:
  B1: LoRA adapter 自动检测
  B2: 评测锁（串行化 GPU 访问）
  B3: model_path 去重缓存
  B4: error 字段透传

运行: python -m rdagent.scenarios.rl.autorl_bench.test.test_fixes
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

PASS = 0
FAIL = 0


def report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


# ============================================================
# B1: LoRA adapter 自动检测
# ============================================================
def test_b1_lora_detection():
    print("\n=== B1: LoRA adapter detection ===")
    from rdagent.scenarios.rl.autorl_bench.core.opencompass import OpenCompassEvaluator

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "lora_output"
        adapter_dir.mkdir()
        base_model_dir = Path(tmpdir) / "base_model"
        base_model_dir.mkdir()
        (base_model_dir / "config.json").write_text("{}")

        # Case 1: adapter_config.json 存在且 base model 存在
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": str(base_model_dir)
        }))

        config = MagicMock()
        config.id = "gsm8k"
        config.eval_config = {}
        evaluator = OpenCompassEvaluator(config)

        with patch.object(evaluator, '_get_model_inference_config', return_value={
            "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9,
            "dtype": "auto", "max_seq_len": 4096, "max_out_len": 512,
            "batch_size": 8, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "repetition_penalty": 1.0, "enable_thinking": False,
            "use_cot_postprocessor": False,
        }), patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="test", stdout="")
            result = evaluator.run_eval(
                model_path=str(adapter_dir),
                workspace_path=tmpdir,
                model_name="test-model",
            )
            if mock_run.called:
                config_path = Path(tmpdir) / "opencompass_config.py"
                if config_path.exists():
                    content = config_path.read_text()
                    report("LoRA detected → is_lora=True in config",
                           "enable_lora=True" in content,
                           f"config has enable_lora={'enable_lora=True' in content}")
                    report("lora_path set in config",
                           "lora_path=" in content,
                           f"lora_path found={'lora_path=' in content}")
                    report("model_path points to base model",
                           str(base_model_dir) in content,
                           f"base_model in config={str(base_model_dir) in content}")
                else:
                    report("OpenCompass config generated", False, "config file not found")
            else:
                report("OpenCompass was called", False, "subprocess.run not called")

        # Case 2: adapter_config.json with missing base model
        bad_adapter_dir = Path(tmpdir) / "bad_lora"
        bad_adapter_dir.mkdir()
        (bad_adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "/nonexistent/model"
        }))
        result = evaluator.run_eval(
            model_path=str(bad_adapter_dir),
            workspace_path=tmpdir,
            model_name="test-model",
        )
        report("Missing base model → returns error",
               "error" in result and "not found" in result["error"],
               result.get("error", "no error"))

        # Case 3: normal model (no adapter_config.json) — should NOT set is_lora
        normal_dir = Path(tmpdir) / "normal_model"
        normal_dir.mkdir()
        (normal_dir / "config.json").write_text("{}")
        with patch.object(evaluator, '_get_model_inference_config', return_value={
            "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9,
            "dtype": "auto", "max_seq_len": 4096, "max_out_len": 512,
            "batch_size": 8, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "repetition_penalty": 1.0, "enable_thinking": False,
            "use_cot_postprocessor": False,
        }), patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="test", stdout="")
            evaluator.run_eval(
                model_path=str(normal_dir),
                workspace_path=tmpdir,
                model_name="test-model",
            )
            config_path = Path(tmpdir) / "opencompass_config.py"
            if config_path.exists():
                content = config_path.read_text()
                report("Normal model → no enable_lora",
                       "enable_lora" not in content,
                       f"enable_lora absent={'enable_lora' not in content}")


# ============================================================
# B2+B3: 评测锁 + 去重缓存
# ============================================================
def test_b2b3_lock_and_cache():
    print("\n=== B2+B3: Eval lock + dedup cache ===")
    from rdagent.scenarios.rl.autorl_bench.core.server import GradingServer

    with tempfile.TemporaryDirectory() as tmpdir:
        server = GradingServer("gsm8k", "test-model", Path(tmpdir))

        report("Server has _eval_lock", hasattr(server, '_eval_lock'))
        report("Server has _eval_cache", hasattr(server, '_eval_cache'))

        # Mock evaluator to track concurrency
        call_log = []
        active_count = [0]
        max_concurrent = [0]

        def mock_run_eval(**kwargs):
            active_count[0] += 1
            max_concurrent[0] = max(max_concurrent[0], active_count[0])
            call_log.append(kwargs.get("model_path", ""))
            time.sleep(0.3)
            active_count[0] -= 1
            return {"score": 85.0, "accuracy_summary": {}}

        mock_evaluator = MagicMock()
        mock_evaluator.run_eval = mock_run_eval

        with patch.object(server, 'get_evaluator', return_value=mock_evaluator):
            # B2 test: concurrent submits should be serialized
            model_a = Path(tmpdir) / "model_a"
            model_b = Path(tmpdir) / "model_b"
            model_a.mkdir()
            model_b.mkdir()
            (model_a / "config.json").write_text("{}")
            (model_b / "config.json").write_text("{}")

            threads = []
            results = []

            def submit_wrapper(mp):
                r = server.submit(str(mp))
                results.append(r)

            t1 = threading.Thread(target=submit_wrapper, args=(model_a,))
            t2 = threading.Thread(target=submit_wrapper, args=(model_b,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            report("B2: max concurrent evals = 1 (lock works)",
                   max_concurrent[0] == 1,
                   f"max_concurrent={max_concurrent[0]}")
            report("B2: both evaluations completed",
                   len(results) == 2,
                   f"results={len(results)}")

            # B3 test: same model_path should hit cache
            call_log.clear()
            server.submit(str(model_a))  # should hit cache

            report("B3: duplicate submit uses cache (no re-eval)",
                   str(model_a.resolve()) not in [str(Path(p).resolve()) for p in call_log],
                   f"call_log after cache hit: {call_log}")

            # B3 test: failed eval should NOT be cached
            def mock_fail_eval(**kwargs):
                return {"score": 0.0, "error": "GPU OOM", "accuracy_summary": {}}

            mock_evaluator.run_eval = mock_fail_eval
            fail_model = Path(tmpdir) / "fail_model"
            fail_model.mkdir()
            (fail_model / "config.json").write_text("{}")

            r1 = server.submit(str(fail_model))
            report("B3: failed eval not cached",
                   str(fail_model.resolve()) not in server._eval_cache,
                   f"cached={str(fail_model.resolve()) in server._eval_cache}")


# ============================================================
# B4: error 字段透传
# ============================================================
def test_b4_error_passthrough():
    print("\n=== B4: Error field passthrough ===")
    from rdagent.scenarios.rl.autorl_bench.core.server import GradingServer

    with tempfile.TemporaryDirectory() as tmpdir:
        server = GradingServer("gsm8k", "test-model", Path(tmpdir))

        def mock_error_eval(**kwargs):
            return {
                "score": 0.0,
                "error": "vLLM model load failed: config.json not found",
                "accuracy_summary": {},
            }

        mock_evaluator = MagicMock()
        mock_evaluator.run_eval = mock_error_eval

        model_dir = Path(tmpdir) / "error_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        with patch.object(server, 'get_evaluator', return_value=mock_evaluator):
            result = server.submit(str(model_dir))
            report("Error field present in response",
                   "error" in result,
                   f"error={result.get('error', 'MISSING')}")
            report("Score is 0.0",
                   result.get("score") == 0.0)

    # Test _parse_results with non-numeric values (B4 in opencompass)
    from rdagent.scenarios.rl.autorl_bench.core.opencompass import OpenCompassEvaluator
    import pandas as pd

    config = MagicMock()
    config.id = "gsm8k"
    config.eval_config = {}
    evaluator = OpenCompassEvaluator(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        ts_dir = work_dir / "20260306_120000"
        summary_dir = ts_dir / "summary"
        summary_dir.mkdir(parents=True)

        csv_path = summary_dir / "results.csv"
        df = pd.DataFrame({"dataset": ["gsm8k"], "rl-gsm8k": ["-"]})
        df.to_csv(csv_path, index=False)

        result = {"score": 0.0, "accuracy_summary": {}, "benchmark": "gsm8k", "model_path": "/test"}
        result = evaluator._parse_results(work_dir, result)

        report("Non-numeric score → error field set",
               "error" in result,
               result.get("error", "MISSING")[:80])
        report("Score remains 0.0 on parse failure",
               result["score"] == 0.0)


def main():
    test_b1_lora_detection()
    test_b2b3_lock_and_cache()
    test_b4_error_passthrough()

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print(f"{'='*50}")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

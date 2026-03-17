"""
AlpacaEval 2.0 Evaluator

流程：
1. 读取 AlpacaEval 2.0 参考输出（gpt4 baseline）
2. 用 vLLM 生成模型输出
3. 调用 alpaca_eval 进行 head-to-head 评测（Length-Controlled Win Rate）
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator

DEFAULT_REFERENCE_FILE = "alpaca_eval_gpt4_baseline.json"
DEFAULT_ANNOTATORS_CONFIG = "weighted_alpaca_eval_gpt4_turbo"


class AlpacaEvalEvaluator(BaseEvaluator):
    """AlpacaEval 2.0 评测器（LLM Judge）"""

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
        result["eval_type"] = "alpacaeval"

        if not self.validate_model(model_path):
            result["error"] = f"Model not found: {model_path}"
            return result

        try:
            from alpaca_eval import evaluate as alpaca_evaluate
        except Exception as e:
            result["error"] = f"alpaca_eval import failed: {e}"
            return result

        # 1) Load reference outputs (AlpacaEval 2.0)
        reference_file = self.eval_config.get("reference_file", DEFAULT_REFERENCE_FILE)
        reference_outputs = self._load_reference_outputs(reference_file)

        # Optionally limit instances for quick eval
        max_instances = self.eval_config.get("max_instances")
        if isinstance(max_instances, int) and max_instances > 0:
            reference_outputs = reference_outputs[:max_instances]

        # 2) Generate model outputs with vLLM
        work_dir = Path(workspace_path) / "benchmark_results" / "alpacaeval"
        work_dir.mkdir(parents=True, exist_ok=True)
        model_outputs = self._generate_model_outputs(
            model_path=model_path,
            model_name=model_name,
            reference_outputs=reference_outputs,
            gpu_count=gpu_count,
        )
        try:
            (work_dir / "model_outputs.json").write_text(json.dumps(model_outputs, ensure_ascii=False, indent=2))
        except Exception:
            logger.warning("Failed to save AlpacaEval model outputs")

        # 3) AlpacaEval scoring
        annotators_config = self.eval_config.get("annotators_config", DEFAULT_ANNOTATORS_CONFIG)
        config_path = Path(annotators_config)
        if not config_path.is_absolute():
            local_path = Path(__file__).parent / annotators_config
            if local_path.exists():
                annotators_config = str(local_path)

        try:
            df_leaderboard, all_crossannotations = alpaca_evaluate(
                model_outputs=model_outputs,
                reference_outputs=reference_outputs,
                annotators_config=annotators_config,
                name=model_name or "model",
                output_path=str(work_dir),
                is_return_instead_of_print=True,
            )
        except Exception as e:
            result["error"] = f"alpaca_eval failed: {e}"
            return result

        # Extract score
        score, summary = self._extract_score(df_leaderboard, model_name or "model")
        result["score"] = score
        summary.update(
            {
                "num_samples": len(reference_outputs),
                "annotators_config": annotators_config,
                "reference_file": reference_file,
            }
        )
        result["accuracy_summary"] = summary

        logger.info(f"AlpacaEval score: {result['score']}")
        return result

    def _load_reference_outputs(self, filename: str) -> List[dict]:
        path = hf_hub_download(
            repo_id="tatsu-lab/alpaca_eval",
            repo_type="dataset",
            filename=filename,
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect list of dicts with keys: instruction, output, generator
        return data

    def _format_prompt(self, instruction: str, tokenizer) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": instruction}]
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return instruction

    def _generate_model_outputs(
        self,
        model_path: str,
        model_name: str,
        reference_outputs: List[dict],
        gpu_count: int,
    ) -> List[dict]:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        max_model_len = int(self.eval_config.get("max_model_len", 4096))
        max_tokens = int(self.eval_config.get("max_tokens", 512))

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tp_size = 1
        if gpu_count and gpu_count > 0:
            power = 0
            while (1 << (power + 1)) <= gpu_count:
                power += 1
            tp_size = 1 << power
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )

        prompts = [self._format_prompt(item["instruction"], tokenizer) for item in reference_outputs]
        outputs = llm.generate(prompts, sampling_params)

        model_outputs = []
        for item, out in zip(reference_outputs, outputs):
            text = out.outputs[0].text.strip() if out.outputs else ""
            model_outputs.append(
                {
                    "instruction": item.get("instruction", ""),
                    "output": text,
                    "generator": model_name or Path(model_path).name,
                    "dataset": item.get("dataset", ""),
                }
            )

        # Save raw outputs for debugging
        return model_outputs

    def _extract_score(self, df_leaderboard, name: str) -> tuple[float, dict]:
        row = None
        if name in df_leaderboard.index:
            row = df_leaderboard.loc[name]
        elif "model" in df_leaderboard.columns:
            matched = df_leaderboard[df_leaderboard["model"] == name]
            if not matched.empty:
                row = matched.iloc[0]

        if row is None:
            return 0.0, {"error": "model not found in leaderboard"}

        summary = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        score = None
        for key in ("length_controlled_winrate", "win_rate", "winrate"):
            if key in summary:
                score = summary[key]
                break
        if score is None:
            score = 0.0
        return float(score), summary

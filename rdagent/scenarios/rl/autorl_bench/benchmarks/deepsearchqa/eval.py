# benchmarks/deepsearchqa/eval.py

"""
DeepSearchQA Evaluator

使用 vLLM 加载本地模型，结合 Web Search 工具，
在 DeepSearchQA 数据集上评测模型的多步信息检索能力。

数据集: https://huggingface.co/datasets/google/deepsearchqa
评测方式: LLM Judge (推荐 gemini-2.5-flash)
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.benchmarks.deepsearchqa.data import (
    DATASET_NAME,
    DEFAULT_EVAL_SIZE,
    SOURCE_SPLIT,
    TRAIN_SIZE,
    load_source_dataset,
    split_dataset,
)
from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator

REACT_SYSTEM_PROMPT = """You are a research assistant that answers questions by searching the web.

You must follow this format strictly:
Thought: [your reasoning]
Action: search[your search query]   <- to search the web
  OR
Action: answer[Paris]   <- to give final answer

Rules:
- Always start with a Thought
- Use search[] to find information
- Use answer[] ONLY when you have enough information
- For Set Answer questions, list all items separated by commas
- Be concise and factual
"""


class DeepSearchQAEvaluator(BaseEvaluator):
    """
    DeepSearchQA 评测器

    流程：
    1. 从 HuggingFace 加载数据集
    2. 对每道题运行 ReAct 循环（模型 + 搜索工具）
    3. 用 LLM Judge 对比模型答案与 gold answer
    4. 返回 F1/EM 分数
    """

    def __init__(self, config):
        self.config = config
        self.benchmark_id = config.id
        self.eval_config = config.eval_config or {}

    def run_eval(self, model_path: str, workspace_path: str, **kwargs) -> Dict[str, Any]:
        from vllm import LLM, SamplingParams

        result = self.get_default_result(self.benchmark_id, model_path)
        result["eval_type"] = "deepsearchqa"

        if not self.validate_model(model_path):
            result["error"] = f"Model not found: {model_path}"
            return result

        # Deterministic held-out evaluation split: 100 train / 800 eval.
        num_samples = self.eval_config.get("num_samples", DEFAULT_EVAL_SIZE)
        dataset = load_source_dataset()
        _, eval_dataset = split_dataset(dataset)
        samples = list(eval_dataset.select(range(min(num_samples, len(eval_dataset)))))
        logger.info(
            f"DeepSearchQA held-out eval: {len(samples)} samples "
            f"(train={TRAIN_SIZE}, eval={len(eval_dataset)}, source={DATASET_NAME}/{SOURCE_SPLIT})"
        )

        # load model (vLLM)
        logger.info(f"Loading model: {model_path}")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=4096,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop=["\nAction:", "\nThought:", "\nObservation:"],
        )

        # search tool
        search_fn = self._get_search_function()

        # evaluation loop
        generated_records = []

        for i, sample in enumerate(samples):
            question = sample["problem"]
            gold_answer = sample["answer"]
            answer_type = sample.get("answer_type", "Single Answer")

            logger.info(f"\n[{i+1}/{len(samples)}] {question[:80]}...")

            # ReAct loop
            predicted = self._react_loop(
                llm,
                sampling_params,
                search_fn,
                question,
                answer_type,
            )

            generated_records.append(
                {
                    "idx": i,
                    "question": question[:100],
                    "gold": gold_answer,
                    "predicted": predicted,
                    "answer_type": answer_type,
                }
            )
            logger.info(f"  Predicted: {predicted[:80]}")
            logger.info(f"  Gold:      {gold_answer[:80]}")

        judge_workers = int(self.eval_config.get("judge_workers", 8))
        logger.info(f"Running parallel answer judging with {judge_workers} workers")

        results_detail = [None] * len(generated_records)
        correct = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=max(1, judge_workers)) as executor:
            future_to_record = {
                executor.submit(
                    self._judge_answer,
                    record["predicted"],
                    record["gold"],
                    record["answer_type"],
                ): record
                for record in generated_records
            }

            for future in as_completed(future_to_record):
                record = future_to_record[future]
                score = future.result()
                if score:
                    correct += 1
                completed += 1

                results_detail[record["idx"]] = {
                    "question": record["question"],
                    "gold": record["gold"],
                    "predicted": record["predicted"],
                    "answer_type": record["answer_type"],
                    "correct": score,
                }
                logger.info(
                    f"  Judge {completed}/{len(generated_records)} | "
                    f"Correct={score} | Running accuracy: {correct}/{completed} = {correct / completed:.2%}"
                )

        accuracy = correct / len(samples) if samples else 0.0
        result["score"] = accuracy * 100
        result["accuracy_summary"] = {
            "correct": correct,
            "total": len(samples),
            "accuracy": accuracy,
            "details": results_detail,
        }
        logger.info(f"\nDeepSearchQA done: {correct}/{len(samples)} = {accuracy:.2%}")
        return result

    # ----------------------------------------------------------
    # ReAct loop
    # ----------------------------------------------------------

    def _react_loop(
        self,
        llm: "LLM",
        sampling_params: "SamplingParams",
        search_fn,
        question: str,
        answer_type: str,
    ) -> str:
        """ReAct multi-step reasoning loop, return final answer string"""
        from vllm import SamplingParams

        max_steps = self.eval_config.get("max_steps", 6)

        conversation = f"Question: {question}\n" f"Answer type: {answer_type}\n\n" "Thought:"
        full_prompt = f"{REACT_SYSTEM_PROMPT}\n\n{conversation}"

        # for step in range(max_steps):
        #     outputs = llm.generate([full_prompt], sampling_params)
        #     generated = outputs[0].outputs[0].text.strip()
        #     full_prompt += f" {generated}"

        #     # parse Action
        #     action_match = re.search(r"Action:\s*(search|answer)\[(.+?)\]", full_prompt, re.DOTALL)
        #     if not action_match:
        #         # force append an answer
        #         full_prompt += "\nAction: answer["
        #         outputs2 = llm.generate([full_prompt], SamplingParams(temperature=0, max_tokens=128, stop=["]"]))
        #         return outputs2[0].outputs[0].text.strip()

        #     action_type = action_match.group(1)
        #     action_content = action_match.group(2).strip()

        #     if action_type == "answer":
        #         return action_content

        #     # execute search
        #     observation = search_fn(action_content)
        #     logger.info(f"  Step {step+1} | Search: {action_content[:60]}")
        #     logger.info(f"  Observation: {observation[:120]}")

        #     full_prompt += (
        #         f"\nObservation: {observation}\n"
        #         "Thought:"
        #     )
        # exceed max steps, extract last answer
        # last_answer = re.findall(r"Action:\s*answer\[(.+?)\]", full_prompt, re.DOTALL)
        # return last_answer[-1].strip() if last_answer else "I don't know"

        # ...existing code...
        model_trace = ""

        for step in range(max_steps):
            outputs = llm.generate([full_prompt], sampling_params)
            generated = outputs[0].outputs[0].text.strip()
            model_trace += ("\n" + generated) if model_trace else generated
            full_prompt += f" {generated}"

            # parse Action ONLY from current model output
            action_match = re.search(r"Action:\s*(search|answer)\[(.+?)\]", generated, re.DOTALL)
            if not action_match:
                # force append an answer
                full_prompt += "\nAction: answer["
                outputs2 = llm.generate([full_prompt], SamplingParams(temperature=0, max_tokens=128, stop=["]"]))
                generated2 = outputs2[0].outputs[0].text.strip()
                model_trace += ("\n" + generated2) if generated2 else ""
                # reject template placeholder
                if generated2.strip().lower() == "your final answer":
                    continue
                return generated2.strip()

            action_type = action_match.group(1)
            action_content = action_match.group(2).strip()

            if action_type == "answer":
                # reject template placeholder
                if action_content.lower() == "your final answer":
                    # treat as no valid action, let loop continue
                    full_prompt += "\nThat is not a valid answer. Please think again.\nThought:"
                    continue
                return action_content

            # execute search
            observation = search_fn(action_content)
            logger.info(f"  Step {step+1} | Search: {action_content[:60]}")
            logger.info(f"  Observation: {observation[:120]}")

            full_prompt += f"\nObservation: {observation}\n" "Thought:"

        # exceed max steps, extract last answer from model output only
        last_answer = re.findall(r"Action:\s*answer\[(.+?)\]", model_trace, re.DOTALL)
        # filter out template placeholder
        real_answers = [a.strip() for a in last_answer if a.strip().lower() != "your final answer"]
        return real_answers[-1] if real_answers else "I don't know"

    # ----------------------------------------------------------
    # search tool
    # ----------------------------------------------------------

    def _get_search_function(self):
        """返回搜索函数，优先使用 SerpAPI，降级到 DuckDuckGo"""
        import os

        serpapi_key = os.environ.get("SERPAPI_KEY") or self.eval_config.get("serpapi_key")

        if serpapi_key:
            logger.info("Using SerpAPI for web search")
            return lambda q: self._serpapi_search(q, serpapi_key)
        else:
            logger.info("Using DuckDuckGo for web search (no SERPAPI_KEY)")
            return self._duckduckgo_search

    def _serpapi_search(self, query: str, api_key: str) -> str:
        """SerpAPI 搜索，返回摘要文本"""
        try:
            resp = requests.get(
                "https://serpapi.com/search",
                params={"q": query, "api_key": api_key, "num": 3},
                timeout=10,
            )
            data = resp.json()
            snippets = [r.get("snippet", "") for r in data.get("organic_results", [])[:3]]
            return " | ".join(snippets) or "No results found."
        except Exception as e:
            return f"Search error: {e}"

    def _duckduckgo_search(self, query: str) -> str:
        """DuckDuckGo 即时答案 API（免费，但结果较少）"""
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
                timeout=10,
            )
            data = resp.json()
            abstract = data.get("AbstractText", "")
            related = " | ".join(r.get("Text", "") for r in data.get("RelatedTopics", [])[:2] if isinstance(r, dict))
            return abstract or related or "No results found."
        except Exception as e:
            return f"Search error: {e}"

    # ----------------------------------------------------------
    # LLM Judge
    # ----------------------------------------------------------

    def _judge_answer(
        self,
        predicted: str,
        gold: str,
        answer_type: str,
    ) -> bool:
        from rdagent.oai.llm_utils import APIBackend

        judge_prompt = f"""You are an answer evaluator. Compare the predicted answer to the gold answer.
        Question answer type: {answer_type}
        Gold answer: {gold}
        Predicted answer: {predicted}
        For "Single Answer": The predicted answer is correct if it contains the same key information as the gold answer.
        For "Set Answer": The predicted answer is correct if it contains ALL items from the gold answer (order doesn't matter, minor wording variations are OK).
        Reply with ONLY "correct" or "incorrect". No explanation."""

        try:
            response = (
                APIBackend()
                .build_messages_and_create_chat_completion(
                    user_prompt=judge_prompt,
                    system_prompt="You are a strict answer evaluator.",
                )
                .strip()
                .lower()
            )
            normalized = response.splitlines()[0].strip().strip(".!,;: \t\r\n").lower()
            return normalized == "correct"
        except Exception as e:
            logger.warning(f"Judge failed: {e}, falling back to string match")
            return self._string_match(predicted, gold, answer_type)

    def _string_match(self, predicted: str, gold: str, answer_type: str) -> bool:
        """fallback: simple string matching"""
        pred = predicted.strip().lower()
        gold = gold.strip().lower()
        if answer_type == "Single Answer":
            return gold in pred or pred in gold
        else:
            gold_items = [x.strip() for x in gold.split(",")]
            return all(item in pred for item in gold_items)

"""
WebShop Evaluator - E-commerce website interactive environment

Evaluating LLM in WebShop environment using ReAct agent.
Two backends are supported:
- vllm: local model inference
- api: OpenAI compatible API

WebShop official code: https://github.com/princeton-nlp/webshop
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator

from .data import WEBSHOP_REPO_DIR, _clone_webshop_repo, _ensure_repo_in_path

# Log directory
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "log"


class _Tee:
"""Output to both terminal and log file"""

    def __init__(self, filepath):
        self.terminal = sys.__stdout__
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.terminal.fileno()


def _log(msg: str):
"""Simple print log (will be written to the file by Tee at the same time)"""
    print(msg, flush=True)


# ============================================================
# LLM backend factory
# ============================================================


def create_llm_fn(backend: str, model_path: str, **kwargs) -> Tuple[Callable, Callable]:
    """
Create a unified llm(prompt, stop) function.

backend="vllm": local model, text completion
backend="api": OpenAI compatible chat API

    Returns:
(llm_fn, cleanup_fn): cleanup_fn releases resources
    """
    if backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.distributed.parallel_state import destroy_model_parallel

        llm_engine = LLM(
            model=model_path, tensor_parallel_size=kwargs.get("tensor_parallel_size", 1), trust_remote_code=True
        )

        _vllm_sys_msg = (
            "You are a helpful shopping assistant browsing an e-commerce website. "
            "Given a user instruction, current observation, and available actions, "
            "pick the best action to find and purchase a matching product. "
            "Output ONLY one action (e.g., 'search[red shoes]', 'click[buy now]') "
            "with NO extra text, NO explanation."
        )

        def vllm_fn(prompt: str, stop: List[str] = None) -> str:
            messages = [
                {"role": "system", "content": _vllm_sys_msg},
                {"role": "user", "content": prompt},
            ]
            params = SamplingParams(temperature=0, max_tokens=100, stop=stop or ["\n"])
            outputs = llm_engine.chat(messages, sampling_params=params)
            return outputs[0].outputs[0].text

        def cleanup():
            nonlocal llm_engine
            import gc

            import torch

            destroy_model_parallel()
            llm_engine = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log("vLLM engine released, GPU memory freed.")

        return vllm_fn, cleanup

    elif backend == "api":
        from openai import OpenAI

        client = OpenAI(
            api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=kwargs.get("api_base") or os.getenv("OPENAI_API_BASE"),
        )
        model_name = model_path

        system_msg = (
            "You are a helpful shopping assistant browsing an e-commerce website. "
            "Given a user instruction, current observation, and available actions, "
            "pick the best action to find and purchase a matching product. "
            "Output ONLY one action (e.g., 'search[red shoes]', 'click[buy now]') "
            "with NO extra text, NO explanation."
        )

        def api_fn(prompt: str, stop: List[str] = None) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=100,
                stop=stop or ["\n"],
            )
            text = response.choices[0].message.content or ""
            return text.strip()

        return api_fn, lambda: None

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'vllm' or 'api'.")


# ============================================================
# ReAct Agent core logic
# ============================================================


def _format_available_actions(avail: dict) -> str:
"""Format the available_actions returned by the environment into a text list"""
    lines = []
    if avail.get("has_search_bar"):
        lines.append("search[<your query>]")
    for txt in avail.get("clickables", []):
        lines.append(f"click[{txt}]")
    return "\n".join(f"  {a}" for a in lines)


def build_react_prompt(
    instruction: str,
    history: List[Tuple[str, str]],
    observation: str,
    available_actions: str = "",
    history_window: int = 5,
) -> str:
"""Build ReAct style prompts with available_actions and limited history window"""
    prompt = f"""You are shopping on an e-commerce website. Find and purchase a product matching the user's instruction.

Instruction: {instruction}

Available actions:
{available_actions}

Rules:
- Output ONLY one action from the available actions list above.
- For search, use: search[your query]
- For clicking, use: click[exact text from the list]
- Do NOT output anything other than the action.

Now it's your turn:
"""

    recent = history[-history_window:] if len(history) > history_window else history
    offset = len(history) - len(recent)

    for i, (action, obs) in enumerate(recent):
        step_num = offset + i + 1
        prompt += f"\nObservation {step_num}: {obs}\n"
        prompt += f"Action {step_num}: {action}\n"

    prompt += f"\nObservation {len(history)+1}: {observation}\n"
    prompt += f"Action {len(history)+1}:"

    return prompt


def webshop_run(
    llm_fn: Callable,
    env,
    instruction: str,
    observation: str,
    max_steps: int = 50,
    history_window: int = 5,
) -> Tuple[float, int, bool]:
    """
Single-round WebShop evaluation logic.

    Args:
        llm_fn: llm(prompt, stop) -> str
env: WebShop environment instance
instruction: user instruction
observation: initial observation
max_steps: maximum number of steps
history_window: the number of recent historical steps retained in prompt

    Returns:
(reward, steps, success): reward is the final reward, steps is the actual number of steps, success is whether it is successful or not
    """
    history = []

    for step in range(1, max_steps + 1):
        avail = env.get_available_actions()
        avail_text = _format_available_actions(avail)

        prompt = build_react_prompt(
            instruction,
            history,
            observation,
            available_actions=avail_text,
            history_window=history_window,
        )

        action = llm_fn(prompt, stop=["\n"]).strip()

# Clean action prefix
        if action.startswith("Action:"):
            action = action[7:].strip()
        if action.startswith("choose["):
            action = "click[" + action[7:]

        _log(f"  Step {step}: {action}")

        observation, reward, done, info = env.step(action)

        _log(f"  Obs {step}: {observation[:200]}...")
        _log(f"  Reward: {reward}, Done: {done}")

        history.append((action, observation))

        if done:
            success = reward >= 0.5
            return reward, step, success

    return 0.0, max_steps, False


# ============================================================
# Evaluator
# ============================================================


class WebShopEvaluator(BaseEvaluator):
    """
WebShop Evaluator (ReAct agent)

eval_config field:
max_steps: Maximum number of steps per task (default 50)
num_instructions: Number of evaluation instructions (default 100)
backend: "vllm" or "api" (automatically determined by default)
api_key: API key (when backend=api)
api_base: API address (when backend=api)
num_products: Number of products loaded (default 1000, optional 1000 or all)
    """

    def __init__(self, config):
        self.config = config
        self.benchmark_id = config.id
        self.eval_config = config.eval_config or {}

    def run_eval(
        self,
        model_path: str,
        workspace_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
"""Run WebShop Review"""
        result = self.get_default_result(self.benchmark_id, model_path)
        result["eval_type"] = "webshop"

# Merge kwargs into eval_config
        cfg = {**self.eval_config, **kwargs}
        max_steps = cfg.get("max_steps", 50)
        num_instructions = cfg.get("num_instructions", 100)
        num_products = cfg.get("num_products", 1000)

# --- Set up log Tee ---
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        model_safe = model_path.replace("/", "_").replace("\\", "_")
        log_file = LOG_DIR / f"webshop_{model_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        old_stdout = sys.stdout
        sys.stdout = _Tee(log_file)

        try:
            _log(f"Log: {log_file}")

# --- Make sure the WebShop repository is available ---
            _clone_webshop_repo()
            _ensure_repo_in_path()

# --- Judgment backend ---
            backend = cfg.get("backend")
            if backend is None:
                backend = "api" if not Path(model_path).exists() else "vllm"
            _log(f"WebShop eval: backend={backend}, model={model_path}")

# --- Create LLM function ---
            llm_fn, llm_cleanup = create_llm_fn(
                backend=backend,
                model_path=model_path,
                api_key=cfg.get("api_key"),
                api_base=cfg.get("api_base"),
                tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            )

# --- Initialize WebShop environment ---
            try:
                from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
            except ImportError as e:
                result["error"] = f"Failed to import WebShop: {e}. Please check WebShop installation."
                sys.stdout = old_stdout
                return result

            env = WebAgentTextEnv(
                observation_mode="text",
                num_products=num_products,
            )

# --- Load evaluation instructions ---
            instruction_idxs = list(range(min(num_instructions, 12000)))

            _log(f"WebShop: {len(instruction_idxs)} instructions, max {max_steps} steps each")

# --- Evaluation loop ---
            total_reward = 0.0
            success_count = 0
            total_steps = 0

            for idx, instr_idx in enumerate(instruction_idxs):
                try:
# Reset environment
                    observation, _ = env.reset(session=instr_idx)
                    instruction = env.get_instruction_text()

                    _log(f"\n[Task {idx + 1}/{len(instruction_idxs)}] {instruction[:80]}...")

# run agent
                    reward, steps, success = webshop_run(
                        llm_fn=llm_fn,
                        env=env,
                        instruction=instruction,
                        observation=observation,
                        max_steps=max_steps,
                    )

                    total_reward += reward
                    total_steps += steps
                    if success:
                        success_count += 1

                    _log(f"  Result: {'SUCCESS' if success else 'FAIL'} (reward={reward:.2f}, steps={steps})")

# Print progress
                    current_success_rate = success_count / (idx + 1)
                    _log(f"  Running: {success_count}/{idx + 1} = {current_success_rate:.1%}")

                except Exception as e:
                    _log(f"  ERROR: {e}")
                    import traceback

                    _log(traceback.format_exc())
                    continue

# --- Summary results ---
            total_count = len(instruction_idxs)
            success_rate = success_count / total_count if total_count > 0 else 0.0
            avg_reward = total_reward / total_count if total_count > 0 else 0.0
            avg_steps = total_steps / total_count if total_count > 0 else 0.0

result["score"] = success_rate * 100 # Convert to percentage
            result["accuracy_summary"] = {
                "success_count": success_count,
                "total_count": total_count,
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "avg_steps": avg_steps,
                "total_reward": total_reward,
            }

            _log(f"\nWebShop done: {success_count}/{total_count} = {success_rate:.2%}")
            _log(f"  Average reward: {avg_reward:.3f}")
            _log(f"  Average steps: {avg_steps:.1f}")

        except Exception as e:
            result["error"] = str(e)
            _log(f"ERROR: {e}")
            import traceback

            _log(traceback.format_exc())

        finally:
# --- Cleanup ---
            if "env" in locals():
                env.close()

# Release LLM resources
            if "llm_cleanup" in locals():
                llm_cleanup()

#Restore stdout
            sys.stdout = old_stdout

        return result

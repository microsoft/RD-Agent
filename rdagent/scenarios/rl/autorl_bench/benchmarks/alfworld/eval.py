"""
ALFWorld Evaluator - Interactive text gaming environment

Evaluating LLM in ALFWorld using the ReAct agent (few-shot + full history).
Two backends are supported:
- vllm: local model reasoning (text completion, consistent with the original version of ReAct)
- api: OpenAI compatible API (chat completion)

ReAct official code: https://github.com/ysymyth/ReAct/blob/main/alfworld.ipynb
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator

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
# ReAct agent core logic (from official alfworld.ipynb)
# ============================================================

# Task type → mapping of few-shot prompt key
TASK_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def process_ob(ob: str) -> str:
"""Official ReAct observation cleaning"""
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def alfworld_run(llm_fn: Callable, env, prompt: str, ob: str, max_steps: int = 50) -> tuple:
    """
ReAct’s official single-game evaluation logic.

    Args:
        llm_fn: llm(prompt, stop) -> str
env: ALFWorld environment instance
prompt: few-shot prompt (2 examples included)
ob: initial observation
max_steps: maximum number of steps

    Returns:
(reward, steps): reward=1 indicates success, steps is the actual number of steps
    """
    init_prompt = prompt + ob + "\n>"
    history = ""
    for i in range(1, max_steps + 1):
        action = llm_fn(init_prompt + history, stop=["\n"]).strip()
        observation, reward, done, info = env.step([action])
        observation = process_ob(observation[0])
        reward = info["won"][0]
        done = done[0]
        if action.startswith("think:"):
            observation = "OK."
        _log(f"  Act {i}: {action}")
        _log(f"  Obs {i}: {observation}")
        history += f" {action}\n{observation}\n>"
        if done:
            return reward, i
    return 0, max_steps


# ============================================================
# LLM backend factory
# ============================================================


def create_llm_fn(backend: str, model_path: str, **kwargs) -> tuple:
    """
Create a unified llm(prompt, stop) function.

backend="vllm": local model, text completion (consistent with ReAct original behavior)
backend="api": OpenAI compatible chat API

    Returns:
(llm_fn, cleanup_fn): cleanup_fn releases GPU memory
    """
    if backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.distributed.parallel_state import destroy_model_parallel

        llm_engine = LLM(
            model=model_path, tensor_parallel_size=kwargs.get("tensor_parallel_size", 1), trust_remote_code=True
        )

        def vllm_fn(prompt: str, stop: List[str] = None) -> str:
            params = SamplingParams(temperature=0, max_tokens=100, stop=stop or ["\n"])
            outputs = llm_engine.generate([prompt], params)
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
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY")),
            base_url=kwargs.get("api_base", os.getenv("OPENAI_API_BASE")),
        )
        model_name = model_path

        system_msg = (
            "You are playing a text-based household game. "
            "You will be given a task and interaction history. "
            "Output ONLY the next action (e.g. 'go to desk 1', 'take mug 1 from desk 1', "
            "'use desklamp 1', 'think: I need to find...') with NO extra text, "
            "NO prefix like '>' or 'Action:', just the raw action string."
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
            text = text.strip()
            if text.startswith("> "):
                text = text[2:]
            return text

        return api_fn, lambda: None

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'vllm' or 'api'.")


# ============================================================
# Evaluator
# ============================================================


class ALFWorldEvaluator(BaseEvaluator):
    """
ALFWorld evaluator (ReAct agent)

eval_config field:
max_steps: Maximum number of steps per round (default 50)
env_num: Number of evaluation games (default 134)
react_prompts: ReAct few-shot prompts file path
backend: "vllm" or "api" (automatically determined by default)
api_key: API key (when backend=api)
api_base: API address (when backend=api)
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
"""Run the ALFWorld review"""
        result = self.get_default_result(self.benchmark_id, model_path)
        result["eval_type"] = "alfworld"

# Merge kwargs into eval_config
        cfg = {**self.eval_config, **kwargs}
        max_steps = cfg.get("max_steps", 50)
        env_num = cfg.get("env_num", 134)

# --- Set up log Tee ---
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        model_safe = model_path.replace("/", "_")
        log_file = LOG_DIR / f"alfworld_{model_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        sys.stdout = _Tee(log_file)

# --- Judgment backend ---
        backend = cfg.get("backend")
        if backend is None:
            backend = "api" if not Path(model_path).exists() else "vllm"
        _log(f"Log: {log_file}")
        _log(f"ALFWorld eval: backend={backend}, model={model_path}")

# --- Create LLM function ---
        llm_fn, llm_cleanup = create_llm_fn(
            backend=backend,
            model_path=model_path,
            api_key=cfg.get("api_key"),
            api_base=cfg.get("api_base"),
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        )

# --- Load ReAct few-shot prompts ---
        prompts_path = cfg.get("react_prompts")
        if prompts_path is None:
#Default path: react_prompts.json in the same directory as eval.py
            prompts_path = Path(__file__).parent / "react_prompts.json"
        with open(prompts_path) as f:
            react_prompts = json.load(f)

# --- Make sure ALFWorld game data is downloaded ---
        self._ensure_alfworld_data()

# --- Initialize ALFWorld environment ---
        workspace = Path(workspace_path)

        from rdagent.scenarios.rl.autorl_bench.benchmarks.alfworld.data import (
            _ensure_alfworld_data,
        )

        alfworld_data = str(_ensure_alfworld_data())
        os.environ["ALFWORLD_DATA"] = alfworld_data

# env_config: Read the official base_config.yaml in the same directory and expand $ALFWORLD_DATA
        config_yaml = Path(__file__).parent / "base_config.yaml"
        with open(config_yaml) as f:
            import yaml

            env_config = yaml.safe_load(f)
        env_config = self._expand_env_vars(env_config)

        from alfworld.agents.environment import get_environment

        split = cfg.get("split", "eval_out_of_distribution")
        env_type = env_config.get("env", {}).get("type", "AlfredTWEnv")
        alfred_env = get_environment(env_type)(env_config, train_eval=split)
        env = alfred_env.init_env(batch_size=1)

        num_games = min(env_num, alfred_env.num_games)
        _log(f"ALFWorld: {num_games} games, max {max_steps} steps, split={split}")

# --- Evaluation loop (ReAct official logic) ---
        cnts = [0] * 6
        rs = [0] * 6

        for game_no in range(num_games):
            ob, info = env.reset()
            ob = "\n".join(ob[0].split("\n\n")[1:])
            name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
            _log(f"\n[Game {game_no + 1}/{num_games}] {name}")

            matched = False
            for i, (prefix, prompt_key) in enumerate(TASK_PREFIXES.items()):
                if name.startswith(prefix):
                    prompt = (
                        "Interact with a household to solve a task. Here are two examples.\n"
                        + react_prompts[f"react_{prompt_key}_1"]
                        + react_prompts[f"react_{prompt_key}_0"]
                        + "\nHere is the task.\n"
                    )
                    reward, steps = alfworld_run(llm_fn, env, prompt, ob, max_steps)
                    rs[i] += reward
                    cnts[i] += 1
                    matched = True
                    _log(f"  Result: {'WON' if reward else 'LOST'} ({steps} steps)")
                    break

            if not matched:
                _log(f"  WARNING: Unknown task type: {name}, skipping")
                continue

            total_r, total_c = sum(rs), sum(cnts)
            _log(f"  Running: {total_r}/{total_c} = {total_r / max(total_c, 1):.1%}")

        env.close()
        llm_cleanup()

# --- Summary results ---
        total_success = sum(rs)
        total_count = sum(cnts)
        success_rate = total_success / total_count if total_count > 0 else 0.0

        per_task = {}
        for (prefix, _), s, c in zip(TASK_PREFIXES.items(), rs, cnts):
            if c > 0:
                per_task[prefix] = {"success": s, "total": c, "rate": s / c}

        result["score"] = success_rate * 100
        result["accuracy_summary"] = {
            "success_count": total_success,
            "total_count": total_count,
            "success_rate": success_rate,
            "per_task": per_task,
        }

        _log(f"\nALFWorld done: {total_success}/{total_count} = {success_rate:.2%}")
        for prefix, stats in per_task.items():
            _log(f"  {prefix:30s} {stats['success']}/{stats['total']} = {stats['rate']:.0%}")

#Restore stdout
        sys.stdout = sys.__stdout__

        return result

    @staticmethod
    def _ensure_alfworld_data():
"""Check ALFWorld game data (~2.1GB), if not, download it automatically"""
        import subprocess

        cache_dir = Path.home() / ".cache" / "alfworld"
        if (cache_dir / "json_2.1.1").exists():
            return
        _log("Downloading ALFWorld game data (~2.1GB, first time only)...")
        subprocess.run(["alfworld-download"], check=True)
        _log(f"ALFWorld data downloaded to {cache_dir}")

    def _expand_env_vars(self, obj):
"""Recursively expand $ENV_VAR"""
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        elif isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(x) for x in obj]
        return obj

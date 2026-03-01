"""
ALFWorld Evaluator - 交互式文本游戏环境

使用 ReAct agent（few-shot + 完整历史）在 ALFWorld 中评测 LLM。
支持两种后端：
  - vllm: 本地模型推理（text completion，和 ReAct 原版一致）
  - api:  OpenAI 兼容 API（chat completion）

ReAct 官方代码: https://github.com/ysymyth/ReAct/blob/main/alfworld.ipynb
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List
from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator

# 日志目录
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "log"


class _Tee:
    """同时输出到终端和日志文件"""
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
    """简单的 print 日志（会被 Tee 同时写入文件）"""
    print(msg, flush=True)


# ============================================================
# ReAct agent 核心逻辑（来自官方 alfworld.ipynb）
# ============================================================

# 任务类型 → few-shot prompt key 的映射
TASK_PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def process_ob(ob: str) -> str:
    """官方 ReAct 的 observation 清洗"""
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def alfworld_run(llm_fn: Callable, env, prompt: str, ob: str, max_steps: int = 50) -> tuple:
    """
    ReAct 官方的单局评测逻辑。

    Args:
        llm_fn: llm(prompt, stop) -> str
        env: ALFWorld 环境实例
        prompt: few-shot prompt（含 2 个示例）
        ob: 初始 observation
        max_steps: 最大步数

    Returns:
        (reward, steps): reward=1 表示成功，steps 为实际步数
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
# LLM 后端工厂
# ============================================================

def create_llm_fn(backend: str, model_path: str, **kwargs) -> Callable:
    """
    创建统一的 llm(prompt, stop) 函数。

    backend="vllm": 本地模型，text completion（和 ReAct 原版行为一致）
    backend="api":  OpenAI 兼容 chat API
    """
    if backend == "vllm":
        from vllm import LLM, SamplingParams

        llm_engine = LLM(model=model_path, tensor_parallel_size=kwargs.get("tensor_parallel_size", 1), trust_remote_code=True)

        def vllm_fn(prompt: str, stop: List[str] = None) -> str:
            params = SamplingParams(temperature=0, max_tokens=100, stop=stop or ["\n"])
            outputs = llm_engine.generate([prompt], params)
            return outputs[0].outputs[0].text

        return vllm_fn

    elif backend == "api":
        from openai import OpenAI

        client = OpenAI(
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY")),
            base_url=kwargs.get("api_base", os.getenv("OPENAI_API_BASE")),
        )
        model_name = model_path  # API 模式下 model_path 就是模型名

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

        return api_fn

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'vllm' or 'api'.")


# ============================================================
# Evaluator
# ============================================================

class ALFWorldEvaluator(BaseEvaluator):
    """
    ALFWorld 评测器（ReAct agent）

    eval_config 字段：
        max_steps:    每局最大步数（默认 50）
        env_num:      评测局数（默认 134）
        react_prompts: ReAct few-shot prompts 文件路径
        backend:      "vllm" 或 "api"（默认自动判断）
        api_key:      API 密钥（backend=api 时）
        api_base:     API 地址（backend=api 时）
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
        """运行 ALFWorld 评测"""
        result = self.get_default_result(self.benchmark_id, model_path)
        result["eval_type"] = "alfworld"

        # 合并 kwargs 到 eval_config
        cfg = {**self.eval_config, **kwargs}
        max_steps = cfg.get("max_steps", 50)
        env_num = cfg.get("env_num", 134)

        # --- 设置日志 Tee ---
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        model_safe = model_path.replace("/", "_")
        log_file = LOG_DIR / f"alfworld_{model_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        sys.stdout = _Tee(log_file)

        # --- 判断 backend ---
        backend = cfg.get("backend")
        if backend is None:
            backend = "api" if not Path(model_path).exists() else "vllm"
        _log(f"Log: {log_file}")
        _log(f"ALFWorld eval: backend={backend}, model={model_path}")

        # --- 创建 LLM 函数 ---
        llm_fn = create_llm_fn(
            backend=backend,
            model_path=model_path,
            api_key=cfg.get("api_key"),
            api_base=cfg.get("api_base"),
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        )

        # --- 加载 ReAct few-shot prompts ---
        prompts_path = cfg.get("react_prompts")
        if prompts_path is None:
            # 默认路径：和 eval.py 同目录下的 react_prompts.json
            prompts_path = Path(__file__).parent / "react_prompts.json"
        with open(prompts_path) as f:
            react_prompts = json.load(f)

        # --- 确保 ALFWorld 游戏数据已下载 ---
        self._ensure_alfworld_data()

        # --- 初始化 ALFWorld 环境 ---
        workspace = Path(workspace_path)

        from rdagent.scenarios.rl.autorl_bench.benchmarks.alfworld.data import _ensure_alfworld_data
        alfworld_data = str(_ensure_alfworld_data())
        os.environ["ALFWORLD_DATA"] = alfworld_data

        # env_config: 读同目录下官方 base_config.yaml，展开 $ALFWORLD_DATA
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

        # --- 评测循环（ReAct 官方逻辑） ---
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

        # --- 汇总结果 ---
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

        # 恢复 stdout
        sys.stdout = sys.__stdout__

        return result

    @staticmethod
    def _ensure_alfworld_data():
        """检查 ALFWorld 游戏数据（~2.1GB），没有就自动下载"""
        import subprocess
        cache_dir = Path.home() / ".cache" / "alfworld"
        if (cache_dir / "json_2.1.1").exists():
            return
        _log("Downloading ALFWorld game data (~2.1GB, first time only)...")
        subprocess.run(["alfworld-download"], check=True)
        _log(f"ALFWorld data downloaded to {cache_dir}")

    def _expand_env_vars(self, obj):
        """递归展开 $ENV_VAR"""
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        elif isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(x) for x in obj]
        return obj

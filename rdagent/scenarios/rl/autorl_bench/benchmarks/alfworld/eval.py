"""
ALFWorld Evaluator - 交互式文本游戏环境

使用 vLLM 加载本地模型，在 ALFWorld 环境中进行评测。
"""
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from alfworld.agents.environment import get_environment
from vllm import LLM, SamplingParams

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.core.evaluator import BaseEvaluator


class ALFWorldEvaluator(BaseEvaluator):
    """
    ALFWorld 评测器
    
    在文本交互环境中评测模型的任务完成能力。
    """
    
    def __init__(self, config):
        self.config = config
        self.benchmark_id = config.id
        self.eval_config = config.eval_config or {}
    
    def run_eval(
        self,
        model_path: str,
        workspace_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """运行 ALFWorld 评测"""
        result = self.get_default_result(self.benchmark_id, model_path)
        result["eval_type"] = "alfworld"
        
        if not self.validate_model(model_path):
            result["error"] = f"Model not found: {model_path}"
            return result
        
        workspace = Path(workspace_path)
        
        # 设置环境
        alfworld_data = workspace / "data" / "alfworld" / "data"
        os.environ["ALFWORLD_DATA"] = str(alfworld_data)
        
        config_path = workspace / "data" / "configs" / "eval_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config = self._expand_env_vars(config)
        
        # 加载模型
        logger.info(f"Loading model: {model_path}")
        llm = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0, max_tokens=64, stop=["\n"])
        
        def get_action(obs: str, admissible: List[str]) -> str:
            prompt = self._build_prompt(obs, admissible)
            outputs = llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
        
        # 运行评测
        return self._run_eval_loop(config, get_action, result)
    
    def _run_eval_loop(self, config: dict, get_action, result: dict) -> dict:
        """评测循环"""
        env_type = config.get("env", {}).get("type", "AlfredTWEnv")
        alfred_env = get_environment(env_type)(config, train_eval="eval_in_distribution")
        env = alfred_env.init_env(batch_size=1)
        
        max_steps = self.eval_config.get("max_steps", 50)
        env_num = self.eval_config.get("env_num", 140)
        num_games = min(env_num, alfred_env.num_games)
        
        logger.info(f"ALFWorld eval: {num_games} games, max {max_steps} steps")
        
        success_count = 0
        for game_idx in range(num_games):
            obs, info = env.reset()
            obs = obs[0]
            logger.info(f"\n=== Game {game_idx + 1}/{num_games} ===")
            logger.info(f"Task: {obs[:200]}...")
            
            for step in range(max_steps):
                admissible = info.get("admissible_commands", [[]])[0]
                
                action = get_action(obs, admissible)
                action = self._match_action(action, admissible)
                logger.info(f"Step {step + 1}: {action}")
                
                obs, reward, done, info = env.step([action])
                obs = obs[0]
                
                if done[0]:
                    won = info.get("won", [False])[0]
                    if won:
                        success_count += 1
                    logger.info(f"Done! Won: {won}")
                    break
        
        env.close()
        
        success_rate = success_count / num_games if num_games > 0 else 0.0
        result["score"] = success_rate * 100
        result["accuracy_summary"] = {
            "success_count": success_count,
            "total_count": num_games,
            "success_rate": success_rate,
        }
        
        logger.info(f"\nALFWorld done: {success_count}/{num_games} = {success_rate:.2%}")
        return result
    
    def _expand_env_vars(self, obj):
        """递归展开环境变量"""
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        elif isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(x) for x in obj]
        return obj
    
    def _build_prompt(self, obs: str, admissible: List[str]) -> str:
        """构造 prompt"""
        actions_str = "\n".join(f"- {a}" for a in admissible[:20])
        return f"""You are in a text-based home environment.

Observation:
{obs}

Available actions:
{actions_str}

Choose ONE action from the list. Output ONLY the action, nothing else.

Action:"""
    
    def _match_action(self, action: str, admissible: List[str]) -> str:
        """匹配合法动作"""
        action = action.strip().lower()
        
        for a in admissible:
            if a.lower() == action:
                return a
        
        for a in admissible:
            if a.lower().startswith(action) or action.startswith(a.lower()):
                return a
        
        return admissible[0] if admissible else "look"

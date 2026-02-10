"""
ALFWorld GPT 评测（直接复制 ReAct 官方代码，仅改 LLM API 调用）

官方代码: https://github.com/ysymyth/ReAct/blob/main/alfworld.ipynb
改动: openai.Completion.create → openai.ChatCompletion (chat API)

用法:
    conda activate cwy-rl
    cd /Data/home/v-wanyichen/cwy/program/RD-Agent
    python rdagent/scenarios/rl/autorl_bench/test/test_alfworld_gpt.py \
        --model gpt-5.2 --num-games 2
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 文件加载环境变量
load_dotenv()

# ============================================================
# 路径配置
# ============================================================
LOG_DIR = Path(__file__).resolve().parent.parent / "log"
ALFWORLD_DATA = "/Data/home/v-wanyichen/cwy/program/benchmark/alfworld/alfworld/data"
EVAL_CONFIG = Path(__file__).resolve().parent.parent / "workspace/alfworld/data/configs/eval_config.yaml"
REACT_PROMPTS = "/Data/home/v-wanyichen/cwy/program/benchmark/react/prompts/alfworld_3prompts.json"


# ============================================================
# 和 ReAct 官方一模一样的代码（只改了 llm 函数）
# ============================================================

# --- 官方 Cell 0: LLM 函数 ---
# 原版: openai.Completion.create(model="text-davinci-002", prompt=prompt, ...)
# 我们: openai chat completions API
client = None
MODEL = "gpt-5.2"

SYSTEM_MSG = (
    "You are playing a text-based household game. "
    "You will be given a task and interaction history. "
    "Output ONLY the next action (e.g. 'go to desk 1', 'take mug 1 from desk 1', "
    "'use desklamp 1', 'think: I need to find...') with NO extra text, "
    "NO prefix like '>' or 'Action:', just the raw action string."
)

def llm(prompt, stop=["\n"]):
    """替换官方的 completion API 为 chat API（加 system message 引导）"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=100,
        stop=stop,
    )
    text = response.choices[0].message.content or ""
    text = text.strip()
    # chat 模型可能自动加 "> " 前缀，去掉
    if text.startswith('> '):
        text = text[2:]
    return text


# --- 官方 Cell 1: 环境初始化 + process_ob ---
def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


# --- 官方 Cell 3: alfworld_run（一字不改，原封不动） ---
def alfworld_run(prompt, to_print=True, ob=''):
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 50):
        action = llm(init_prompt + prompt, stop=['\n']).strip()
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        if done:
            return reward
    return 0


# --- 官方 Cell 4: 主评测循环（task type prefix 映射） ---
prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}


# ============================================================
# Tee: 同时输出到终端和日志文件
# ============================================================
class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
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


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.2"))
    parser.add_argument("--num-games", type=int, default=134)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE"))
    args = parser.parse_args()

    MODEL = args.model

    # 日志
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "_")
    log_file = LOG_DIR / f"alfworld_gpt_{model_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Tee(log_file)
    print(f"Log: {log_file}")
    print(f"Model: {MODEL}")

    # API（从环境变量或命令行参数读取）
    assert args.api_key, "请设置 OPENAI_API_KEY 环境变量或传入 --api-key"
    assert args.api_base, "请设置 OPENAI_API_BASE 环境变量或传入 --api-base"
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    print("Testing API...")
    test = llm("Say 'ok'.", stop=["\n"])
    print(f"API OK: {test.strip()}")

    # ReAct prompts（官方的 few-shot 示例）
    with open(REACT_PROMPTS) as f:
        d = json.load(f)

    # ALFWorld 环境
    os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA
    from alfworld.agents.environment import get_environment
    with open(EVAL_CONFIG) as reader:
        config = yaml.safe_load(reader)

    # 展开环境变量
    def expand(obj):
        if isinstance(obj, str): return os.path.expandvars(obj)
        elif isinstance(obj, dict): return {k: expand(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [expand(x) for x in obj]
        return obj
    config = expand(config)

    split = "eval_out_of_distribution"
    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    alfred_env = get_environment(env_type)(config, train_eval=split)
    env = alfred_env.init_env(batch_size=1)

    num_games = min(args.num_games, alfred_env.num_games)
    print(f"Split: {split}")
    print(f"Games: {num_games}")
    print("=" * 60)

    # --- 官方 Cell 4: 评测循环（几乎原封不动） ---
    cnts = [0] * 6
    rs = [0] * 6
    start = time.time()

    for game_no in range(num_games):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(f"\n[Game {game_no+1}/{num_games}] {name}")
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                # 2 个 few-shot 示例（和官方一样）
                prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0'] + '\nHere is the task.\n'
                r = alfworld_run(prompt, ob=ob)
                rs[i] += r
                cnts[i] += 1
                break
        print(f"Result: {'WON' if r else 'LOST'}")
        print(f"Running: {game_no+1} games, rs={rs}, cnts={cnts}, rate={sum(rs)}/{sum(cnts)}={sum(rs)/max(sum(cnts),1):.1%}")
        print('---')

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Model:   {MODEL}")
    print(f"Games:   {sum(cnts)}")
    print(f"Success: {sum(rs)}/{sum(cnts)} = {sum(rs)/max(sum(cnts),1):.1%}")
    print(f"Score:   {sum(rs)/max(sum(cnts),1)*100:.1f}")
    print(f"Time:    {elapsed:.0f}s")
    print(f"\nPer task:")
    for (k, v), s, c in zip(prefixes.items(), rs, cnts):
        if c > 0:
            print(f"  {k:30s} {s}/{c} = {s/c:.0%}")
    print("=" * 60)

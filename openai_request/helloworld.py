import os
import openai
from openai import Client

os.environ["OPENAI_API_KEY"] = "sk-1234"
os.environ["OPENAI_BASE_URL"] = "http://10.150.240.117:38888"

# ANSI 颜色码，用于高亮显示
LIGHT_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

# 按家族分类的模型列表
model_names = [
    # ===== GPT-4o 家族（最新、性能最好） =====
    "gpt-4o",                                    # 基础版
    "gpt-4o_2024-05-13",                        # 日期特定版本
    "gpt-4o_2024-08-06",
    "gpt-4o_2024-11-20",
    "gpt-4o-mini",                              # 轻量版
    "gpt-4o-mini_2024-07-18",
    "gpt-4o-realtime-preview",                  # 实时预览版
    "gpt-4o-realtime-preview_2024-10-01",

    # ===== GPT-4 家族 =====
    "gpt-4",                                    # 基础版
    "gpt-4_0125-Preview",                       # 预览版
    "gpt-4_0314",                               # 日期特定版本
    "gpt-4-32k",                               # 32k上下文窗口版本
    "gpt-4-32k_0314",
    "gpt-4-32k_0613",
    "gpt-4.1",                                 # 4.1版本
    "gpt-4.1_2025-04-14",
    "gpt-4.1-mini",                            # 4.1轻量版
    "gpt-4.1-mini_2025-04-14",
    "gpt-4.1-nano",                            # 4.1超轻量版
    "gpt-4.1-nano_2025-04-14",
    "gpt-4.5-preview",                         # 4.5预览版
    "gpt-4.5-preview_2025-02-27",
    "gpt-4_turbo-2024-04-09",                 # turbo版本

    # ===== GPT-3.5 家族 =====
    "gpt-35-turbo",                            # 基础版
    "gpt-35-turbo_0301",                       # 日期特定版本
    "gpt-35-turbo_1106",
    "gpt-35-turbo-instruct",                   # instruct版本
    "gpt-35-turbo-instruct_0914",

    # ===== O系列（内部优化版本） =====
    "o1",                                      # 第一代
    "o1-mini",
    "o1-mini_2024-09-12",
    "o1-preview",
    "o1_2024-12-17",
    "o3",                                      # 第三代
    "o3-mini",
    "o3-mini_2025-01-31",
    "o3_2025-04-16",
    "o4-mini",                                 # 第四代
    "o4-mini_2025-04-16",

    # ===== DeepSeek 家族 =====
    "DeepSeek-V3",                             # V3版本
    "Deepseek-V3-0324",
    "Deepseek-R1",                             # R1版本
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3-0324",

    # ===== Phi 家族（微软） =====
    "phi-3-5-mini-instruct",                   # 3.5系列
    "phi-3-5-moe-instruct",
    "phi-3-medium-128k-instruct",              # 128k上下文窗口
    "phi-4",                                   # 4.0系列
    "phi-4-mini-reasoning",                    # 推理增强版
    "phi-4-reasoning",

    # ===== Llama 家族（Meta） =====
    "Meta-Llama-3.1-8B-Instruct",              # 3.1版本，8B参数
    "meta-llama/Llama-3.3-70B-Instruct",       # 3.3版本，70B参数

    # ===== Mixtral 家族 =====
    "mistralai/Mixtral-8x7B-Instruct-v0.1",    # 8专家混合模型

    # ===== Embedding 模型（用于文本向量化） =====
    "text-embedding-ada-002",                   # 不要用于对话，专门用于生成文本向量
    "text-embedding-ada-002_2",
]

# 遍历模型名称
for m in model_names:
    os.environ["CHAT_MODEL"] = m
    client = Client()
    print(f"{LIGHT_GREEN}{m}{RESET_COLOR}")
    try:
        # 跳过 Embedding 模型，因为它们不支持对话
        if "embedding" in m.lower():
            print("Skipping embedding model as it doesn't support chat completions")
            continue
            
        response = client.chat.completions.create(
            model=os.environ["CHAT_MODEL"],
            messages=[
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": "Who were the founders of Microsoft?"},
            ],
        )
        print(response.choices[0].message.content)
    except openai.RateLimitError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
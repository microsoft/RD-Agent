import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 构造数据，每个字典对应表中一行记录
data = [
    {"Model": "GPT-4o", "DataType": "Data I",    "Exec": 0.714, "Format": 0.330, "Corr1": 0.367, "Corr2": 0.540},
    {"Model": "GPT-4o", "DataType": "Data II",   "Exec": 0.540, "Format": 0.111, "Corr1": 1.000, "Corr2": 1.000},
    {"Model": "GPT-4o", "DataType": "Data III",  "Exec": 0.778, "Format": 0.531, "Corr1": 0.422, "Corr2": 0.861},
    {"Model": "GPT-4o", "DataType": "Mean Value","Exec": 0.677, "Format": 0.324, "Corr1": 0.494, "Corr2": 0.741},

    {"Model": "LLaMa-3.1-70b", "DataType": "Data I",    "Exec": 0.690, "Format": 0.265, "Corr1": 0.239, "Corr2": 0.493},
    {"Model": "LLaMa-3.1-70b", "DataType": "Data II",   "Exec": 0.889, "Format": 0.003, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "LLaMa-3.1-70b", "DataType": "Data III",  "Exec": 0.806, "Format": 0.569, "Corr1": 0.145, "Corr2": 0.261},
    {"Model": "LLaMa-3.1-70b", "DataType": "Mean Value","Exec": 0.794, "Format": 0.279, "Corr1": 0.186, "Corr2": 0.363},

    {"Model": "GPT-4-turbo", "DataType": "Data I",    "Exec": 0.717, "Format": 0.456, "Corr1": 0.665, "Corr2": 0.949},
    {"Model": "GPT-4-turbo", "DataType": "Data II",   "Exec": 0.711, "Format": 0.056, "Corr1": 0.522, "Corr2": 0.556},
    {"Model": "GPT-4-turbo", "DataType": "Data III",  "Exec": 0.967, "Format": 0.622, "Corr1": 0.518, "Corr2": 1.000},
    {"Model": "GPT-4-turbo", "DataType": "Mean Value","Exec": 0.798, "Format": 0.378, "Corr1": 0.568, "Corr2": 0.835},

    {"Model": "GPT-35-turbo", "DataType": "Data I",    "Exec": 0.556, "Format": 0.100, "Corr1": 0.323, "Corr2": 0.453},
    {"Model": "GPT-35-turbo", "DataType": "Data II",   "Exec": 0.567, "Format": 0.000, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "GPT-35-turbo", "DataType": "Data III",  "Exec": 0.767, "Format": 0.389, "Corr1": 0.431, "Corr2": 0.696},
    {"Model": "GPT-35-turbo", "DataType": "Mean Value","Exec": 0.630, "Format": 0.163, "Corr1": 0.251, "Corr2": 0.383},

    {"Model": "Claude 3.5 Sonnet", "DataType": "Data I",    "Exec": 0.874, "Format": 0.592, "Corr1": 0.052, "Corr2": 0.159},
    {"Model": "Claude 3.5 Sonnet", "DataType": "Data II",   "Exec": 0.578, "Format": 0.094, "Corr1": 0.117, "Corr2": 0.444},
    {"Model": "Claude 3.5 Sonnet", "DataType": "Data III",  "Exec": 0.756, "Format": 0.234, "Corr1": -0.000,"Corr2": 0.000},
    {"Model": "Claude 3.5 Sonnet", "DataType": "Mean Value","Exec": 0.736, "Format": 0.307, "Corr1": 0.056, "Corr2": 0.201},

    {"Model": "Phi3-128k", "DataType": "Data I",    "Exec": 0.117, "Format": 0.111, "Corr1": 0.186, "Corr2": 0.222},
    {"Model": "Phi3-128k", "DataType": "Data II",   "Exec": 0.172, "Format": 0.000, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "Phi3-128k", "DataType": "Data III",  "Exec": 0.056, "Format": 0.022, "Corr1": 0.063, "Corr2": 0.084},
    {"Model": "Phi3-128k", "DataType": "Mean Value","Exec": 0.115, "Format": 0.044, "Corr1": 0.083, "Corr2": 0.102},

    {"Model": "phi4:14b-16k", "DataType": "Data I",    "Exec": 0.635, "Format": 0.158, "Corr1": 0.003, "Corr2": 0.043},
    {"Model": "phi4:14b-16k", "DataType": "Data II",   "Exec": 0.532, "Format": 0.001, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "phi4:14b-16k", "DataType": "Data III",  "Exec": 0.516, "Format": 0.102, "Corr1": 0.000, "Corr2": 0.001},
    {"Model": "phi4:14b-16k", "DataType": "Mean Value","Exec": 0.561, "Format": 0.087, "Corr1": 0.001, "Corr2": 0.015},

    {"Model": "mistral:7b-16k", "DataType": "Data I",    "Exec": 0.571, "Format": 0.064, "Corr1": 0.011, "Corr2": 0.112},
    {"Model": "mistral:7b-16k", "DataType": "Data II",   "Exec": 0.516, "Format": 0.001, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "mistral:7b-16k", "DataType": "Data III",  "Exec": 0.540, "Format": 0.137, "Corr1": 0.005, "Corr2": 0.031},
    {"Model": "mistral:7b-16k", "DataType": "Mean Value","Exec": 0.542, "Format": 0.067, "Corr1": 0.005, "Corr2": 0.048},

    {"Model": "mistral-nemo:12b-16k", "DataType": "Data I",    "Exec": 0.603, "Format": 0.125, "Corr1": 0.015, "Corr2": 0.143},
    {"Model": "mistral-nemo:12b-16k", "DataType": "Data II",   "Exec": 0.508, "Format": 0.002, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "mistral-nemo:12b-16k", "DataType": "Data III",  "Exec": 0.548, "Format": 0.072, "Corr1": 0.000, "Corr2": 0.002},
    {"Model": "mistral-nemo:12b-16k", "DataType": "Mean Value","Exec": 0.553, "Format": 0.066, "Corr1": 0.005, "Corr2": 0.048},

    {"Model": "mistral-small:22b-16k", "DataType": "Data I",    "Exec": 0.770, "Format": 0.397, "Corr1": 0.133, "Corr2": 0.462},
    {"Model": "mistral-small:22b-16k", "DataType": "Data II",   "Exec": 0.825, "Format": 0.039, "Corr1": 0.033, "Corr2": 0.306},
    {"Model": "mistral-small:22b-16k", "DataType": "Data III",  "Exec": 0.849, "Format": 0.623, "Corr1": 0.021, "Corr2": 0.268},
    {"Model": "mistral-small:22b-16k", "DataType": "Mean Value","Exec": 0.815, "Format": 0.353, "Corr1": 0.062, "Corr2": 0.345},

    {"Model": "gemma2:9b-16k", "DataType": "Data I",    "Exec": 0.635, "Format": 0.158, "Corr1": 0.003, "Corr2": 0.043},
    {"Model": "gemma2:9b-16k", "DataType": "Data II",   "Exec": 0.532, "Format": 0.001, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "gemma2:9b-16k", "DataType": "Data III",  "Exec": 0.516, "Format": 0.102, "Corr1": 0.000, "Corr2": 0.001},
    {"Model": "gemma2:9b-16k", "DataType": "Mean Value","Exec": 0.561, "Format": 0.087, "Corr1": 0.001, "Corr2": 0.015},

    {"Model": "gemma2:27b-16k", "DataType": "Data I",    "Exec": 0.619, "Format": 0.127, "Corr1": 0.010, "Corr2": 0.118},
    {"Model": "gemma2:27b-16k", "DataType": "Data II",   "Exec": 0.500, "Format": 0.001, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "gemma2:27b-16k", "DataType": "Data III",  "Exec": 0.548, "Format": 0.134, "Corr1": 0.001, "Corr2": 0.006},
    {"Model": "gemma2:27b-16k", "DataType": "Mean Value","Exec": 0.556, "Format": 0.087, "Corr1": 0.004, "Corr2": 0.041},

    {"Model": "qwen2.5:7b-16k", "DataType": "Data I",    "Exec": 0.778, "Format": 0.378, "Corr1": 0.042, "Corr2": 0.104},
    {"Model": "qwen2.5:7b-16k", "DataType": "Data II",   "Exec": 0.849, "Format": 0.030, "Corr1": 0.039, "Corr2": 0.333},
    {"Model": "qwen2.5:7b-16k", "DataType": "Data III",  "Exec": 0.913, "Format": 0.691, "Corr1": -0.001,"Corr2": 0.087},
    {"Model": "qwen2.5:7b-16k", "DataType": "Mean Value","Exec": 0.847, "Format": 0.366, "Corr1": 0.027, "Corr2": 0.175},

    {"Model": "qwen2.5-coder:32b-16k", "DataType": "Data I",    "Exec": 0.786, "Format": 0.430, "Corr1": 0.065, "Corr2": 0.412},
    {"Model": "qwen2.5-coder:32b-16k", "DataType": "Data II",   "Exec": 0.825, "Format": 0.036, "Corr1": 0.028, "Corr2": 0.231},
    {"Model": "qwen2.5-coder:32b-16k", "DataType": "Data III",  "Exec": 0.857, "Format": 0.627, "Corr1": 0.065, "Corr2": 0.137},
    {"Model": "qwen2.5-coder:32b-16k", "DataType": "Mean Value","Exec": 0.823, "Format": 0.364, "Corr1": 0.053, "Corr2": 0.260},

    {"Model": "marco-o1:7b-16k", "DataType": "Data I",    "Exec": 0.611, "Format": 0.136, "Corr1": 0.000, "Corr2": 0.003},
    {"Model": "marco-o1:7b-16k", "DataType": "Data II",   "Exec": 0.492, "Format": 0.001, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "marco-o1:7b-16k", "DataType": "Data III",  "Exec": 0.619, "Format": 0.188, "Corr1": 0.001, "Corr2": 0.005},
    {"Model": "marco-o1:7b-16k", "DataType": "Mean Value","Exec": 0.574, "Format": 0.108, "Corr1": 0.001, "Corr2": 0.002},

    {"Model": "deepseek-r1:32b-16k", "DataType": "Data I",    "Exec": 0.698, "Format": 0.220, "Corr1": 0.048, "Corr2": 0.361},
    {"Model": "deepseek-r1:32b-16k", "DataType": "Data II",   "Exec": 0.627, "Format": 0.084, "Corr1": 0.069, "Corr2": 0.556},
    {"Model": "deepseek-r1:32b-16k", "DataType": "Data III",  "Exec": 0.857, "Format": 0.621, "Corr1": 0.312, "Corr2": 1.000},
    {"Model": "deepseek-r1:32b-16k", "DataType": "Mean Value","Exec": 0.728, "Format": 0.308, "Corr1": 0.143, "Corr2": 0.639},

    {"Model": "qwq:32b-16k", "DataType": "Data I",    "Exec": 0.579, "Format": 0.105, "Corr1": 0.022, "Corr2": 0.112},
    {"Model": "qwq:32b-16k", "DataType": "Data II",   "Exec": 0.500, "Format": 0.005, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "qwq:32b-16k", "DataType": "Data III",  "Exec": 0.540, "Format": 0.131, "Corr1": 0.001, "Corr2": 0.004},
    {"Model": "qwq:32b-16k", "DataType": "Mean Value","Exec": 0.540, "Format": 0.081, "Corr1": 0.008, "Corr2": 0.039},

    {"Model": "llama3.3:70b-16k", "DataType": "Data I",    "Exec": 0.698, "Format": 0.271, "Corr1": 0.069, "Corr2": 0.475},
    {"Model": "llama3.3:70b-16k", "DataType": "Data II",   "Exec": 0.722, "Format": 0.001, "Corr1": 0.000, "Corr2": 0.000},
    {"Model": "llama3.3:70b-16k", "DataType": "Data III",  "Exec": 0.619, "Format": 0.280, "Corr1": 0.083, "Corr2": 0.338},
    {"Model": "llama3.3:70b-16k", "DataType": "Mean Value","Exec": 0.754, "Format": 0.376, "Corr1": 0.071, "Corr2": 0.191},

    {"Model": "qwen2.5:72b-16k", "DataType": "Data I",    "Exec": 0.794, "Format": 0.428, "Corr1": 0.116, "Corr2": 0.338},
    {"Model": "qwen2.5:72b-16k", "DataType": "Data II",   "Exec": 0.587, "Format": 0.038, "Corr1": 0.012, "Corr2": 0.111},
    {"Model": "qwen2.5:72b-16k", "DataType": "Data III",  "Exec": 0.881, "Format": 0.664, "Corr1": 0.083, "Corr2": 0.124},
    {"Model": "qwen2.5:72b-16k", "DataType": "Mean Value","Exec": 0.741, "Format": 0.386, "Corr1": 0.091, "Corr2": 0.377},

    {"Model": "deepseek-r1:70b-16k", "DataType": "Data I",    "Exec": 0.873, "Format": 0.584, "Corr1": 0.165, "Corr2": 0.467},
    {"Model": "deepseek-r1:70b-16k", "DataType": "Data II",   "Exec": 0.659, "Format": 0.019, "Corr1": 0.026, "Corr2": 0.222},
    {"Model": "deepseek-r1:70b-16k", "DataType": "Data III",  "Exec": 0.889, "Format": 0.692, "Corr1": 0.113, "Corr2": 0.356},
    {"Model": "deepseek-r1:70b-16k", "DataType": "Mean Value","Exec": 0.807, "Format": 0.432, "Corr1": 0.102, "Corr2": 0.348},

    {"Model": "mistral-large:123b-16k", "DataType": "Data I",    "Exec": 0.881, "Format": 0.546, "Corr1": 0.112, "Corr2": 0.446},
    {"Model": "mistral-large:123b-16k", "DataType": "Data II",   "Exec": 0.556, "Format": 0.016, "Corr1": 0.022, "Corr2": 0.222},
    {"Model": "mistral-large:123b-16k", "DataType": "Data III",  "Exec": 0.881, "Format": 0.671, "Corr1": 0.077, "Corr2": 0.141},
    {"Model": "mistral-large:123b-16k", "DataType": "Mean Value","Exec": 0.772, "Format": 0.411, "Corr1": 0.070, "Corr2": 0.270},

    {"Model": "DeepSeek-R1", "DataType": "Data I",    "Exec": 0.978, "Format": 0.813, "Corr1": 0.064, "Corr2": 0.154},
    {"Model": "DeepSeek-R1", "DataType": "Data II",   "Exec": 0.933, "Format": 0.365, "Corr1": 0.500, "Corr2": 1.000},
    {"Model": "DeepSeek-R1", "DataType": "Data III",  "Exec": 0.516, "Format": 0.102, "Corr1": 0.000, "Corr2": 0.001},
    {"Model": "DeepSeek-R1", "DataType": "Mean Value","Exec": 0.561, "Format": 0.087, "Corr1": 0.001, "Corr2": 0.015},

    {"Model": "DeepSeek-V3", "DataType": "Data I",    "Exec": 0.889, "Format": 0.596, "Corr1": 0.157, "Corr2": 0.394},
    {"Model": "DeepSeek-V3", "DataType": "Data II",   "Exec": 0.711, "Format": 0.026, "Corr1": 0.036, "Corr2": 0.222},
    {"Model": "DeepSeek-V3", "DataType": "Data III",  "Exec": 0.896, "Format": 0.654, "Corr1": 0.154, "Corr2": 0.489},
    {"Model": "DeepSeek-V3", "DataType": "Mean Value","Exec": 0.832, "Format": 0.425, "Corr1": 0.116, "Corr2": 0.368},
]

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 获取所有模型名称，并生成横轴位置映射（按字母排序）
models = sorted(df["Model"].unique())
model_to_x = {model: i for i, model in enumerate(models)}

# 为不同类型定义横向偏移（便于区分重叠数据）
offsets = {"Data I": -0.2, "Data II": -0.1, "Data III": 0.0, "Mean Value": 0.2}
# 定义颜色：将所有"Data"系列统一为一种颜色，Mean Value 为另一颜色
colors = {"Data I": "steelblue", "Data II": "steelblue", "Data III": "steelblue", "Mean Value": "darkorange"}

# 指定需要绘制的指标
metrics = ["Exec", "Format", "Corr1", "Corr2"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    # 为防止重复图例记录，记录已添加的标签
    added_labels = set()
    # 对于每个数据点，根据模型横坐标加上偏移绘制散点
    for _, row in df.iterrows():
        x = model_to_x[row["Model"]] + offsets[row["DataType"]]
        y = row[metric]
        label = row["DataType"] if row["DataType"] not in added_labels else None
        if label is not None:
            added_labels.add(label)
        ax.scatter(x, y, color=colors[row["DataType"]], s=80, label=label, zorder=3)
    # 设置横轴
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=90, fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(metric, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

# 设置整体图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)
fig.suptitle("Performance of Various Models on Different Metrics (Exec, Format, Corr1, Corr2)\n(Blue Dots: Data I/II/III; Orange Dots: Mean Value)", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# 输出图片到文件
plt.savefig("output.png", dpi=300, bbox_inches='tight')
plt.show()

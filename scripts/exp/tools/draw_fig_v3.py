import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 构造数据，每个字典对应表中一行记录
data = [
    {"Model": "GPT-4o", "DataType": "Mean Value","Exec": 0.677, "Format": 0.324, "Corr1": 0.494, "Corr2": 0.741, "num_gold": 11},
    {"Model": "GPT-4-turbo", "DataType": "Mean Value","Exec": 0.798, "Format": 0.378, "Corr1": 0.568, "Corr2": 0.835, "num_gold": 19},
    {"Model": "GPT-3.5-turbo", "DataType": "Mean Value","Exec": 0.630, "Format": 0.163, "Corr1": 0.251, "Corr2": 0.383, "num_gold": 9},
    {"Model": "Mistral:7b", "DataType": "Mean Value","Exec": 0.542, "Format": 0.067, "Corr1": 0.005, "Corr2": 0.048, "num_gold": 1},
    {"Model": "Mistral NeMo:12b", "DataType": "Mean Value","Exec": 0.553, "Format": 0.066, "Corr1": 0.005, "Corr2": 0.048, "num_gold": 1},
    {"Model": "Mistral Small 3:22b", "DataType": "Mean Value","Exec": 0.815, "Format": 0.353, "Corr1": 0.062, "Corr2": 0.345, "num_gold": 3},
    {"Model": "Qwen2.5:7b", "DataType": "Mean Value","Exec": 0.847, "Format": 0.366, "Corr1": 0.027, "Corr2": 0.175, "num_gold": 3},
    {"Model": "Qwen2.5-Coder:32b", "DataType": "Mean Value","Exec": 0.823, "Format": 0.364, "Corr1": 0.053, "Corr2": 0.260, "num_gold": 3},
    {"Model": "DeepSeek-R1:32b", "DataType": "Mean Value","Exec": 0.728, "Format": 0.308, "Corr1": 0.143, "Corr2": 0.639, "num_gold": 16},
    {"Model": "QwQ:32b", "DataType": "Mean Value","Exec": 0.540, "Format": 0.081, "Corr1": 0.008, "Corr2": 0.039, "num_gold": 1},
    {"Model": "Qwen2.5:72b", "DataType": "Mean Value","Exec": 0.741, "Format": 0.386, "Corr1": 0.091, "Corr2": 0.377, "num_gold": 8},
    {"Model": "DeepSeek-R1:70b", "DataType": "Mean Value","Exec": 0.807, "Format": 0.432, "Corr1": 0.102, "Corr2": 0.348, "num_gold": 8},
    {"Model": "Mistral Large 2:123b", "DataType": "Mean Value","Exec": 0.772, "Format": 0.411, "Corr1": 0.070, "Corr2": 0.270, "num_gold": 5},
    {"Model": "DeepSeek-V3:671b", "DataType": "Mean Value","Exec": 0.561, "Format": 0.087, "Corr1": 0.001, "Corr2": 0.015, "num_gold": 19},
    {"Model": "DeepSeek-R1:671b", "DataType": "Mean Value","Exec": 0.832, "Format": 0.425, "Corr1": 0.116, "Corr2": 0.368, "num_gold": 9}
]


# 输出图片到文件
plt.savefig("output.png", dpi=300, bbox_inches='tight')
plt.show()

# 数据集管理模块

本模块提供 LLM Finetune 场景的数据集管理功能，支持从 HuggingFace 加载数据集并导出到本地。

## 设计目标

1. **防止数据泄露**：只加载和导出训练数据（train split），测试数据永远不会出现在本地
2. **任务分离**：每个 benchmark 任务独立配置，如 PANORAMA 的三个专利任务分别注册
3. **接口简洁**：提供统一的 `prepare()` 和 `load_split()` 接口

## 架构概览

```
HuggingFace Dataset
        │
        ▼
load_dataset_split()  ──→  Dataset 对象  ──→  export_dataset()  ──→  本地 JSON 文件
        │                       │                                        │
        │                       ▼                                        ▼
        │               用于分析/统计                           process_data.py
        │                                                              │
        ▼                                                              ▼
   load_split()                                                 LlamaFactory 训练
```

## 核心接口

### 高层接口（推荐使用）

```python
from rdagent.scenarios.finetune.datasets import prepare, load_split, DATASETS

# 1. 查看已注册的数据集
print(DATASETS.keys())
# ['panorama-par4pc', 'panorama-noc4pc', 'panorama-pi4pc', 'deepscaler']

# 2. 准备数据集（下载 train split 并导出到本地）
path = prepare("panorama-par4pc")
# 输出: datasets/panorama-par4pc/train.json

# 3. 加载为 Dataset 对象（用于分析）
ds = load_split("panorama-par4pc")
print(f"样本数: {len(ds)}, 列: {ds.column_names}")

# 4. 加载验证集（如果配置了 eval_split）
ds_eval = load_split("panorama-par4pc", split="eval")
```

### 底层接口（灵活使用）

```python
from rdagent.scenarios.finetune.download.hf import load_dataset_split, export_dataset

# 直接从 HuggingFace 加载任意数据集
ds = load_dataset_split(
    repo_id="LG-AI-Research/PANORAMA",
    split="train",
    data_dir="PAR4PC"  # 指定子目录
)

# 导出为本地文件
export_dataset(ds, "/path/to/output.json", format="json")
# 支持格式: json, jsonl, csv, parquet
```

## 数据集配置

每个数据集通过 `DatasetConfig` 配置：

```python
@dataclass
class DatasetConfig:
    repo_id: str                          # HuggingFace 仓库 ID
    train_split: str = "train"            # 训练集 split 名称
    eval_split: str | None = None         # 验证集 split 名称（可选）
    data_dir: str | None = None           # 仓库内子目录（如 "PAR4PC"）
    data_files: str | list[str] | None = None  # 指定文件路径
    name: str | None = None               # subset/config 名称
    export_format: str = "parquet"        # 导出格式
```

### 自定义数据处理（prepare_fn）

对于需要特殊处理的数据集（如列名映射、数据过滤等），可以在数据集目录下创建 `prepare.py` 文件：

```python
# datasets/my-dataset/prepare.py
from datasets import Dataset

def prepare(ds: Dataset) -> Dataset:
    """Transform dataset after loading."""
    # 示例：重命名列
    if "old_column" in ds.column_names:
        ds = ds.rename_column("old_column", "new_column")
    return ds
```

系统会自动加载并应用这个函数，无需在 `DatasetConfig` 中显式配置。

### 调用链路

```
load_split("chemcot-rxn") 或 prepare("chemcot-rxn")
              │
              ▼
    _load_prepare_fn("chemcot-rxn")
              │
              ├──► 查找 chemcot-rxn/prepare.py
              │         │
              │         ▼ 存在
              │    动态导入 prepare() 函数
              │
              ▼
    load_dataset_split(..., prepare_fn)
              │
              ├──► 多文件场景 (data_files 为列表)
              │         │
              │         ▼
              │    逐文件加载 → prepare(ds) → 合并
              │
              ▼
    返回处理后的 Dataset
```

**关键点**：
- 系统自动检测 `<dataset>/prepare.py` 是否存在
- 多文件场景：每个文件单独加载并应用 `prepare()`，然后合并
- 单文件场景：加载后直接应用 `prepare()`

### 已注册数据集

| 名称 | 仓库 | 任务描述 | 训练样本数 |
|------|------|----------|------------|
| `panorama-par4pc` | LG-AI-Research/PANORAMA | 专利先前技术检索 | 54,028 |
| `panorama-noc4pc` | LG-AI-Research/PANORAMA | 专利新颖性/非显而易见性分类 | 136,211 |
| `panorama-pi4pc` | LG-AI-Research/PANORAMA | 专利段落识别 | 64,210 |
| `deepscaler` | agentica-org/DeepScaleR-Preview-Dataset | 数学推理 | 40,315 |
| `chemcot-mol_und` | OpenMol/ChemCoTDataset | 分子理解（官能团计数、环计数） | ~3,000 |
| `chemcot-mol_edit` | OpenMol/ChemCoTDataset | 分子编辑（添加/删除官能团） | ~4,000 |
| `chemcot-mol_opt` | OpenMol/ChemCoTDataset | 分子优化（LogP、溶解度、QED） | 5,587 |
| `chemcot-rxn` | OpenMol/ChemCoTDataset | 反应预测 + 条件推荐 | 6,820 |

## 添加新数据集

在 `__init__.py` 的 `DATASETS` 字典中添加配置：

```python
DATASETS["my-dataset"] = DatasetConfig(
    repo_id="organization/dataset-name",
    train_split="train",
    eval_split="validation",  # 可选
    data_dir="subset-dir",    # 如果数据在子目录
)
```

然后可选地在 `datasets/my-dataset/` 目录下添加自定义 `README.md`。

## 目录结构

```
datasets/
├── __init__.py          # 主模块：prepare(), load_split(), DATASETS
├── README.md            # 本文档
├── deepscaler/
│   └── README.md        # DeepScaleR 数据集说明
├── panorama-par4pc/
│   └── README.md        # PAR4PC 任务说明
├── panorama-noc4pc/
│   └── README.md        # NOC4PC 任务说明
├── panorama-pi4pc/
│   └── README.md        # PI4PC 任务说明
├── chemcot-mol_und/
│   └── README.md        # ChemCoT 分子理解任务
├── chemcot-mol_edit/
│   └── README.md        # ChemCoT 分子编辑任务
├── chemcot-mol_opt/
│   └── README.md        # ChemCoT 分子优化任务
└── chemcot-rxn/
    ├── README.md        # ChemCoT 反应预测任务
    └── prepare.py       # 自定义处理：统一 schema（rcr.json 列名映射）
```

## 与 LlamaFactory 集成

`prepare()` 函数导出的 JSON 文件可以直接被 `process_data.py` 读取并转换为 LlamaFactory 所需的格式：

```python
# process_data.py 示例
import json
from pathlib import Path

# 读取 prepare() 导出的数据
data_path = Path(os.environ["FT_DATASET_PATH"]) / "panorama-par4pc" / "train.json"
with open(data_path) as f:
    raw_data = [json.loads(line) for line in f]

# 转换为 Alpaca 格式
output = []
for item in raw_data:
    output.append({
        "instruction": f"Given the patent claim, select relevant prior art:\n{item['context']}",
        "input": json.dumps(item['options']),
        "output": ", ".join(item['gold_answers'])
    })

# 保存
with open("data.json", "w") as f:
    json.dump(output, f, indent=2)
```

## 注意事项

1. **数据安全**：`prepare()` 只会导出 `train_split`，测试数据永远不会泄露
2. **缓存机制**：HuggingFace datasets 库会自动缓存下载的数据
3. **Token 配置**：私有数据集需要设置 `HF_TOKEN` 环境变量
4. **eval_split 用途**：用于训练过程中的验证评估，可通过 `load_split(name, split="eval")` 加载

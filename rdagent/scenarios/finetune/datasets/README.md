# 数据集管理模块

本模块管理 LLM Finetune 场景的数据集，通过 `snapshot_download` 下载完整的 HuggingFace 仓库。

## 设计目标

1. **简洁性**: 下载完整的 HF 仓库，保留原始文件结构
2. **可扩展性**: 支持可选的 `post_download_fn` 进行自定义处理（如删除测试集）

## 使用方法

```python
from rdagent.scenarios.finetune.datasets import prepare, prepare_all, DATASETS

# 1. 查看已注册的数据集
print(DATASETS.keys())
# ['chemcot', 'panorama', 'deepscaler', 'financeiq']

# 2. 准备单个数据集（下载到本地）
path = prepare("chemcot")
# 下载至: datasets/chemcot/

# 3. 准备所有数据集
prepare_all()
```

## 数据集配置

每个数据集通过 `DatasetConfig` 配置：

```python
@dataclass
class DatasetConfig:
    repo_id: str                                          # HuggingFace 仓库 ID
    post_download_fn: Optional[Callable[[str], None]]     # 下载后处理函数
```

## 已注册数据集

| 名称 | 仓库 | 描述 |
|------|------|------|
| `chemcot` | OpenMol/ChemCoTDataset | 化学推理 + CoT |
| `panorama` | LG-AI-Research/PANORAMA | 专利审查基准 |
| `deepscaler` | agentica-org/DeepScaleR-Preview-Dataset | 数学推理 |
| `financeiq` | Duxiaoman-DI/FinanceIQ | 金融问答 |

## 添加新数据集

在 `__init__.py` 的 `DATASETS` 字典中添加配置：

```python
DATASETS["my-dataset"] = DatasetConfig(
    repo_id="organization/dataset-name",
    post_download_fn=my_cleanup_function,  # 可选
)
```

---

## README 替换机制

**重要**: 下载数据集时，本地 README 会覆盖 HuggingFace 原始 README。

### 工作原理

```python
# __init__.py 中的逻辑
custom_readme = Path(__file__).parent / name / "README.md"
if custom_readme.exists():
    shutil.copy(custom_readme, out_dir / "README.md")
```

1. 数据集下载完成后，检查 `datasets/{name}/README.md` 是否存在
2. 如果存在，用本地版本覆盖下载目录中的 README
3. 这样可以为每个数据集提供**定制化的文档**

### 目录结构

```
rdagent/scenarios/finetune/datasets/
├── __init__.py          # 主模块: prepare(), prepare_all(), DATASETS
├── README.md            # 本文档
├── chemcot/
│   └── README.md        # ChemCoT 数据集文档（会覆盖 HF 原版）
├── panorama/
│   └── README.md        # PANORAMA 数据集文档（会覆盖 HF 原版）
├── deepscaler/
│   └── README.md        # DeepScaleR 数据集文档（会覆盖 HF 原版）
└── financeiq/
    └── README.md        # FinanceIQ 数据集文档（会覆盖 HF 原版）
```

---

## README 编写规范

为每个数据集编写 README 时，建议包含以下内容：

### 1. 基础信息（必需）

```markdown
# 数据集名称

简要描述 + 论文链接

**Repository**: [HuggingFace 链接]

## Overview

数据集规模、来源、用途的概述
```

### 2. 数据集规模（必需）

```markdown
## Dataset Scale

| 类别 | 子任务 | 样本数 |
|------|--------|--------|
| xxx | xxx | 1,234 |
| **Total** | **N subtasks** | **总数** |
```

### 3. 数据字段说明（必需）

```markdown
## Data Fields

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | string | 唯一标识符 |
| `query` | string | 问题/指令 |
| `answer` | string | 答案 |
| ... | ... | ... |
```

### 4. CoT 质量评估（关键）

这是最重要的部分，直接告诉使用者数据是否可用、如何处理：

```markdown
## CoT Quality Assessment

**IMPORTANT**: [数据质量的核心警告]

| Dimension | Value |
|-----------|-------|
| baseline_quality | low / medium / high / N/A |
| task_type | math / chemistry / legal / ... |
| polish_difficulty | low / medium / high |

**Baseline**: [详细说明]
- 如果有 CoT: 说明来源、验证方式、质量问题
- 如果没有 CoT: 明确标注 "NO CoT"，说明必须生成
```

### 5. Baseline 性能（推荐）

```markdown
## Baseline Performance

| Task | Best Model | Score |
|------|-----------|-------|
| xxx | GPT-4o | 85.2% |
```

### 6. 许可证（必需）

```markdown
## License

MIT / CC-BY-NC-4.0 / ...
```

---

## 示例参考

- **DeepScaleR**: [deepscaler/README.md](deepscaler/README.md) - 标杆示例，CoT Quality Assessment 写得最清晰
- **ChemCoT**: [chemcot/README.md](chemcot/README.md) - 有 CoT 但需要精化的情况
- **PANORAMA**: [panorama/README.md](panorama/README.md) - 没有 CoT 的情况

---

## 注意事项

1. **Token**: 私有数据集需要设置 `HF_TOKEN` 环境变量
2. **缓存**: HuggingFace hub 会自动缓存下载内容
3. **强制刷新**: 使用 `prepare(name, force=True)` 重新下载
4. **README 优先级**: 本地 README 会覆盖 HuggingFace 原版，确保文档一致性

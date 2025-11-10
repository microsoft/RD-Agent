# RD-Agent Data 组件

数据集处理 pipeline，从搜索到 SFT 格式转换的完整流程。

## 两阶段处理流程

### Phase 1: 数据采集（test_pipeline.py）

**流程**：搜索 → 下载 → 检查 → LLM过滤 → 选择性迁移

**涉及模块**：
- `search_api.py`：HuggingFace API 封装，支持 3 维搜索（domain/size/language）
- `dataset_agent.py`：LLM 驱动的搜索代理，自动选择最佳数据集
- `dataset_inspector.py`：数据集检查器，LLM 分析哪些文件有用
- `dataset_manager.py`：存储管理，选择性迁移有用文件到 `./datasets/raw/`

**运行命令**：
```bash
python test_pipeline.py
```

**输出**：
- 数据集下载到临时目录 `/tmp/dataset_staging`
- 有用文件迁移到 `./datasets/raw/`
- 自动过滤垃圾文件，节省存储空间

---

### Phase 2: SFT 转换（test_convert_pipeline.py）

**流程**：加载数据 → Schema分析 → 智能路由 → 转换为 Alpaca 格式

**涉及模块**：
- `schema_analyzer.py`：LLM 分析数据 schema，识别 instruction/output 列
- `data_converter.py`：转换为 Alpaca 格式，支持单轮/多轮对话
- `data_cleaner.py`：数据清洗（去重、长度过滤、LLM质量打分）
- `sft_processor.py`：主流程编排，智能路由（Light Path/Heavy Path）

**智能路由**：
- **Light Path**：数据质量 >0.8 → 简单转换 + 清洗
- **Heavy Path**：数据复杂 → 直接 LLM 批量转换

**运行命令**：
```bash
python test_convert_pipeline.py  # 需要先运行 test_pipeline.py
```

**输出**：
- Alpaca JSON 格式文件保存到 `./datasets/sft/`
- 包含 instruction/input/output 字段
- 经过去重和质量过滤（≥7.0分）

## 文件详细说明

### Phase 1: 数据采集相关文件

#### search_api.py（135行）
- **核心类**：`HuggingFaceSearchAPI`
- **主要功能**：封装 HuggingFace Hub API，提供数据集搜索能力
- **关键方法**：
  - `search_datasets()`：支持 domain（模糊匹配）、size、language 三维搜索
  - `get_dataset_info()`：获取单个数据集的详细信息
- **特点**：自动过滤需要申请权限的 gated datasets，返回结构化搜索结果

#### dataset_agent.py（499行）
- **核心类**：`DatasetSearchAgent`
- **主要功能**：LLM 驱动的智能搜索代理，自动生成搜索参数并选择最佳数据集
- **关键方法**：
  - `search_and_download()`：完整流程（搜索→选择→下载）
  - `_generate_search_params()`：LLM 根据任务描述生成搜索参数
  - `_select_best_dataset()`：LLM 基于 4 维评估选择最佳数据集
  - `_apply_license_blacklist()`：过滤 NC/ND/GPL 等限制性 license
- **特点**：混合重试策略（第1次 LLM 智能调整，后续规则式放松参数）

#### dataset_inspector.py（658行）
- **核心类**：`DatasetInspector`
- **主要功能**：数据集质量检查和文件分析
- **关键方法**：
  - `inspect()`：加载数据集并提取结构信息（列名、样本数、数据类型等）
  - `check_quality()`：规则式质量检查（不依赖 LLM）
  - `analyze_files_for_sft()`：LLM 分析哪些文件对 SFT 训练有用
  - `_preview_xxx_file()`：支持 csv/json/parquet 等格式的文件预览
- **特点**：智能文件分类，自动识别并过滤垃圾文件，节省存储空间

#### dataset_manager.py（109行）
- **核心类**：`DatasetManager`
- **主要功能**：数据集存储和迁移管理
- **关键方法**：
  - `migrate_dataset_selective()`：基于文件分析结果，只迁移有用文件
- **特点**：组织化存储结构（raw/ 和 converted/ 分离），自动创建目录

### Phase 2: SFT 转换相关文件

#### schema_analyzer.py
- **核心类**：`SchemaAnalyzer`
- **主要功能**：LLM 分析数据集的 schema 结构
- **关键方法**：
  - `analyze()`：识别 instruction/input/output 列，判断单轮/多轮对话
  - `_validate_schema_result()`：验证 LLM 输出格式是否正确
- **返回格式**：包含 data_type、instruction_col、output_col、input_col、reasoning
- **特点**：3 次重试机制，失败时有启发式 fallback

#### data_converter.py
- **核心类**：`DataConverter`
- **主要功能**：将各种格式数据转换为标准 Alpaca 格式
- **关键方法**：
  - `convert_to_alpaca()`：主转换入口
  - `_convert_single_turn()`：单轮 QA 转换逻辑
  - `_convert_multi_turn()`：多轮对话转换，保留历史作为 context
  - `_extract_metadata()`：智能提取元数据（白名单优先，黑名单排除）
- **支持格式**：csv、json、jsonl、parquet、arrow

#### data_cleaner.py
- **核心类**：`DataCleaner`
- **主要功能**：数据清洗和质量过滤
- **清洗流程**：
  1. 去重：基于 instruction+output 的 MD5 哈希
  2. 长度过滤：设置最小/最大长度阈值
  3. 质量打分：LLM 批量评分（10条/批），保留 ≥7.0 分
- **特点**：20 workers 并行处理，采样策略（超过 10000 条只评分前 10000）

#### sft_processor.py
- **核心类**：`SFTProcessor`、`CheckpointManager`
- **主要功能**：生产级 SFT 数据准备系统，完整 pipeline 编排
- **智能路由**：
  - Light Path（质量>0.8）：schema分析 → 简单转换 → 清洗
  - Heavy Path（质量≤0.8）：直接 LLM 批量转换
- **关键特性**：
  - 断点续传：batch 级别 checkpoint，中断可恢复
  - 并行处理：20 workers 同时处理
  - 增量保存：每完成 1 个 batch 立即保存
- **特点**：整合所有上述模块，提供统一入口

### 辅助文件

#### prompts.yaml
- **功能**：集中管理所有 LLM 提示词模板
- **包含提示词**：
  - search_params：生成搜索参数
  - dataset_selection：数据集选择评估
  - schema_analysis_for_sft：schema 结构分析
  - quality_scoring_batch：批量质量打分
  - heavy_conversion：Heavy Path 直接转换
- **特点**：使用模板系统渲染，便于维护和更新

#### __init__.py
- **功能**：模块导出和便捷函数
- **导出内容**：所有主要类 + `convert_to_sft()` 一行代码函数
- **便捷函数**：自动完成从搜索到输出的完整流程

## 快速使用

### 方式一：两步运行
```bash
# Phase 1: 数据采集
python test_pipeline.py

# Phase 2: SFT 转换
python test_convert_pipeline.py
```

### 方式二：一行代码
```python
from rdagent.components.data import convert_to_sft

convert_to_sft(
    input_path="data/raw/",
    output_file="output/alpaca.json",
    task_description="数学推理数据集"
)
```




## 依赖关系

```
Phase 1:                          Phase 2:
dataset_agent → search_api        sft_processor
     ↓                                 ├── schema_analyzer
dataset_inspector                      ├── data_converter
     ↓                                 └── data_cleaner
dataset_manager                              ↑
                                       prompts.yaml
```

## Alpaca 输出格式

```json
{
    "instruction": "问题或指令",
    "input": "输入上下文（可选）",
    "output": "回答或输出",
    "metadata": {
        "category": "分类",
        "difficulty": "难度"
    }
}
```


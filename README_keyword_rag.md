# Keyword-Only RAG System Usage Guide

这个文档说明如何使用 HybridRAGSystem 的关键词搜索模式，以及专门的 KeywordOnlyRAGSystem。

## 概述

我们提供了三种方式来使用仅基于关键词（BM25）的检索，不依赖 embedding API：

1. **KeywordOnlyRAGSystem** - 专门的关键词检索系统
2. **HybridRAGSystem** 禁用语义搜索 - 通过参数控制
3. **运行时切换模式** - 动态切换检索模式

## 使用方法

### 方法 1: KeywordOnlyRAGSystem

```python
from rdagent.scenarios.data_science.proposal.exp_gen.rag_hybrid_2 import KeywordOnlyRAGSystem

# 初始化（不需要 API backend）
rag_system = KeywordOnlyRAGSystem(
    api_backend=None,  # 不需要 API
    cache_dir="./keyword_cache"
)

# 添加想法
ideas = {
    "idea_1": {
        "problem": "模型过拟合",
        "method": "使用正则化和数据增强",
        "context": "深度学习小数据集场景"
    }
}
rag_system.add_ideas(ideas)

# 检索相关想法
results = rag_system.retrieve_relevant_ideas(
    problem_name="overfitting_issue",
    problem_data={
        "problem": "模型在训练集上表现好但测试集差",
        "reason": "可能是过拟合问题"
    },
    top_k=3
)
```

### 方法 2: HybridRAGSystem 禁用语义搜索

```python
from rdagent.scenarios.data_science.proposal.exp_gen.rag_hybrid_2 import HybridRAGSystem

# 初始化时禁用语义搜索
rag_system = HybridRAGSystem(
    api_backend=None,  # 可以为 None
    cache_dir="./hybrid_keyword_cache",
    enable_semantic=False  # 关键参数
)

# 使用方式与 KeywordOnlyRAGSystem 相同
```

### 方法 3: 运行时切换模式

```python
from rdagent.scenarios.data_science.proposal.exp_gen.rag_hybrid_2 import HybridRAGSystem

# 初始化为混合模式
rag_system = HybridRAGSystem(api_backend=your_api_backend)

# 切换到关键词模式
rag_system.set_semantic_mode(False)

# 检查当前模式
stats = rag_system.get_statistics()
print(f"当前模式: {stats['search_mode']['mode']}")  # 输出: keyword_only

# 切换回混合模式（如果需要）
rag_system.set_semantic_mode(True)
```

## 主要优势

### Keyword-Only 模式的优势：

1. **无需 API 调用** - 不需要调用 embedding API，节省成本
2. **更快的响应** - 跳过 embedding 计算，检索速度更快
3. **离线工作** - 完全不依赖网络连接
4. **精确匹配** - 对关键词和短语匹配效果好
5. **资源消耗低** - 内存和计算需求较小

### 适用场景：

- **快速原型开发和测试**
- **没有 embedding API 访问权限**
- **需要精确关键词匹配的场景**
- **离线环境使用**
- **资源受限的环境**

## 配置选项

### 权重调整（仅关键词模式）

```python
# 调整字段权重
rag_system.update_weights(
    field_weights={
        'problem': 1.0,    # 问题描述权重最高
        'method': 0.8,     # 方法描述权重中等
        'context': 0.5     # 上下文权重较低
    }
)
```

### 检索参数

```python
results = rag_system.retrieve_relevant_ideas(
    problem_name="your_problem",
    problem_data=your_problem_data,
    top_k=5,           # 返回前5个结果
    min_score=0.3,     # 最低相似度阈值
    use_reranking=False # 关键词模式一般不需要重排
)
```

## 性能比较

```python
import time

# 测试关键词检索性能
start = time.time()
keyword_results = keyword_rag.retrieve_relevant_ideas(problem_name, problem_data)
keyword_time = time.time() - start

print(f"关键词检索时间: {keyword_time:.3f} 秒")
print(f"找到结果数: {len(keyword_results)}")
```

## 系统状态检查

```python
stats = rag_system.get_statistics()
print(f"检索模式: {stats['search_mode']['mode']}")
print(f"BM25索引状态: {stats['index_status']['bm25']}")
print(f"总想法数: {stats['total_ideas']}")
print(f"系统类型: {stats.get('system_type', 'hybrid')}")
```

## 缓存管理

关键词模式的缓存更简单，只包含：
- 想法库 (idea_pool.json)
- BM25索引 (bm25_index.pkl)
- 元数据 (metadata.json)

不包含：
- embeddings (idea_embeddings.pkl)
- FAISS索引 (faiss.index)

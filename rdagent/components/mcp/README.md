# MCP (Model Context Protocol) 集成

RD-Agent的MCP集成提供了统一的文档搜索和错误解决能力，通过配置驱动的服务架构支持多种MCP服务。

## 🚀 快速开始

### 1. 基础配置

创建 `mcp_config.json` 配置文件：

```json
{
  "mcpServices": {
    "context7": {
      "url": "http://localhost:8123/mcp",
      "timeout": 30.0,
      "handler": "Context7Handler", 
      "enabled": true,
      "extra_config": {
        "model": "gpt-4",
        "api_key": "your-api-key",
        "max_retries": 3
      }
    }
  }
}
```

### 2. 环境变量设置

在 `.env` 文件中配置全局开关：

```bash
# MCP系统总开关
MCP_ENABLED=true

# 全局缓存开关（可选）
MCP_CACHE_ENABLED=false
```

### 3. 基本使用

```python
import asyncio
from rdagent.components.mcp import query_mcp

async def main():
    # 方式1: 自动选择最佳服务 (推荐)
    result = await query_mcp(
        query="LightGBMError: No OpenCL device found",
        full_code="import lightgbm as lgb..."
    )
    
    # 方式2: 指定单个服务
    result = await query_mcp(
        query="AttributeError: 'DataFrame' object has no attribute 'append'",
        services="context7",
        strategy="auto"
    )
    
    # 方式3: 并发查询多个服务对比
    results = await query_mcp(
        query="Best Python web framework comparison",
        services=["context7", "other_service"],
        strategy="parallel"
    )
    
    print(result)

asyncio.run(main())
```

## 📋 配置架构

MCP系统采用**双层配置架构**，实现了灵活的控制粒度：

### 全局控制层 (环境变量)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MCP_ENABLED` | `true` | 控制整个MCP系统开关 |
| `MCP_CACHE_ENABLED` | `false` | 控制全局缓存功能 |

### 服务配置层 (JSON配置)

```json
{
  "mcpServices": {
    "服务名": {
      "url": "服务地址",
      "timeout": 超时时间(秒),
      "handler": "处理器类名",
      "enabled": 是否启用该服务,
      "extra_config": {
        "服务特定配置项": "配置值"
      }
    }
  }
}
```

## 🔧 支持的服务

### Context7 - 文档搜索服务

Context7提供基于错误消息的智能文档搜索和解决方案建议。

**配置示例：**
```json
{
  "mcpServices": {
    "context7": {
      "url": "http://localhost:8123/mcp",
      "timeout": 30.0,
      "handler": "Context7Handler",
      "enabled": true,
      "extra_config": {
        "model": "gpt-4",
        "api_key": "sk-your-key",
        "max_retries": 3,
        "retry_delay": 2.0
      }
    }
  }
}
```

**使用示例：**
```python
# 错误消息调试
result = await query_mcp(
    query="错误信息或问题描述",
    services="context7",
    full_code="完整的出错代码（可选）",
    max_rounds=3,
    verbose=True
)
```

## 📖 完整使用示例

参考 `context7/examples/example.py`，展示了三种典型使用场景：

### 1. 错误调试场景
```python
async def debug_error():
    error_message = """
    LightGBMError: No OpenCL device found
    File "main.py", line 77, in lgb_objective
    gbm = lgb.train(params, train_data, num_boost_round=100)
    """
    
    full_code = """
    params = {
        'device': 'gpu',  # 问题出现在这里
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    }
    gbm = lgb.train(params, train_data, num_boost_round=100)
    """
    
    result = await query_mcp(
        "context7", 
        query=error_message,
        full_code=full_code
    )
```

### 2. 自动服务选择
```python
async def auto_selection():
    # 让系统自动选择最佳服务处理问题
    result = await query_mcp_auto(
        query="pandas DataFrame.append method removed in 2.0"
    )
```

### 3. 服务管理
```python
from rdagent.components.mcp import (
    list_available_mcp_services,
    is_service_available,
    get_service_status
)

# 检查可用服务
services = list_available_mcp_services()
print(f"可用服务: {services}")

# 检查特定服务状态
if is_service_available("context7"):
    print("Context7服务可用")

# 获取整体状态
status = get_service_status()
print(status)
```

## 🛠️ 服务管理API

### 初始化和注册
```python
from rdagent.components.mcp import initialize_mcp_registry, register_mcp_handler

# 从配置文件初始化
registry = initialize_mcp_registry("path/to/mcp_config.json")

# 手动注册自定义服务
custom_handler = MyCustomHandler("custom_service")
register_mcp_handler("custom_service", custom_handler)
```

### 状态检查
```python
from rdagent.components.mcp import (
    list_available_mcp_services,
    is_service_available, 
    get_mcp_service_info
)

# 列出所有可用服务
services = list_available_mcp_services()

# 检查特定服务
available = is_service_available("context7")

# 获取详细信息
info = get_mcp_service_info()
```

## 🔍 故障排除

### 常见问题

**1. "MCP system is globally disabled"**
```bash
# 解决：检查环境变量设置
export MCP_ENABLED=true
```

**2. "No API key found for Context7Handler"**
```json
{
  "mcpServices": {
    "context7": {
      "extra_config": {
        "api_key": "your-actual-api-key"
      }
    }
  }
}
```

**3. "Service not available"**
- 检查 `mcp_config.json` 中服务的 `enabled` 字段
- 确认服务URL可访问
- 检查网络连接和超时设置

### 调试模式
```python
# 启用详细日志
result = await query_mcp(
    "context7",
    query="your query",
    verbose=True  # 显示详细处理过程
)
```

## 🔧 扩展开发

### 添加新的MCP服务

1. **实现服务处理器**：
```python
from rdagent.components.mcp.handlers import BaseMCPHandler

class MyServiceHandler(BaseMCPHandler):
    async def query(self, query: str, **kwargs) -> str:
        # 实现具体查询逻辑
        return "response"
```

2. **更新配置文件**：
```json
{
  "mcpServices": {
    "my_service": {
      "url": "http://localhost:8125/mcp",
      "handler": "MyServiceHandler",
      "enabled": true
    }
  }
}
```

3. **注册和使用**：
```python
from rdagent.components.mcp import register_mcp_handler

handler = MyServiceHandler("my_service", service_url="http://localhost:8125/mcp")
register_mcp_handler("my_service", handler)

# 使用新的统一接口
result = await query_mcp(query="test query", services="my_service")
```

## 📊 性能优化

### 缓存配置
```bash
# 启用全局缓存
MCP_CACHE_ENABLED=true
```

### 超时设置
```json
{
  "mcpServices": {
    "context7": {
      "timeout": 60.0,  // 增加超时时间
      "extra_config": {
        "max_retries": 5,    // 增加重试次数
        "retry_delay": 1.0   // 重试间隔
      }
    }
  }
}
```

### 并发控制
```python
import asyncio

# 批量查询
async def batch_query():
    queries = ["query1", "query2", "query3"]
    tasks = [query_mcp("context7", query=q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

## 📚 相关资源

- [Model Context Protocol 规范](https://modelcontextprotocol.io/)
- [Context7 文档服务](https://context7.ai/)
- [RD-Agent 项目主页](https://github.com/microsoft/RD-Agent)

## 🚀 新统一接口详解

### `query_mcp()` 统一函数

```python
async def query_mcp(
    query: str,                                    # 必须: 查询内容
    services: Optional[Union[str, List[str]]] = None,  # 可选: 服务选择
    strategy: str = "auto",                        # 可选: 查询策略
    **kwargs                                       # 可选: 传递给Handler的参数
) -> Union[str, Dict[str, str], None]:
```

#### 参数说明
- **`query`**: 查询内容（错误消息、问题描述等）
- **`services`**: 服务选择
  - `None`: 自动选择所有可用服务 
  - `"context7"`: 指定单个服务
  - `["context7", "other"]`: 指定多个服务列表
- **`strategy`**: 查询策略
  - `"auto"`: 自动选择，返回单个最佳结果 (`str`)
  - `"parallel"`: 并发查询，返回所有服务结果 (`Dict[service_name, result]`)

#### 返回值
- **strategy="auto"**: 返回 `str` 或 `None`
- **strategy="parallel"**: 返回 `Dict[str, str]` 或 `{}`

### 使用模式总结

```python
# 🎯 模式1: 最简单 - 自动选择最佳服务
result = await query_mcp("ImportError: No module named 'pandas'")

# 🎯 模式2: 指定服务 - 明确使用某个服务
result = await query_mcp(
    query="LightGBM GPU error",
    services="context7"
)

# 🎯 模式3: 并发对比 - 获得多个服务的不同观点
results = await query_mcp(
    query="What's the best Python web framework?",
    services=["context7", "stackoverflow_api", "github_copilot"],
    strategy="parallel"
)
# 返回: {
#   "context7": "Flask is lightweight...",
#   "stackoverflow_api": "Django is robust...",
#   "github_copilot": "FastAPI is modern..."
# }

# 🎯 模式4: 故障转移 - 有优先级的服务列表
result = await query_mcp(
    query="complex algorithm question",
    services=["premium_service", "free_service", "backup_service"],
    strategy="auto"  # 会按顺序尝试，使用第一个可用的
)
```

### 向后兼容性

旧的API仍然可用但会显示废弃警告：

```python
# ⚠️ 废弃但可用
result = await query_mcp_auto("error message")  
# 等价于: query_mcp("error message", strategy="auto")

# ❌ 旧接口已移除
result = await query_mcp("service", "query")  # 不再支持位置参数
```

---

**注意**：确保在生产环境中妥善管理API密钥，避免在代码中硬编码敏感信息。推荐使用环境变量或安全的配置管理工具。
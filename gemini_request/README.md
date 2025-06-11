# OpenAI Compatible API Client

这是一个用于调用OpenAI兼容API的Python客户端示例。

## 功能特性

- 支持使用 `requests` 库直接调用API
- 支持使用 `openai` SDK调用API
- 支持流式响应
- 完整的错误处理
- 易于扩展和定制

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 配置API

在 `try.py` 文件中修改以下配置：

```python
BASE_URL = "https://amstrongzyf-gemini-57.deno.dev/chat/completions"
API_KEY = "your-api-key-here"  # 根据需要设置API密钥
```

### 2. 运行示例

```bash
python try.py
```

### 3. 在代码中使用

```python
from try import OpenAICompatibleClient

# 创建客户端
client = OpenAICompatibleClient(BASE_URL, API_KEY)

# 准备消息
messages = [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "你好！"}
]

# 使用requests方法调用
response = client.chat_completion_requests(messages)

# 使用OpenAI SDK方法调用
response = client.chat_completion_openai_sdk(messages)
```

## API参数说明

- `messages`: 对话消息列表
- `model`: 使用的模型名称 (默认: "gpt-3.5-turbo")
- `temperature`: 温度参数，控制随机性 (默认: 0.7)
- `max_tokens`: 最大token数量
- `stream`: 是否使用流式响应 (默认: False)

## 注意事项

1. 请确保API URL是正确的OpenAI兼容接口
2. 根据实际需要设置正确的API密钥
3. 流式响应目前仅在requests方法中实现
4. 网络超时设置为30秒，可根据需要调整 
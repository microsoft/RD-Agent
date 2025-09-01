# MCP (Model Context Protocol) for RD-Agent

## 🎯 What is MCP?

MCP (Model Context Protocol) is a unified interface system that enables RD-Agent to connect with various external services for enhanced code generation, documentation search, and error resolution. It provides a **configuration-driven, parallel-processing architecture** that allows multiple MCP services to work together seamlessly.

### Key Features:
- 🚀 **Parallel Multi-Service Processing**: Query multiple services simultaneously with a single API call
- 🔧 **Unified Interface**: Single `query_mcp()` function with three flexible modes
- 📋 **Configuration-Driven**: JSON-based service registration and management
- 🔄 **Auto-Registration**: Services are automatically initialized on first use
- 💾 **Intelligent Caching**: Optional caching system for improved performance
- 🔌 **Plugin Architecture**: Easy to add new MCP services through handlers
- 🌐 **StreamableHTTP Transport**: Uses HTTP/SSE for reliable communication with MCP servers

## 🚀 Quick Start

> **Important**: RD-Agent uses **StreamableHTTP** transport to communicate with MCP servers. 
> All MCP servers must be running with **HTTP transport mode** (not stdio).

> **Configuration Architecture**: The MCP system uses a two-tier configuration approach:
> - **Global controls** → `.env` file (system enable/disable, caching)  
> - **Service configs** → `mcp_config.json` file (URLs, API keys, models, handlers)

### Prerequisites: Start Your MCP Server(Use context7 as an example)

Before using MCP in RD-Agent, you need to start your MCP server with HTTP transport: (*for each mcp tool we provide, we have provided instructions for deployment in the corresponding folder*)

```bash
# Example: Starting Context7 MCP server
cd /path/to/context7
bun run dist/index.js --transport http --port 8123

# Or run in background
nohup bun run dist/index.js --transport http --port 8123 > context7.log 2>&1 &
```

**Key Points:**
- ✅ Use `--transport http` (required for RD-Agent)
- ❌ Don't use `--transport stdio` (not compatible)
- 📝 Note the server URL (e.g., `http://localhost:8123/mcp`)
- 🔧 This URL goes in your `mcp_config.json`

### 1. Environment Configuration

Add the following to your `.env` file:

```bash
# Enable MCP system globally
MCP_ENABLED=True

# Enable caching for better performance (optional)
MCP_CACHE_ENABLED=False

# Enable MCP in Data Science coder (for Kaggle scenarios)
DS_ENABLE_MCP_DOCUMENTATION_SEARCH=True
```

**Note:** Service-specific configurations (URLs, API keys, models) can be optionally configured in `mcp_config.json`. If not specified, the service will use RD-Agent's default LLM configuration.

### 2. Service Configuration

Create or update `mcp_config.json` in your project root:

```json
{
  "mcpServices": {
    "context7": {
      "url": "http://localhost:8123/mcp",
      "timeout": 20.0,
      "handler": "rdagent.components.mcp.context7.handler:Context7Handler",
      "enabled": true,
      "extra_config": {
        // Optional: Override default LLM settings for this service
        "model": "gpt-4o",  // Optional, defaults to RD-Agent's model
        "api_key": "your-api-key",  // Optional, defaults to RD-Agent's API key
        "api_base": "https://api.openai.com/v1"  // Optional, defaults to RD-Agent's base URL
      }
    }
    // Add more services here as needed
  }
}
```

### 3. Basic Code Example

```python
from rdagent.components.mcp import query_mcp
import asyncio

async def main():
    # Auto mode - uses ALL available services in parallel
    result = await query_mcp("How to use pandas DataFrame?")
    
    # Single service mode - query specific service
    result = await query_mcp("lightgbm GPU error", services="context7")
    
    # Multi-service mode - use specified services in parallel
    result = await query_mcp(
        "async/await in Python",
        services=["context7", "another_service"]
    )
    
    print(result)

asyncio.run(main())
```

## 📖 Detailed Configuration Guide

### Environment Variables (.env)

Only these global control variables are needed in `.env`:

| Variable | Description | Default | Required |
|----------|------------|---------|----------|
| `MCP_ENABLED` | Enable/disable entire MCP system | `True` | Yes |
| `MCP_CACHE_ENABLED` | Enable query result caching | `False` | No |
| `MCP_CACHE_PATH` | Path to cache database | `./mcp_cache.db` | No |
| `DS_ENABLE_MCP_DOCUMENTATION_SEARCH` | Enable MCP in Data Science scenarios | `False` | No |

**Important:** Service-specific configurations (API keys, URLs, models) can be optionally placed in `mcp_config.json`. If not provided, services will automatically use RD-Agent's default LLM configuration from your environment.

### Service Configuration (mcp_config.json)

#### Minimal Configuration (uses RD-Agent's default LLM):
```json
{
  "mcpServices": {
    "context7": {
      "url": "http://localhost:8123/mcp",
      "handler": "rdagent.components.mcp.context7.handler:Context7Handler",
      "enabled": true
    }
  }
}
```

#### Full Configuration (with custom LLM settings):
```json
{
  "mcpServices": {
    "<service_name>": {
      "url": "Service endpoint URL",
      "timeout": 120.0,
      "handler": "module.path:HandlerClass",
      "enabled": true,
      "headers": {
        "Custom-Header": "value"
      },
      "extra_config": {
        // Optional: Override RD-Agent's default LLM settings
        "model": "custom-model",
        "api_key": "custom-api-key",
        "api_base": "custom-api-base"
      }
    }
  }
}
```

#### Configuration Fields:
- `url`: MCP service endpoint (required)
- `timeout`: Connection timeout in seconds (default: 120.0)
- `handler`: Handler class path in format `module.path:ClassName` (required)
- `enabled`: Whether the service is active (default: true)
- `headers`: Custom HTTP headers (optional)
- `extra_config`: Service-specific parameters (optional)
  - `model`: Override RD-Agent's default LLM model (optional)
  - `api_key`: Override RD-Agent's default API key (optional)
  - `api_base`: Override RD-Agent's default API base URL (optional)
  - Other service-specific parameters as needed

## 🔧 Usage Modes

### 1. Auto Mode (Default)
Query all available services in parallel. The LLM can access tools from all services simultaneously.

```python
result = await query_mcp("error message")
```

### 2. Single Service Mode
Direct query to a specific service.

```python
result = await query_mcp("error message", services="context7")
```

### 3. Multi-Service Mode
Query specified services in parallel.

```python
result = await query_mcp(
    "error message",
    services=["context7", "code_search"]
)
```

### Additional Parameters

```python
result = await query_mcp(
    query="Your query here",
    services="context7",          # Service selection
    full_code="...",              # Full code context (optional)
    max_rounds=5,                 # Max tool-calling rounds (optional)
    verbose=True,                 # Enable detailed logging
    # Any additional kwargs are passed to the handler
)
```

## 🧪 Testing

### 1. Check Service Status

```python
from rdagent.components.mcp import get_service_status

# Check all services
status = get_service_status()
print(status)
# Output: {
#   "mcp_enabled": true,
#   "available_services": ["context7"],
#   "service_details": {...}
# }
```

### 2. Run Example Tests

```bash
# Run the example with environment variables
dotenv run -- python rdagent/components/mcp/context7/examples/example.py
```

### 3. Simple Integration Test

Create `test_mcp.py`:

```python
import asyncio
from rdagent.components.mcp import (
    query_mcp,
    list_available_mcp_services,
    is_service_available,
    get_service_status
)

async def test_mcp():
    # Check system status
    print("MCP Status:", get_service_status())
    
    # List available services
    print("Available Services:", list_available_mcp_services())
    
    # Check specific service
    print("Context7 Available:", is_service_available("context7"))
    
    # Test query
    if is_service_available("context7"):
        result = await query_mcp(
            "How to fix pandas DataFrame.append() error?",
            services="context7",
            verbose=True
        )
        print("Query Result:", result)

if __name__ == "__main__":
    asyncio.run(test_mcp())
```

Run with:
```bash
python test_mcp.py
```

## 🏗️ Architecture

### Transport Layer

RD-Agent exclusively uses **StreamableHTTP** transport for MCP communication:

- **Protocol**: HTTP with Server-Sent Events (SSE) for streaming
- **Connector**: `StreamableHTTPConnector` handles all MCP server connections
- **Compatibility**: Works with any MCP server running in HTTP mode
- **Not Supported**: stdio, WebSocket, or other transport modes

### Core Components

```
rdagent/components/mcp/
├── __init__.py          # Public API exports
├── unified.py           # Unified interface (query_mcp)
├── registry.py          # Service registry and management
├── connector.py         # StreamableHTTP connector implementation
├── conf.py             # Global configuration
├── cache.py            # Caching system
├── general_handler.py  # Base handler class with LiteLLM integration
└── context7/           # Context7 service implementation
    ├── handler.py      # Context7-specific handler
    ├── prompts/        # Prompt templates
    └── examples/       # Usage examples
```

### Data Flow

```
User Query
    ↓
query_mcp() [Unified Interface]
    ↓
MCPRegistry [Service Management]
    ↓
┌─────────────┬─────────────┬─────────────┐
│  Service 1  │  Service 2  │  Service N  │  [Parallel Processing]
│  (Handler)  │  (Handler)  │  (Handler)  │
└─────────────┴─────────────┴─────────────┘
    ↓              ↓              ↓
StreamableHTTPConnector [Communication Layer]
    ↓
MCP Service Endpoints
    ↓
Combined Results → User
```

## 🔌 Adding New Services

### Method 1: Use GeneralMCPHandler (Recommended for Most Services)

For most MCP services, you can simply use the built-in `GeneralMCPHandler` without writing any code:

#### 1. Start Your MCP Server with HTTP Transport

```bash
# Your MCP server MUST use HTTP transport
your-mcp-server --transport http --port 9000

# Example with different MCP implementations:
# For Node.js based servers
node server.js --transport http --port 9000

# For Python based servers  
python -m mcp_server --transport http --port 9000
```

#### 2. Add Configuration to `mcp_config.json`

```json
{
  "mcpServices": {
    "my_new_service": {
      "url": "http://localhost:9000/mcp",
      "handler": "rdagent.components.mcp.general_handler:GeneralMCPHandler",
      "enabled": true,
      "extra_config": {
        // Optional: Override LLM settings for this service
        "model": "gpt-4",  // Optional
        "api_key": "your-key",  // Optional
        "api_base": "https://api.openai.com/v1"  // Optional
      }
    }
  }
}
```

#### 3. Test Your Service

```python
import asyncio
from rdagent.components.mcp import query_mcp, is_service_available

async def test_new_service():
    if is_service_available("my_new_service"):
        result = await query_mcp("test query", services="my_new_service")
        print(result)
    else:
        print("Service not available")

asyncio.run(test_new_service())
```

That's it! The `GeneralMCPHandler` provides:
- ✅ Multi-round tool calling
- ✅ Automatic retry with checkpoints
- ✅ Caching support
- ✅ Rate limit detection
- ✅ LiteLLM backend integration

### Method 2: Create Custom Handler (For Special Requirements)

Only create a custom handler if you need special processing logic:

#### When to Create a Custom Handler?

- **Custom query preprocessing** - Need to enhance or modify queries
- **Special tool result processing** - Need to format or validate results
- **Service-specific validation** - Need custom error detection
- **Complex prompt engineering** - Need specialized prompts

#### 1. Create a Handler by Extending GeneralMCPHandler

Create `rdagent/components/mcp/myservice/handler.py`:

```python
from rdagent.components.mcp.general_handler import GeneralMCPHandler

class MyServiceHandler(GeneralMCPHandler):
    """Custom handler with special processing logic."""
    
    def preprocess_query(self, query: str, **kwargs) -> str:
        """Add custom preprocessing to the query."""
        # Example: Add context or format the query
        context = kwargs.get("context", "")
        if context:
            return f"Context: {context}\n\nQuery: {query}"
        return query
    
    def handle_tool_result(self, result_text: str, tool_name: str, tool_index: int = 1) -> str:
        """Process tool results with custom logic."""
        # Example: Format or filter results
        if "error" in result_text.lower():
            return f"⚠️ Tool error: {result_text}"
        return result_text
    
    def validate_tool_response(self, tool_name: str, response_text: str) -> None:
        """Custom validation for tool responses."""
        # Example: Check for specific patterns
        if "invalid" in response_text.lower():
            raise MCPConnectionError(f"Invalid response from {tool_name}")
    
    def detect_rate_limit(self, response_text: str) -> bool:
        """Custom rate limit detection."""
        # Add service-specific patterns
        if "quota exceeded" in response_text.lower():
            return True
        return super().detect_rate_limit(response_text)
```

#### 2. Update Configuration

```json
{
  "mcpServices": {
    "myservice": {
      "url": "http://localhost:8080/mcp",
      "handler": "rdagent.components.mcp.myservice.handler:MyServiceHandler",
      "enabled": true
    }
  }
}
```

### Quick Decision Guide

| Use Case | Solution | Handler |
|----------|----------|---------|
| Standard MCP service | Just configure in JSON | `GeneralMCPHandler` |
| Need custom prompts | Create custom handler | Extend `GeneralMCPHandler` |
| Need result formatting | Create custom handler | Extend `GeneralMCPHandler` |
| Need special validation | Create custom handler | Extend `GeneralMCPHandler` |
| Basic tool calling | Just configure in JSON | `GeneralMCPHandler` |

## 🐛 Debugging

### Enable Verbose Logging

```python
# In your code
result = await query_mcp("query", verbose=True)

# Or set environment variable
export RDAGENT_LOG_LEVEL=DEBUG
```



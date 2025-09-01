# Context7 MCP Service for RD-Agent

## 🎯 What is Context7?

Context7 is a Model Context Protocol (MCP) service that provides **up-to-date documentation and code examples** for various libraries and frameworks. It ensures that code generation is based on current, accurate API information rather than outdated training data.

### Key Features:
- 🔍 **Real-time Documentation Search**: Fetches latest library documentation
- 📚 **Version-specific Information**: Provides documentation for specific library versions
- 🛠️ **API Discovery**: Resolves library names to standardized documentation IDs
- 🌐 **HTTP/SSE Transport**: RD-Agent uses HTTP transport for reliable communication

### RD-Agent Integration Features:
- 🎯 **Error-focused Queries**: Optimized for error code analysis and resolution
- 🧠 **Multi-round Tool Calling**: Enhanced workflow with sequential documentation lookups
- 📋 **Code Context Awareness**: Analyzes full code context alongside error messages
- 🔄 **Retry Mechanisms**: Built-in retry logic for robust error handling
- ⚡ **Caching System**: Intelligent caching for repeated queries

## 🚀 Server Deployment

### Step 1: Install Context7 Server

```bash
# Option A: Use the adapted Context7 (with RD-Agent optimizations)
git clone https://github.com/Hoder-zyf/context7

# Option B: Use the official Context7
git clone https://github.com/upstash/context7.git

cd context7

# Install dependencies (requires Bun)
bun install

# Build the project
bun run build
```

### Step 2: Start the Server

> **⚠️ Important**: RD-Agent requires **HTTP transport mode**. Do NOT use stdio mode.

```bash
# Start with HTTP transport on port 8123 (recommended)
bun run dist/index.js --transport http --port 8123

# Run in background (production)
nohup bun run dist/index.js --transport http --port 8123 > context7.log 2>&1 &

# Or use tmux/screen for persistent sessions
tmux new -d -s context7 'bun run dist/index.js --transport http --port 8123'
screen -dmS context7 bun run dist/index.js --transport http --port 8123
```

### Server Options

| Option | Description | Required for RD-Agent |
|--------|-------------|----------------------|
| `--transport http` | Use HTTP transport mode | ✅ Yes (mandatory) |
| `--port 8123` | Port number for HTTP server | Recommended |


### Verify Server is Running

```bash
# Check if server is listening
curl http://localhost:8123/mcp
# Should return MCP protocol information

# Check process
ps aux | grep "context7"

# View logs (if using nohup)
tail -f context7.log
```



## ⚙️ RD-Agent Configuration

### Step 3: Configure RD-Agent

#### Environment Variables (`.env`)

```bash
# Enable MCP system globally
MCP_ENABLED=True

# Optional: Enable caching for better performance
MCP_CACHE_ENABLED=False

# Enable MCP in Data Science scenarios (Kaggle)
DS_ENABLE_MCP_DOCUMENTATION_SEARCH=True
```

#### Service Configuration (`mcp_config.json`)

```json
{
  "mcpServices": {
    "context7": {
      "url": "http://localhost:8123/mcp",
      "timeout": 20.0,
      "handler": "rdagent.components.mcp.context7.handler:Context7Handler",
      "enabled": true,
      "extra_config": {
        // Optional: Override default LLM settings
        // If not specified, uses RD-Agent's default configuration
        "model": "gpt-4",           // Optional
        "api_key": "your-key",      // Optional
        "api_base": "https://..."   // Optional
      }
    }
  }
}
```

### Complete Setup Example

```bash
# 1. Start Context7 server (in terminal 1)
cd /path/to/context7
bun run dist/index.js --transport http --port 8123

# 2. Configure RD-Agent (in project root)
# Edit .env file:
echo "MCP_ENABLED=True" >> .env
echo "DS_ENABLE_MCP_DOCUMENTATION_SEARCH=True" >> .env

# 3. Create/update mcp_config.json with the configuration above

# 4. Test the connection
python -c "from rdagent.components.mcp import get_service_status; print(get_service_status())"
```

## 🧪 Testing the Integration

### Quick Test

```python
# test_context7.py
import asyncio
from rdagent.components.mcp import query_mcp, is_service_available

async def test():
    if is_service_available("context7"):
        result = await query_mcp(
            "How to use pandas DataFrame append?",
            services="context7"
        )
        print(result)
    else:
        print("Context7 service not available")
        print("Check: 1) Server running? 2) mcp_config.json correct?")

asyncio.run(test())
```

### Run the Full Example

```bash
# After server is running and configuration is complete
dotenv run -- python rdagent/components/mcp/context7/examples/example.py
```

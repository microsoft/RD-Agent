# Context7 MCP Server for RD-Agent

## 🎯 What is Context7 MCP?

Context7 is a Model Context Protocol (MCP) server that provides **up-to-date documentation and code examples** for various libraries and frameworks. It acts as a bridge between AI agents and real-time documentation, ensuring that code generation is based on current, accurate API information rather than outdated training data.

### Key Features:
- 🔍 **Real-time Documentation Search**: Fetches latest library documentation
- 📚 **Version-specific Information**: Provides documentation for specific library versions
- 🛠️ **API Discovery**: Resolves library names to standardized documentation IDs
- 🚀 **Multi-transport Support**: Works with stdio, HTTP, and SSE protocols

## 🔄 RD-Agent Integration vs Official Context7

### **Official Context7 MCP Server**
- General-purpose documentation search
- Designed for interactive coding assistants (Cursor, Claude Desktop, etc.)
- Broad library coverage for general development

### **RD-Agent Context7 Integration** 
- 🎯 **Error-focused Queries**: Specifically optimized for error code analysis and resolution
- 🧠 **Multi-round Tool Calling**: Enhanced workflow with sequential documentation lookups
- 📋 **Code Context Awareness**: Analyzes full code context alongside error messages
- 🔄 **Retry Mechanisms**: Built-in retry logic for robust error handling
- 📊 **Research Integration**: Seamlessly integrated into RD-Agent's research workflow
- ⚡ **Caching System**: Intelligent caching for repeated queries in research contexts

## 💻 Setup for RD-Agent integration

### 1. Clone and Build

```bash
# Clone the self-adapted Context7
git clone https://github.com/Hoder-zyf/context7
# Or otherwise use the official Context7(you should use the following command)
# git clone https://github.com/upstash/context7.git

cd context7

# Install dependencies
bun i

# Build the project
bun run build
```

### 2. Run the Server

**Important**: RD-Agent's client uses **HTTP transport by default**, so you should run the server with HTTP mode:

```bash
# Development mode (foreground)
bun run dist/index.js --transport http --port 8123

# Production mode (background with nohup)
nohup bun run dist/index.js --transport http --port 8123 > context7.log 2>&1 &

# Or use tmux for persistent sessions
tmux new-session -d -s context7 'bun run dist/index.js --transport http --port 8123'
```

### CLI Arguments

`context7-mcp` accepts the following CLI flags:

- `--transport <stdio|http|sse>` – Transport to use (`stdio` by default, but **RD-Agent uses `http`**)
- `--port <number>` – Port to listen on when using `http` or `sse` transport (in RD-Agent, we recommend to use port `8123`)



## ⚙️ Environment Configuration in `.env`

### Global MCP Settings

```bash
# Enable MCP system globally
MCP_ENABLED=True

# Enable caching for better performance
MCP_CACHE_ENABLED=True
```

### Context7 Specific Settings

```bash
# Enable Context7 integration
CONTEXT7_ENABLED=true

# Set your MCP server URL (for HTTP transport) 
# should be the same as the port you use to run the server
CONTEXT7_MCP_URL="http://localhost:8123/mcp"

# Configure LLM settings
CONTEXT7_MODEL="gpt-4-turbo"
CONTEXT7_API_KEY="your-openai-api-key"
CONTEXT7_API_BASE="https://api.openai.com/v1"  
```

### Complete Setup Example

```bash
# Start the Context7 MCP server in your terminal
cd context7
nohup bun run dist/index.js --transport http --port 8123 

# Configure RD-Agent environment in .env
MCP_ENABLED=true
MCP_CACHE_ENABLED=true
CONTEXT7_ENABLED=true
CONTEXT7_MCP_URL="http://localhost:8123/mcp"
CONTEXT7_MODEL="gpt-4-turbo" 
CONTEXT7_API_KEY="your-openai-api-key"
CONTEXT7_API_BASE="https://api.openai.com/v1"  
```

## 🧪 Testing the Integration

```
# in your terminal(after you have set up the environment variables in .env and server is running)
dotenv run -- python rdagent/components/mcp/context7/examples/example.py
```

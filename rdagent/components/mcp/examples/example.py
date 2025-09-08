"""Simple MCP Example - Quick Start Guide

This example shows the most common ways to use MCP in RD-Agent.
Just the essentials, no complexity.
"""

import asyncio

from rdagent.components.mcp import MCPAgent, create_agent
from rdagent.log import rdagent_logger as logger


def simple_sync_example():
    """Most common usage: sync query with specific service"""
    logger.info("🔧 1.Simple Sync Example")
    logger.info("-" * 30)

    # Create agent for Context7 service
    agent = MCPAgent(toolsets="context7")

    # Simple error query
    query = "pandas read_csv encoding error"
    logger.info(f"Query: {query}")

    # Sync execution with reasonable timeout (Context7 needs time for tool calls)
    try:
        result = agent.run_sync(query, timeout=120)
        if result:
            logger.info("✅ Got solution:")
            logger.info(result)
        else:
            logger.info("❌ No solution found")
    except Exception as e:
        logger.error(f"⚠️ Query failed: {e}")
        logger.error("💡 Try: longer timeout or simpler query")


def multiple_services_example():
    """Using multiple services at once"""
    logger.info("\n🔀 2.Multiple Services Example")
    logger.info("-" * 30)

    # Use multiple services (if you have them configured)
    agent = MCPAgent(toolsets=["context7", "deepwiki"])  # Will skip unavailable services

    query = "numpy array shape error"
    logger.info(f"Query: {query}")

    try:
        result = agent.run_sync(query, timeout=120)
        if result:
            logger.info("✅ Solution from multiple services:")
            logger.info(result)
        else:
            logger.info("❌ No solution found")
    except Exception as e:
        logger.error(f"⚠️ Query failed: {e}")


def auto_mode_example():
    """Let the agent use all available services"""
    logger.info("\n🤖 3.Auto Mode Example")
    logger.info("-" * 30)

    # No toolsets specified = use all available services
    agent = MCPAgent()

    query = "json decode error"
    logger.info(f"Query: {query}")

    try:
        result = agent.run_sync(query, timeout=120)
        if result:
            logger.info("✅ Solution from auto mode:")
            logger.info(result)
        else:
            logger.info("❌ No solution found")
    except Exception as e:
        logger.error(f"⚠️ Query failed: {e}")


async def simple_async_example():
    """Async version for async environments"""
    logger.info("\n🚀 4.Simple Async Example")
    logger.info("-" * 30)

    # Create agent targeting Context7
    agent = MCPAgent(toolsets="context7")

    # Query about error
    query = "requests timeout error"
    logger.info(f"Query: {query}")

    # Async execution
    try:
        result = await agent.run(query)
        if result:
            logger.info("✅ Got solution:")
            logger.info(result)
        else:
            logger.info("❌ No solution found")
    except Exception as e:
        logger.error(f"⚠️ Query failed: {e}")


def create_agent_example():
    """Using the create_agent convenience function"""
    logger.info("\n⚡ 5.create_agent Convenience Function")
    logger.info("-" * 30)

    query = "python dict comprehension syntax"
    logger.info(f"Query: {query}")

    try:
        # One-liner: create agent and run query immediately
        result = create_agent(["context7"]).run_sync(query, timeout=90)
        if result:
            logger.info("✅ Quick solution:")
            logger.info(result[:200] + "..." if len(result) > 200 else result)
        else:
            logger.info("❌ No solution found")
    except Exception as e:
        logger.error(f"⚠️ Query failed: {e}")
        logger.error("💡 Try: create_agent() for auto mode or check service config")


async def main():
    """Run all examples"""
    logger.info("🎯 MCP Quick Examples")
    logger.info("=" * 50)
    logger.info("Make sure your mcp_config.json is configured!")
    logger.info("=" * 50)

    # Sync examples
    simple_sync_example()
    multiple_services_example()
    auto_mode_example()

    # Async example
    await simple_async_example()

    # Convenience function example
    create_agent_example()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())

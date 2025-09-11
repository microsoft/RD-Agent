import unittest
from rdagent.components.agent.mcp.context7 import SETTINGS
from rdagent.components.agent.context7 import Agent


# def simple_sync_example():
#     """Most common usage: sync query with specific service"""
#     logger.info("🔧 1.Simple Sync Example")
#     logger.info("-" * 30)
#
#     # Create agent for Context7 service
#     agent = MCPAgent(toolsets="context7")
#
#     # Simple error query
#     query = "pandas read_csv encoding error"
#     logger.info(f"Query: {query}")
#
#     # Sync execution with reasonable timeout (Context7 needs time for tool calls)
#     try:
#         result = agent.run_sync(query, timeout=120)
#         if result:
#             logger.info("✅ Got solution:")
#             logger.info(result)
#         else:
#             logger.info("❌ No solution found")
#     except Exception as e:
#         logger.error(f"⚠️ Query failed: {e}")
#         logger.error("💡 Try: longer timeout or simpler query")
#
#
# def multiple_services_example():
#     """Using multiple services at once"""
#     logger.info("\n🔀 2.Multiple Services Example")
#     logger.info("-" * 30)
#
#     # Use multiple services (if you have them configured)
#     agent = MCPAgent(toolsets=["context7", "deepwiki"])  # Will skip unavailable services
#
#     query = "numpy array shape error"
#     logger.info(f"Query: {query}")
#
#     try:
#         result = agent.run_sync(query, timeout=120)
#         if result:
#             logger.info("✅ Solution from multiple services:")
#             logger.info(result)
#         else:
#             logger.info("❌ No solution found")
#     except Exception as e:
#         logger.error(f"⚠️ Query failed: {e}")

class TimeUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def to_str(self, obj):
        return "".join(str(obj).split())

    def test_context7(self):
        context7a = Agent()
        res = context7a.query("pandas read_csv encoding error")
        print(res)


if __name__ == "__main__":
    unittest.main()

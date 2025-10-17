import time
import unittest

from rdagent.components.agent.context7 import Agent


class PydanticTest(unittest.TestCase):
    """
    Test Pydantic-AI agent with Prefect caching

    How it works:
    1. Agent wraps query() with @task(cache_policy=INPUTS) when enable_cache=True
    2. First call: executes and caches to Prefect server
    3. Second call with same input: instant cache hit
    """

    def test_context7_cache(self):
        """Test that caching works correctly"""
        query = "pandas read_csv encoding error"

        print("\n" + "=" * 80)
        print("Testing @task-based caching...")
        print("=" * 80 + "\n")

        # Create agent once - caching enabled by CONTEXT7_ENABLE_CACHE
        agent = Agent()

        # First query - will execute and cache
        print("First query (will execute):")
        start1 = time.time()
        res1 = agent.query(query)
        time1 = time.time() - start1

        print(f"  Time: {time1:.2f}s")
        print(f"  Length: {len(res1)} chars")
        print(f"  Preview: {res1[:100]}...\n")

        # Second query - should hit cache (much faster)
        print("Second query (should hit cache):")
        start2 = time.time()
        res2 = agent.query(query)
        time2 = time.time() - start2

        print(f"  Time: {time2:.2f}s")
        print(f"  Speedup: {time1/time2:.1f}x faster")
        print(f"{'='*80}\n")

        self.assertIsNotNone(res1)
        self.assertGreater(len(res1), 0)
        self.assertEqual(res1, res2, "Cache must return identical result")


if __name__ == "__main__":
    unittest.main()

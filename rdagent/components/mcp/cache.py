"""MCP cache management module.

Provides general caching functionality for MCP tools and query result caching.
Reuses RD-Agent's existing SQLite cache system with permanent caching strategy.
"""

import hashlib
from typing import Any, Dict, Optional

from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import SQliteLazyCache
from rdagent.oai.llm_conf import LLM_SETTINGS


class MCPCache:
    """MCP cache manager based on existing SQLite cache system.

    Uses permanent caching strategy, consistent with LITELLM.
    """

    def __init__(self):
        """Initialize cache manager.

        Uses permanent caching without expiration time.
        """
        self._cache = SQliteLazyCache(cache_location=LLM_SETTINGS.prompt_cache_path)
        self._stats = {"tools_hits": 0, "tools_misses": 0, "query_hits": 0, "query_misses": 0}

    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """Get result from SQLite cache."""
        return self._cache.chat_get(cache_key)

    def _set_cached_result(self, cache_key: str, result: str):
        """Set SQLite cache result."""
        self._cache.chat_set(cache_key, result)

    def get_tools(self, mcp_url: str) -> Optional[Any]:
        """Get cached tools.

        Args:
            mcp_url: MCP service URL

        Returns:
            Cached tools list, returns None if cache miss
        """
        # Tool object serialization is complex, temporarily not implementing tool caching
        self._stats["tools_misses"] += 1
        logger.info(f"Tools cache miss for URL: {mcp_url} (tools caching disabled)")
        return None

    def set_tools(self, mcp_url: str, tools: Any):
        """Set tools cache.

        Args:
            mcp_url: MCP service URL
            tools: Tools list to cache (currently unused)
        """
        # Temporarily not caching tool objects as they contain complex objects that are difficult to serialize
        logger.info(f"Tools caching skipped for URL: {mcp_url} (complex objects)")

    def get_query_result(self, error_message: str) -> Optional[str]:
        """Get cached query result.

        Args:
            error_message: Error message

        Returns:
            Cached query result, returns None if cache miss
        """
        cache_key = f"mcp_query:{hashlib.md5(error_message.encode('utf-8')).hexdigest()}"
        cached_result = self._get_cached_result(cache_key)

        if cached_result:
            self._stats["query_hits"] += 1
            logger.info(f"Query cache hit for key: {cache_key[-8:]}...")
            return cached_result

        self._stats["query_misses"] += 1
        logger.info(f"Query cache miss for key: {cache_key[-8:]}...")
        return None

    def set_query_result(self, error_message: str, result: str):
        """Set query result cache.

        Args:
            error_message: Error message
            result: Query result
        """
        cache_key = f"mcp_query:{hashlib.md5(error_message.encode('utf-8')).hexdigest()}"
        self._set_cached_result(cache_key, result)
        logger.info(f"Query result cached for key: {cache_key[-8:]}...")

    def clear_cache(self):
        """Clear all MCP cache."""
        cleared_count = 0

        # Clear all cache keys with mcp_ prefix
        # Note: This requires traversing the entire database, performance may be poor
        logger.warning("Clearing all MCP cache entries...")

        # Due to SQLite interface limitations, we cannot directly traverse keys, so provide hints
        logger.info(
            "To completely clear MCP cache, please delete the SQLite cache file or use clear_mcp_cache_by_pattern()"
        )

        return cleared_count

    def clear_query_cache(self, error_message: str = None):
        """Clear query cache.

        Args:
            error_message: If specified, only clear cache for specific error message; otherwise clear all query cache
        """
        if error_message:
            # Clear cache for specific query
            cache_key = f"mcp_query:{hashlib.md5(error_message.encode('utf-8')).hexdigest()}"
            # SQLite has no direct delete method, we set to None to "delete"
            self._set_cached_result(cache_key, "")  # Set to empty string to indicate deletion
            logger.info(f"Cleared cache for specific query: {cache_key[-8:]}...")
        else:
            logger.info("To clear all query cache, please use clear_all_mcp_cache() or delete the cache file")

    def get_cache_info(self):
        """Get cache information."""
        stats = self.get_cache_stats()
        cache_file = getattr(self._cache, "cache_location", "unknown")

        info = {"cache_file": cache_file, "stats": stats, "cache_type": "SQLite (shared with LITELLM)"}

        logger.info(f"Cache info: {info}")
        return info

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_tools = self._stats["tools_hits"] + self._stats["tools_misses"]
        total_queries = self._stats["query_hits"] + self._stats["query_misses"]

        return {
            "tools_cache": {
                "hits": self._stats["tools_hits"],
                "misses": self._stats["tools_misses"],
                "hit_rate": self._stats["tools_hits"] / max(total_tools, 1),
                "size": "N/A (SQLite)",
            },
            "query_cache": {
                "hits": self._stats["query_hits"],
                "misses": self._stats["query_misses"],
                "hit_rate": self._stats["query_hits"] / max(total_queries, 1),
                "size": "N/A (SQLite)",
            },
        }

    def log_cache_stats(self):
        """Log cache statistics to log."""
        stats = self.get_cache_stats()
        logger.info(
            f"Cache stats - Tools: {stats['tools_cache']['hits']}/{stats['tools_cache']['hits'] + stats['tools_cache']['misses']} hits "
            f"({stats['tools_cache']['hit_rate']:.2%}), "
            f"Queries: {stats['query_cache']['hits']}/{stats['query_cache']['hits'] + stats['query_cache']['misses']} hits "
            f"({stats['query_cache']['hit_rate']:.2%})"
        )


# Global cache instance
_global_cache: Optional[MCPCache] = None


def get_mcp_cache() -> MCPCache:
    """Get global MCP cache instance.

    Returns:
        MCP cache instance (permanent cache)
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = MCPCache()
    return _global_cache


def clear_mcp_cache_by_file():
    """Clear all cache by deleting SQLite cache file.

    Note: This will clear all cache, including LITELLM cache!
    """
    import os

    from rdagent.oai.llm_conf import LLM_SETTINGS

    cache_file = LLM_SETTINGS.prompt_cache_path
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            logger.info(f"Successfully deleted cache file: {cache_file}")

            # Reset global cache instance
            global _global_cache
            _global_cache = None

            return True
        except Exception as e:
            logger.error(f"Failed to delete cache file {cache_file}: {e}")
            return False
    else:
        logger.info(f"Cache file does not exist: {cache_file}")
        return True


def get_cache_file_info():
    """Get cache file information."""
    import os

    from rdagent.oai.llm_conf import LLM_SETTINGS

    cache_file = LLM_SETTINGS.prompt_cache_path

    if os.path.exists(cache_file):
        stat = os.stat(cache_file)
        size_mb = stat.st_size / (1024 * 1024)

        info = {"file_path": cache_file, "exists": True, "size_mb": round(size_mb, 2), "modified_time": stat.st_mtime}
    else:
        info = {"file_path": cache_file, "exists": False, "size_mb": 0, "modified_time": None}

    logger.info(f"Cache file info: {info}")
    return info

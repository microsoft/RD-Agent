"""MCP cache implementation based on SQliteLazyCache inheritance.

This module provides an MCP cache by extending the existing SQliteLazyCache
with MCP-specific functionality while maintaining consistency with the base system.
"""

import hashlib
from typing import Any, Dict, Optional

from rdagent.components.mcp.conf import get_mcp_global_settings
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import SQliteLazyCache


class MCPCache(SQliteLazyCache):
    """MCP cache extending SQliteLazyCache with MCP-specific features.

    This class inherits all functionality from SQliteLazyCache and adds:
    - MCP-specific key formatting with namespace
    - Query result caching with statistics
    - Tool caching support (disabled for now due to serialization complexity)
    - Improved logging and monitoring
    """

    def __init__(self, namespace: str = "mcp"):
        """Initialize MCP cache.

        Args:
            namespace: Cache namespace to avoid key conflicts (default: "mcp")
        """
        # Use MCP-specific cache file
        mcp_settings = get_mcp_global_settings()
        super().__init__(cache_location=mcp_settings.cache_path)
        self.namespace = namespace
        self._stats = {"query_hits": 0, "query_misses": 0, "tools_hits": 0, "tools_misses": 0}

    def _make_namespaced_key(self, key: str) -> str:
        """Create namespaced cache key.

        Args:
            key: Original cache key

        Returns:
            Namespaced cache key with prefix
        """
        return f"{self.namespace}:{key}"

    def _make_query_key(self, error_message: str) -> str:
        """Create standardized query cache key.

        Args:
            error_message: Error message to cache

        Returns:
            Standardized cache key for query results
        """
        hash_digest = hashlib.md5(error_message.encode("utf-8")).hexdigest()
        return self._make_namespaced_key(f"query:{hash_digest}")

    def _make_tools_key(self, mcp_url: str) -> str:
        """Create standardized tools cache key.

        Args:
            mcp_url: MCP service URL

        Returns:
            Standardized cache key for tools
        """
        url_hash = hashlib.md5(mcp_url.encode("utf-8")).hexdigest()
        return self._make_namespaced_key(f"tools:{url_hash}")

    def get_query_result(self, error_message: str) -> Optional[str]:
        """Get cached query result with statistics tracking.

        Args:
            error_message: Error message to look up

        Returns:
            Cached query result or None if not found
        """
        cache_key = self._make_query_key(error_message)
        result = self.chat_get(cache_key)

        if result:
            self._stats["query_hits"] += 1
            logger.info(f"MCP query cache hit for key: {cache_key[-8:]}...")
            return result

        self._stats["query_misses"] += 1
        logger.info(f"MCP query cache miss for key: {cache_key[-8:]}...")
        return None

    def set_query_result(self, error_message: str, result: str) -> None:
        """Cache query result with standardized key.

        Args:
            error_message: Error message (used for key generation)
            result: Query result to cache
        """
        cache_key = self._make_query_key(error_message)
        self.chat_set(cache_key, result)
        logger.info(f"MCP query result cached for key: {cache_key[-8:]}...")

    def get_tools(self, mcp_url: str) -> Optional[Any]:
        """Get cached tools (currently disabled due to serialization complexity).

        Args:
            mcp_url: MCP service URL

        Returns:
            None (tools caching is disabled)
        """
        # Tools caching is disabled due to complex object serialization issues
        self._stats["tools_misses"] += 1
        logger.info(f"MCP tools cache miss for URL: {mcp_url} (tools caching disabled)")
        return None

    def set_tools(self, mcp_url: str, tools: Any) -> None:
        """Set tools cache (currently disabled).

        Args:
            mcp_url: MCP service URL
            tools: Tools object to cache (currently unused)
        """
        # Tools caching is disabled due to complex object serialization issues
        logger.info(f"MCP tools caching skipped for URL: {mcp_url} (complex objects)")

    def clear_query_cache(self, error_message: Optional[str] = None) -> bool:
        """Clear query cache entries.

        Args:
            error_message: If specified, clear only this specific query.
                          If None, provides instructions for clearing all.

        Returns:
            True if operation completed successfully
        """
        if error_message:
            cache_key = self._make_query_key(error_message)
            # Use empty string to indicate deletion (SQLite limitation workaround)
            self.chat_set(cache_key, "")
            logger.info(f"Cleared MCP cache for specific query: {cache_key[-8:]}...")
            return True
        else:
            logger.info("To clear all MCP query cache, use clear_namespace() method or delete the cache file")
            return False

    def clear_namespace(self) -> int:
        """Clear all cache entries for this namespace.

        Note: Due to SQLite interface limitations, this is not efficiently implemented.
        Consider using clear_cache_by_file() for complete cache clearing.

        Returns:
            Number of entries cleared (currently always 0 due to implementation limits)
        """
        logger.warning(f"Clearing all MCP cache entries for namespace: {self.namespace}")
        logger.info("Due to SQLite interface limitations, use clear_cache_by_file() " "for complete cache clearing")
        return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary containing cache statistics and hit rates
        """
        total_tools = self._stats["tools_hits"] + self._stats["tools_misses"]
        total_queries = self._stats["query_hits"] + self._stats["query_misses"]

        return {
            "namespace": self.namespace,
            "cache_file": self.cache_location,
            "tools_cache": {
                "hits": self._stats["tools_hits"],
                "misses": self._stats["tools_misses"],
                "total": total_tools,
                "hit_rate": self._stats["tools_hits"] / max(total_tools, 1),
                "status": "disabled (complex serialization)",
            },
            "query_cache": {
                "hits": self._stats["query_hits"],
                "misses": self._stats["query_misses"],
                "total": total_queries,
                "hit_rate": self._stats["query_hits"] / max(total_queries, 1),
                "status": "active",
            },
        }

    def log_cache_stats(self) -> None:
        """Log current cache statistics."""
        stats = self.get_cache_stats()
        logger.info(
            f"MCP Cache [{self.namespace}] - "
            f"Tools: {stats['tools_cache']['hits']}/{stats['tools_cache']['total']} hits "
            f"({stats['tools_cache']['hit_rate']:.2%}), "
            f"Queries: {stats['query_cache']['hits']}/{stats['query_cache']['total']} hits "
            f"({stats['query_cache']['hit_rate']:.2%})"
        )

    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information including file details.

        Returns:
            Dictionary with cache statistics and file information
        """
        import os

        stats = self.get_cache_stats()

        # Add file information
        if os.path.exists(self.cache_location):
            file_stat = os.stat(self.cache_location)
            stats["file_info"] = {
                "exists": True,
                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                "modified_time": file_stat.st_mtime,
            }
        else:
            stats["file_info"] = {"exists": False, "size_mb": 0, "modified_time": None}

        stats["cache_type"] = "SQLite (MCP dedicated)"

        logger.info(f"MCP Cache info: {stats}")
        return stats


# Global cache instance
_global_cache: Optional[MCPCache] = None


def get_mcp_cache() -> MCPCache:
    """Get global MCP cache instance.

    Returns:
        MCP cache instance with inheritance-based design
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = MCPCache()
    return _global_cache


def clear_mcp_cache_by_file() -> bool:
    """Clear all MCP cache by deleting SQLite cache file.

    This only affects MCP cache, not LiteLLM cache.

    Returns:
        True if successful, False otherwise
    """
    import os

    # Use the same path as configured in MCP settings
    mcp_settings = get_mcp_global_settings()
    cache_file = mcp_settings.cache_path

    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            logger.info(f"Successfully deleted MCP cache file: {cache_file}")

            # Reset global cache instance
            global _global_cache
            _global_cache = None

            return True
        except Exception as e:
            logger.error(f"Failed to delete MCP cache file {cache_file}: {e}")
            return False
    else:
        logger.info(f"MCP cache file does not exist: {cache_file}")
        return True


def get_cache_file_info() -> Dict[str, Any]:
    """Get cache file information.

    Returns:
        Dictionary with file existence, size, and modification details
    """
    return get_mcp_cache().get_cache_info()["file_info"]

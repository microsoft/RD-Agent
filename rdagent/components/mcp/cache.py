"""MCP Cache - Simplified Query Caching

This module provides basic query caching functionality for MCP services.
All unused tools caching and management functions have been removed.
"""

import hashlib
import os
from typing import Any, Dict, Optional

from rdagent.components.mcp.conf import get_mcp_global_settings
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import SQliteLazyCache


class MCPCache:
    """
    Simplified MCP cache for query results only.

    This class provides basic caching functionality for MCP query results
    to improve performance for repeated queries.
    """

    def __init__(self, namespace: str = "mcp"):
        """
        Initialize MCP cache with namespace.

        Args:
            namespace: Cache namespace for isolation
        """
        self.namespace = namespace
        self.settings = get_mcp_global_settings()

        # Initialize cache backend
        if self.settings.cache_enabled:
            # Use the configured cache path directly (namespace is in keys, not filename)
            cache_file = self.settings.cache_path
            # Ensure parent directory exists
            from pathlib import Path

            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            # Initialize SQLite cache (note: TTL is not supported by SQliteLazyCache)
            # SingletonBaseClass requires keyword arguments
            self._cache = SQliteLazyCache(cache_location=cache_file)
            logger.info(f"MCP cache enabled: {cache_file}")
        else:
            self._cache = None
            logger.info("MCP cache disabled")

    def _make_namespaced_key(self, key: str) -> str:
        """Create a namespaced cache key."""
        return f"{self.namespace}:{key}"

    def _make_query_key(self, error_message: str) -> str:
        """
        Generate a consistent cache key for query results.

        Args:
            error_message: The error message to cache results for

        Returns:
            Consistent cache key string
        """
        # Normalize the error message for consistent caching
        normalized = error_message.strip().lower()
        key_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return self._make_namespaced_key(f"query:{key_hash}")

    def get_query_result(self, error_message: str) -> Optional[str]:
        """
        Get cached query result.

        Args:
            error_message: The error message to look up

        Returns:
            Cached result string, or None if not found
        """
        if not self._cache:
            return None

        try:
            key = self._make_query_key(error_message)
            # Fix: SQliteLazyCache uses chat_get, not get
            return self._cache.chat_get(key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def set_query_result(self, error_message: str, result: str) -> None:
        """
        Cache query result.

        Args:
            error_message: The error message key
            result: The result to cache
        """
        if not self._cache or not result:
            return

        try:
            key = self._make_query_key(error_message)
            # Fix: SQliteLazyCache uses chat_set, not set
            self._cache.chat_set(key, result)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get basic cache information."""
        if not self._cache:
            return {"enabled": False, "backend": None}

        try:
            # Fix: SQliteLazyCache uses cache_location, not db_path
            cache_file = getattr(self._cache, "cache_location", "unknown")
            file_info = {
                "path": str(cache_file),
                "exists": os.path.exists(cache_file),
                "size": os.path.getsize(cache_file) if os.path.exists(cache_file) else 0,
            }

            return {
                "enabled": True,
                "backend": "SQLite",
                "namespace": self.namespace,
                "file_info": file_info,
            }
        except Exception as e:
            logger.warning(f"Failed to get cache info: {e}")
            return {"enabled": True, "error": str(e)}



# Global cache instance
_global_cache: Optional[MCPCache] = None


def get_mcp_cache() -> MCPCache:
    """
    Get or create the global MCP cache instance.

    Returns:
        Global MCPCache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = MCPCache()

    return _global_cache

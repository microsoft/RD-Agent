"""MCP缓存管理模块.

提供通用的缓存功能，用于MCP工具和查询结果的缓存管理。
复用RD-Agent现有的SQLite缓存系统，使用永久缓存策略。
"""

import hashlib
from typing import Any, Dict, Optional

from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import SQliteLazyCache
from rdagent.oai.llm_conf import LLM_SETTINGS


class MCPCache:
    """MCP缓存管理器，基于现有的SQLite缓存系统.
    
    使用永久缓存策略，与LITELLM保持一致。
    """
    
    def __init__(self):
        """初始化缓存管理器.
        
        使用永久缓存，不设置过期时间。
        """
        self._cache = SQliteLazyCache(cache_location=LLM_SETTINGS.prompt_cache_path)
        self._stats = {
            "tools_hits": 0,
            "tools_misses": 0,
            "query_hits": 0,
            "query_misses": 0
        }
    
    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """从SQLite缓存获取结果."""
        return self._cache.chat_get(cache_key)
    
    def _set_cached_result(self, cache_key: str, result: str):
        """设置SQLite缓存结果."""
        self._cache.chat_set(cache_key, result)
    
    def get_tools(self, mcp_url: str) -> Optional[Any]:
        """获取缓存的工具.
        
        Args:
            mcp_url: MCP服务URL
            
        Returns:
            缓存的工具列表，如果未命中则返回None
        """
        # 工具对象序列化复杂，暂时不实现工具缓存
        self._stats["tools_misses"] += 1
        logger.info(f"Tools cache miss for URL: {mcp_url} (tools caching disabled)")
        return None
    
    def set_tools(self, mcp_url: str, tools: Any):
        """设置工具缓存.
        
        Args:
            mcp_url: MCP服务URL
            tools: 要缓存的工具列表（当前未使用）
        """
        # 暂时不缓存工具对象，因为它们包含复杂的对象难以序列化
        logger.info(f"Tools caching skipped for URL: {mcp_url} (complex objects)")
    
    def get_query_result(self, error_message: str) -> Optional[str]:
        """获取缓存的查询结果.
        
        Args:
            error_message: 错误消息
            
        Returns:
            缓存的查询结果，如果未命中则返回None
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
        """设置查询结果缓存.
        
        Args:
            error_message: 错误消息
            result: 查询结果
        """
        cache_key = f"mcp_query:{hashlib.md5(error_message.encode('utf-8')).hexdigest()}"
        self._set_cached_result(cache_key, result)
        logger.info(f"Query result cached for key: {cache_key[-8:]}...")
    
    def clear_cache(self):
        """清空所有MCP缓存."""
        cleared_count = 0
        
        # 清理所有以mcp_前缀的缓存键
        # 注意：这需要遍历整个数据库，性能可能较差
        logger.warning("Clearing all MCP cache entries...")
        
        # 由于SQLite接口限制，我们无法直接遍历键，所以给出提示
        logger.info("To completely clear MCP cache, please delete the SQLite cache file or use clear_mcp_cache_by_pattern()")
        
        return cleared_count
    
    def clear_query_cache(self, error_message: str = None):
        """清空查询缓存.
        
        Args:
            error_message: 如果指定，只清空特定错误消息的缓存；否则清空所有查询缓存
        """
        if error_message:
            # 清空特定查询的缓存
            cache_key = f"mcp_query:{hashlib.md5(error_message.encode('utf-8')).hexdigest()}"
            # SQLite没有直接的删除方法，我们设置为None来"删除"
            self._set_cached_result(cache_key, "")  # 设置为空字符串表示已删除
            logger.info(f"Cleared cache for specific query: {cache_key[-8:]}...")
        else:
            logger.info("To clear all query cache, please use clear_all_mcp_cache() or delete the cache file")
    
    def get_cache_info(self):
        """获取缓存信息."""
        stats = self.get_cache_stats()
        cache_file = getattr(self._cache, 'cache_location', 'unknown')
        
        info = {
            "cache_file": cache_file,
            "stats": stats,
            "cache_type": "SQLite (shared with LITELLM)"
        }
        
        logger.info(f"Cache info: {info}")
        return info
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息."""
        total_tools = self._stats["tools_hits"] + self._stats["tools_misses"]
        total_queries = self._stats["query_hits"] + self._stats["query_misses"]
        
        return {
            "tools_cache": {
                "hits": self._stats["tools_hits"],
                "misses": self._stats["tools_misses"],
                "hit_rate": self._stats["tools_hits"] / max(total_tools, 1),
                "size": "N/A (SQLite)"
            },
            "query_cache": {
                "hits": self._stats["query_hits"],
                "misses": self._stats["query_misses"],
                "hit_rate": self._stats["query_hits"] / max(total_queries, 1),
                "size": "N/A (SQLite)"
            }
        }
    
    def log_cache_stats(self):
        """记录缓存统计信息到日志."""
        stats = self.get_cache_stats()
        logger.info(f"Cache stats - Tools: {stats['tools_cache']['hits']}/{stats['tools_cache']['hits'] + stats['tools_cache']['misses']} hits "
                   f"({stats['tools_cache']['hit_rate']:.2%}), "
                   f"Queries: {stats['query_cache']['hits']}/{stats['query_cache']['hits'] + stats['query_cache']['misses']} hits "
                   f"({stats['query_cache']['hit_rate']:.2%})")


# 全局缓存实例
_global_cache: Optional[MCPCache] = None


def get_mcp_cache() -> MCPCache:
    """获取全局MCP缓存实例.
    
    Returns:
        MCP缓存实例（永久缓存）
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = MCPCache()
    return _global_cache


def clear_mcp_cache_by_file():
    """通过删除SQLite缓存文件来清空所有缓存.
    
    注意：这会清空所有缓存，包括LITELLM的缓存！
    """
    import os
    from rdagent.oai.llm_conf import LLM_SETTINGS
    
    cache_file = LLM_SETTINGS.prompt_cache_path
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            logger.info(f"Successfully deleted cache file: {cache_file}")
            
            # 重置全局缓存实例
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
    """获取缓存文件信息."""
    import os
    from rdagent.oai.llm_conf import LLM_SETTINGS
    
    cache_file = LLM_SETTINGS.prompt_cache_path
    
    if os.path.exists(cache_file):
        stat = os.stat(cache_file)
        size_mb = stat.st_size / (1024 * 1024)
        
        info = {
            "file_path": cache_file,
            "exists": True,
            "size_mb": round(size_mb, 2),
            "modified_time": stat.st_mtime
        }
    else:
        info = {
            "file_path": cache_file,
            "exists": False,
            "size_mb": 0,
            "modified_time": None
        }
    
    logger.info(f"Cache file info: {info}")
    return info
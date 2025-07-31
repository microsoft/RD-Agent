"""Context7 MCP configuration using pydantic BaseSettings.

This module provides clean configuration management for Context7 MCP integration.
"""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Context7Settings(BaseSettings):
    """Context7 MCP configuration settings.
    
    All settings can be configured via environment variables with CONTEXT7_ prefix.
    """
    
    # MCP服务配置
    mcp_url: str = Field(
        default="http://localhost:8123/mcp",
        description="MCP service URL for Context7"
    )
    
    # LLM配置
    model: str = Field(
        default="gpt-4-turbo",
        description="LLM model name"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="OpenAI API base URL"
    )
    
    # 缓存配置
    cache_enabled: bool = Field(
        default=True,
        description="Enable MCP caching (permanent cache)"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="CONTEXT7_",
        extra="ignore",
    )
    
    def model_post_init(self, __context):
        """Post-initialization fallback to common environment variables."""
        # 简单的环境变量回退机制
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.api_base is None:
            self.api_base = os.getenv("OPENAI_API_BASE")


# 全局配置实例
context7_settings = Context7Settings()


def get_context7_settings() -> Context7Settings:
    """Get the global Context7 settings instance."""
    return context7_settings

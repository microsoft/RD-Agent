"""MCP (Model Context Protocol) global configuration.

This module provides global configuration for the entire MCP system.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPGlobalSettings(BaseSettings):
    """Global MCP system configuration.

    Simple configuration with just two essential controls:
    - System-wide enable/disable switch
    - Global cache enable/disable
    """

    # Global MCP control - this is the main switch
    enabled: bool = Field(default=True, description="Enable/disable entire MCP system")

    # Global cache control
    cache_enabled: bool = Field(default=False, description="Enable/disable MCP caching system-wide")

    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        extra="ignore",
    )


# Global instance
mcp_global_settings = MCPGlobalSettings()


def get_mcp_global_settings() -> MCPGlobalSettings:
    """Get the global MCP settings instance."""
    return mcp_global_settings


def is_mcp_enabled() -> bool:
    """Check if MCP system is globally enabled."""
    return mcp_global_settings.enabled

"""Utility functions for Context7 MCP configuration.

This module provides configuration loading functions for Context7 MCP integration,
using existing LITELLM_SETTINGS configuration without creating new prefixes.
"""

import os
from typing import Dict, Optional

from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.litellm import LITELLM_SETTINGS
from rdagent.oai.llm_conf import LLM_SETTINGS


def load_mcp_config() -> Dict[str, Optional[str]]:
    """Load MCP configuration using existing LLM settings.

    Returns:
        Dict containing model, api_key, api_base, and mcp_url configuration
    """
    try:
        # Priority 1: Use LLM_SETTINGS as base configuration with fallback to None
        selected_config = {
            "model": getattr(LLM_SETTINGS, "chat_model", None),
            "api_key": getattr(LLM_SETTINGS, "openai_api_key", None),
            "api_base": os.getenv("OPENAI_API_BASE"),
            "mcp_url": os.getenv("CONTEXT7_MCP_URL", "http://localhost:8123/mcp"),
        }

        # Priority 2: Use LITELLM_SETTINGS only if LLM_SETTINGS values are None
        if selected_config["model"] is None and hasattr(LITELLM_SETTINGS, "chat_model") and LITELLM_SETTINGS.chat_model:
            selected_config["model"] = LITELLM_SETTINGS.chat_model

        if (
            selected_config["api_key"] is None
            and hasattr(LITELLM_SETTINGS, "openai_api_key")
            and LITELLM_SETTINGS.openai_api_key
        ):
            selected_config["api_key"] = LITELLM_SETTINGS.openai_api_key

        # Extract specific model for 'coding' or 'running' tags directly
        if LITELLM_SETTINGS.chat_model_map:
            # Directly extract 'coding' model if available
            if "coding" in LITELLM_SETTINGS.chat_model_map:
                coding_config = LITELLM_SETTINGS.chat_model_map["coding"]
                selected_config["model"] = coding_config["model"]
                # Extract API key from coding config if available
                if "api_key" in coding_config:
                    selected_config["api_key"] = coding_config["api_key"]

            # Fallback to 'running' model if 'coding' not available
            elif "running" in LITELLM_SETTINGS.chat_model_map:
                running_config = LITELLM_SETTINGS.chat_model_map["running"]
                selected_config["model"] = running_config["model"]
                # Extract API key from running config if available
                if "api_key" in running_config:
                    selected_config["api_key"] = running_config["api_key"]

        # If still no API key, try to get from environment variables
        if not selected_config["api_key"]:
            selected_config["api_key"] = os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")

        return selected_config

    except ImportError as e:
        logger.error(f"Error importing LLM_SETTINGS: {e}")
        return {"model": None, "api_key": None, "api_base": None, "mcp_url": None}
    except Exception as e:
        logger.error(f"Error accessing LLM_SETTINGS: {e}")
        return {"model": None, "api_key": None, "api_base": None, "mcp_url": None}

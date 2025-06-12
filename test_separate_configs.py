#!/usr/bin/env python3
"""
Test script: Verify RD-Agent's chat and embedding API configurations
Supports both universal configuration (same API) and separate configuration (different APIs)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure correct import path
sys.path.insert(0, str(Path(__file__).parent))

PROMPT = "INTRODUCE YOURSELF"

try:
    from rdagent.log import rdagent_logger as logger
    from rdagent.oai.backend.litellm import LITELLM_SETTINGS
    from rdagent.oai.llm_utils import get_api_backend
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you are running this script from the RD-Agent project root directory")
    exit(1)


def print_config_info():
    """Print current configuration information"""
    print("=== Current Configuration ===")
    print(f"Backend type: {LITELLM_SETTINGS.backend}")
    print(f"Chat model: {LITELLM_SETTINGS.chat_model}")
    print(
        f"Chat API Key: {LITELLM_SETTINGS.chat_openai_api_key[:20]}..."
        if LITELLM_SETTINGS.chat_openai_api_key
        else "Not set"
    )
    print(f"Chat Base URL: {LITELLM_SETTINGS.chat_openai_base_url or 'Not set'}")
    print(f"Embedding model: {LITELLM_SETTINGS.embedding_model}")
    print(
        f"Embedding API Key: {LITELLM_SETTINGS.embedding_openai_api_key[:20]}..."
        if LITELLM_SETTINGS.embedding_openai_api_key
        else "Not set"
    )
    print(f"Embedding Base URL: {LITELLM_SETTINGS.embedding_openai_base_url or 'Not set'}")
    print(
        f"Universal API Key: {LITELLM_SETTINGS.openai_api_key[:20]}..."
        if LITELLM_SETTINGS.openai_api_key
        else "Not set"
    )
    print()


def test_chat():
    """Test chat functionality"""
    print("=== Testing Chat Functionality ===")
    try:
        api_backend = get_api_backend()
        response = api_backend.build_messages_and_create_chat_completion(user_prompt=PROMPT)
        print("✅ Chat test successful!")
        print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
        print()
        return True
    except Exception as e:
        print(f"❌ Chat test failed: {str(e)}")
        print()
        return False


def test_embedding():
    """Test embedding functionality"""
    print("=== Testing Embedding Functionality ===")
    try:
        api_backend = get_api_backend()
        test_text = PROMPT
        embedding = api_backend.create_embedding(test_text)
        print("✅ Embedding test successful!")
        print(f"Vector dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print()
        return True
    except Exception as e:
        print(f"❌ Embedding test failed: {str(e)}")
        print()
        return False


def analyze_config_mode():
    """Analyze current configuration mode"""
    print("=== Configuration Mode Analysis ===")

    # Get actual configuration in use
    api_backend = get_api_backend()
    chat_config = api_backend._get_chat_api_config()
    embedding_config = api_backend._get_embedding_api_config()

    chat_api_key = chat_config.get("api_key", "Not configured")
    chat_base_url = chat_config.get("api_base", "Default")
    embedding_api_key = embedding_config.get("api_key", "Not configured")
    embedding_base_url = embedding_config.get("api_base", "Default")

    print("Actual configuration in use:")
    print(
        f"  Chat API Key: {chat_api_key[:20]}..."
        if chat_api_key != "Not configured"
        else "  Chat API Key: Not configured"
    )
    print(f"  Chat Base URL: {chat_base_url}")
    print(
        f"  Embedding API Key: {embedding_api_key[:20]}..."
        if embedding_api_key != "Not configured"
        else "  Embedding API Key: Not configured"
    )
    print(f"  Embedding Base URL: {embedding_base_url}")
    print()

    # Determine configuration mode
    same_api_key = chat_api_key == embedding_api_key
    same_base_url = chat_base_url == embedding_base_url

    if same_api_key and same_base_url:
        print("🔄 Configuration Mode: **Universal Configuration**")
        print("   ✅ Chat and Embedding use the same API key and endpoint")
        print("   📝 This is a simplified configuration suitable for single API service scenarios")
        config_mode = "universal"
    elif not same_api_key or not same_base_url:
        print("🔀 Configuration Mode: **Separate Configuration**")
        print("   ✅ Chat and Embedding use different API configurations")
        print("   📝 This allows using different API services for different functionalities")
        if not same_api_key:
            print("   🔑 API Keys: Different")
        if not same_base_url:
            print("   🌐 Base URLs: Different")
        config_mode = "separated"

    # Check configuration source
    print("\nConfiguration Source Analysis:")
    if LITELLM_SETTINGS.chat_openai_api_key:
        print("   Chat: Using dedicated configuration (CHAT_OPENAI_API_KEY)")
    elif LITELLM_SETTINGS.openai_api_key:
        print("   Chat: Using universal configuration (OPENAI_API_KEY)")
    else:
        print("   Chat: No API key configured")

    if LITELLM_SETTINGS.embedding_openai_api_key:
        print("   Embedding: Using dedicated configuration (EMBEDDING_OPENAI_API_KEY)")
    elif LITELLM_SETTINGS.openai_api_key:
        print("   Embedding: Using universal configuration (OPENAI_API_KEY)")
    else:
        print("   Embedding: No API key configured")

    print()
    return config_mode


def print_config_recommendations():
    """Provide configuration recommendations based on current setup"""
    print("=== Configuration Recommendations ===")

    has_chat_key = bool(LITELLM_SETTINGS.chat_openai_api_key)
    has_embedding_key = bool(LITELLM_SETTINGS.embedding_openai_api_key)
    has_universal_key = bool(LITELLM_SETTINGS.openai_api_key)

    if has_chat_key and has_embedding_key:
        print("💡 You are using separate configuration mode:")
        print("   - Can use different API services for Chat and Embedding")
        print("   - Suitable for mixed deployment scenarios (e.g., mentor's LiteLLM + third-party embedding)")
    elif has_universal_key and not (has_chat_key or has_embedding_key):
        print("💡 You are using universal configuration mode:")
        print("   - Chat and Embedding share the same API service")
        print("   - Simple configuration, suitable for single service scenarios")
    elif has_universal_key and (has_chat_key or has_embedding_key):
        print("💡 You are using mixed configuration:")
        print("   - Some features use dedicated config, others use universal config")
        print("   - Recommendation: Either use all dedicated configs or all universal configs")
    else:
        print("⚠️  No valid API key configuration detected")
        print("    Please set OPENAI_API_KEY or dedicated CHAT_OPENAI_API_KEY/EMBEDDING_OPENAI_API_KEY")

    print()


def main():
    """Main test function"""
    print("🔍 Starting RD-Agent API Configuration Test...\n")

    # Print configuration information
    print_config_info()

    # Analyze configuration mode
    config_mode = analyze_config_mode()

    # Provide configuration recommendations
    print_config_recommendations()

    # Test chat functionality
    chat_success = test_chat()

    # Test embedding functionality
    embedding_success = test_embedding()

    # Summary
    print("=== Test Summary ===")
    print(f"Configuration Mode: {config_mode} ({'Universal' if config_mode == 'universal' else 'Separate'})")
    print(f"Chat Functionality: {'✅ Normal' if chat_success else '❌ Failed'}")
    print(f"Embedding Functionality: {'✅ Normal' if embedding_success else '❌ Failed'}")

    if chat_success and embedding_success:
        print(f"\n🎉 All tests passed! {config_mode} configuration is working properly.")
        if config_mode == "universal":
            print("📝 You are using universal configuration, Chat and Embedding share the same API service.")
        else:
            print("📝 You are using separate configuration, Chat and Embedding use different API services.")
        return 0
    else:
        print("\n⚠️  Some tests failed, please check your configuration or network connection.")
        return 1


if __name__ == "__main__":
    exit(main())

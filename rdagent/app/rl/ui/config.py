"""
RL UI Configuration Constants
"""

from typing import Literal

# Event type definition
EventType = Literal[
    "scenario",
    "llm_call",
    "template",
    "experiment",
    "code",
    "docker_exec",
    "feedback",
    "token",
    "time",
    "settings",
    "hypothesis",
]

# Event type icons
ICONS = {
    "scenario": "🎯",
    "llm_call": "💬",
    "template": "📋",
    "experiment": "🧪",
    "code": "📄",
    "docker_exec": "🐳",
    "feedback": "📊",
    "token": "🔢",
    "time": "⏱️",
    "settings": "⚙️",
    "hypothesis": "💡",
}

# Always visible event types
ALWAYS_VISIBLE_TYPES = [
    "scenario",
    "hypothesis",
    "llm_call",
    "experiment",
    "code",
    "docker_exec",
    "feedback",
]

# Optional event types with toggle config (label, default_enabled)
OPTIONAL_TYPES = {
    "template": ("📋 Template", False),
    "token": ("🔢 Token", False),
    "time": ("⏱️ Time", False),
    "settings": ("⚙️ Settings", False),
}


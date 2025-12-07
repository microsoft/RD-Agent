"""
FT UI Configuration Constants

Centralized configuration for FT Timeline Viewer.
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
    "evaluator",  # Evaluator feedback (separate from docker_exec)
    "feedback",
    "token",
    "time",
    "settings",
    "hypothesis",
    "dataset_selection",
]

# Event type icons
ICONS = {
    "scenario": "🎯",
    "llm_call": "💬",
    "template": "📋",
    "experiment": "🧪",
    "code": "📄",
    "docker_exec": "🐳",
    "evaluator": "📝",  # Evaluator feedback icon
    "feedback": "📊",
    "token": "🔢",
    "time": "⏱️",
    "settings": "⚙️",
    "hypothesis": "💡",
    "dataset_selection": "📂",
}

# Evaluator configuration mapping (name, default_stage)
EVALUATOR_CONFIG = {
    "FTDataEvaluator": ("Data Processing", "coding"),
    "FTCoderEvaluator": ("Micro-batch Test", "coding"),
    "FTRunnerEvaluator": ("Full Train", "runner"),
}

# Always visible event types
ALWAYS_VISIBLE_TYPES = [
    "scenario",
    "dataset_selection",
    "llm_call",
    "experiment",
    "code",
    "docker_exec",
    "evaluator",
    "feedback",
]

# Optional event types with toggle config (label, default_enabled)
OPTIONAL_TYPES = {
    "template": ("📋 Template", False),
    "token": ("🔢 Token", False),
    "time": ("⏱️ Time", False),
    "settings": ("⚙️ Settings", False),
}

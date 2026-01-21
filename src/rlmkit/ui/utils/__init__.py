"""UI Utilities - Formatting, constants, helpers."""

from typing import Dict, Any

# Constants for UI
APP_TITLE = "RLMKit Chat Studio"
APP_DESCRIPTION = "Interactive chat comparing RLM with Direct LLM"
APP_VERSION = "2.0.0"

# Mode options
EXECUTION_MODES = {
    "rlm_only": {
        "label": "RLM Only",
        "description": "Use RLM with exploration",
        "badge": "🔍 Thorough",
    },
    "direct_only": {
        "label": "Direct LLM",
        "description": "Call LLM directly without exploration",
        "badge": "⚡ Fast",
    },
    "compare": {
        "label": "Compare Both",
        "description": "Run both and compare results",
        "badge": "📊 Detailed",
    },
}

# Color scheme
COLORS = {
    "rlm": "#3B82F6",  # Blue
    "direct": "#10B981",  # Green
    "comparison": "#A855F7",  # Purple
    "success": "#22C55E",  # Light green
    "error": "#EF4444",  # Red
    "warning": "#F59E0B",  # Amber
}

# Provider info
PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4",
        "pricing": {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        },
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "default_model": "claude-3-sonnet",
        "pricing": {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        },
    },
    "ollama": {
        "label": "Ollama (Local)",
        "models": ["llama2", "mistral", "neural-chat"],
        "default_model": "llama2",
        "pricing": {
            "llama2": {"input": 0.0, "output": 0.0},
            "mistral": {"input": 0.0, "output": 0.0},
            "neural-chat": {"input": 0.0, "output": 0.0},
        },
    },
    "lmstudio": {
        "label": "LM Studio (Local)",
        "models": ["local-model"],
        "default_model": "local-model",
        "pricing": {
            "local-model": {"input": 0.0, "output": 0.0},
        },
    },
}


def format_tokens(count: int) -> str:
    """Format token count as human-readable string."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def format_cost(cost_usd: float) -> str:
    """Format cost as human-readable string."""
    return f"${cost_usd:.4f}"


def format_time(seconds: float) -> str:
    """Format time duration as human-readable string."""
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    elif seconds >= 1:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds * 1000:.0f}ms"


def format_memory(mb: float) -> str:
    """Format memory usage as human-readable string."""
    if mb >= 1024:
        return f"{mb / 1024:.1f}GB"
    else:
        return f"{mb:.1f}MB"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format percentage with sign."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimal_places}f}%"


def format_percentage_savings(original: float, new: float) -> str:
    """Format savings as percentage."""
    if original == 0:
        return "N/A"
    savings = ((original - new) / original) * 100
    return format_percentage(savings)


__all__ = [
    "APP_TITLE",
    "APP_DESCRIPTION",
    "APP_VERSION",
    "EXECUTION_MODES",
    "COLORS",
    "PROVIDERS",
    "format_tokens",
    "format_cost",
    "format_time",
    "format_memory",
    "format_percentage",
    "format_percentage_savings",
]

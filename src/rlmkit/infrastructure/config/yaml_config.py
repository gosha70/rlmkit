"""YAML/JSON configuration loading adapter.

Wraps the existing ``rlmkit.config.RLMConfig`` loader and converts
to application-layer DTOs.
"""

from __future__ import annotations

from typing import Optional

from rlmkit.application.dto import RunConfigDTO
from rlmkit.config import RLMConfig


def load_config(config_path: Optional[str] = None) -> RunConfigDTO:
    """Load configuration from file and convert to a RunConfigDTO.

    Args:
        config_path: Path to YAML or JSON config file. If None, uses
            the default search paths defined by RLMConfig.

    Returns:
        RunConfigDTO populated from the configuration file.
    """
    cfg = RLMConfig.load(config_path)

    return RunConfigDTO(
        max_steps=cfg.execution.max_steps,
        max_time_seconds=cfg.execution.default_timeout,
        verbose=False,
        extra={
            "safe_mode": cfg.execution.default_safe_mode,
            "max_output_chars": cfg.execution.max_output_chars,
            "use_json_protocol": cfg.execution.use_json_protocol,
            "max_parse_retries": cfg.execution.max_parse_retries,
        },
    )

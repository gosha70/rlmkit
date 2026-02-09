"""Factory for creating sandbox instances based on configuration."""

from __future__ import annotations

from typing import Optional

from rlmkit.application.ports.sandbox_port import SandboxPort
from .local_sandbox import LocalSandboxAdapter


def create_sandbox(
    sandbox_type: str = "local",
    safe_mode: bool = False,
    allowed_imports: Optional[list[str]] = None,
    max_exec_time_s: float = 5.0,
    max_stdout_chars: int = 10000,
) -> SandboxPort:
    """Create a sandbox adapter based on the requested type.

    Args:
        sandbox_type: Type of sandbox ("local", "restricted", "docker").
        safe_mode: Enable restricted execution.
        allowed_imports: Modules allowed in safe mode.
        max_exec_time_s: Per-execution timeout.
        max_stdout_chars: Maximum stdout capture.

    Returns:
        A sandbox adapter implementing SandboxPort.

    Raises:
        ValueError: If sandbox_type is not supported.
    """
    if sandbox_type == "local":
        return LocalSandboxAdapter(
            safe_mode=safe_mode,
            allowed_imports=allowed_imports,
            max_exec_time_s=max_exec_time_s,
            max_stdout_chars=max_stdout_chars,
        )
    elif sandbox_type == "restricted":
        from .restricted_sandbox import RestrictedSandboxAdapter

        allowed = frozenset(allowed_imports) if allowed_imports else None
        return RestrictedSandboxAdapter(
            allowed_modules=allowed,
            max_stdout_chars=max_stdout_chars,
        )
    elif sandbox_type == "docker":
        from rlmkit.envs.sandbox import DockerExecutor

        if not DockerExecutor.is_available():
            raise ValueError(
                "Docker sandbox requires Docker to be installed and running."
            )
        raise ValueError(
            "Docker sandbox adapter not yet wrapped for SandboxPort. "
            "Use sandbox_type='restricted' for safe execution."
        )
    else:
        raise ValueError(
            f"Unknown sandbox type: {sandbox_type!r}. "
            "Supported: 'local', 'restricted', 'docker'."
        )

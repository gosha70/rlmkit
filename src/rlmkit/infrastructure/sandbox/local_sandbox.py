"""Local sandbox adapter: wraps the existing PyReplEnv to implement SandboxPort."""

from __future__ import annotations

from typing import Any, Optional

from rlmkit.application.dto import ExecutionResultDTO
from rlmkit.application.ports.sandbox_port import SandboxPort
from rlmkit.envs.pyrepl_env import PyReplEnv


class LocalSandboxAdapter:
    """Adapter that wraps :class:`PyReplEnv` to satisfy :class:`SandboxPort`.

    This is the default sandbox for development and testing. It executes
    code in-process with optional safety restrictions.

    Args:
        safe_mode: Enable restricted builtins and import controls.
        allowed_imports: Modules allowed in safe mode.
        max_exec_time_s: Per-execution timeout in seconds.
        max_stdout_chars: Maximum captured stdout characters.
    """

    def __init__(
        self,
        safe_mode: bool = False,
        allowed_imports: Optional[list[str]] = None,
        max_exec_time_s: float = 5.0,
        max_stdout_chars: int = 10000,
    ) -> None:
        self._env = PyReplEnv(
            safe_mode=safe_mode,
            allowed_imports=allowed_imports,
            max_exec_time_s=max_exec_time_s,
            max_stdout_chars=max_stdout_chars,
        )

    def execute(self, code: str) -> ExecutionResultDTO:
        """Execute Python code in the local REPL environment.

        Args:
            code: Python source code to execute.

        Returns:
            ExecutionResultDTO with stdout, stderr, exceptions, etc.
        """
        result = self._env.execute(code)
        return ExecutionResultDTO(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            exception=result.get("exception"),
            timeout=result.get("timeout", False),
            truncated=result.get("truncated", False),
        )

    def reset(self) -> None:
        """Reset the REPL environment to a clean state."""
        self._env.reset()

    def is_healthy(self) -> bool:
        """The local sandbox is always healthy if it exists."""
        return True

    def set_variable(self, name: str, value: Any) -> None:
        """Inject a variable into the REPL namespace.

        Args:
            name: Variable name.
            value: Variable value.
        """
        self._env.env_globals[name] = value
        # If setting content 'P', also rebind the navigation tools
        if name == "P" and isinstance(value, str):
            self._env.set_content(value)

    def get_variable(self, name: str) -> Optional[Any]:
        """Retrieve a variable from the REPL namespace.

        Args:
            name: Variable name.

        Returns:
            The variable's value, or None if not found.
        """
        return self._env.get_var(name)

"""Sandbox port: interface for code execution environments.

Any execution backend (local exec, RestrictedPython, Docker, etc.) must
implement this Protocol to be usable by the application use cases.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from rlmkit.application.dto import ExecutionResultDTO


@runtime_checkable
class SandboxPort(Protocol):
    """Protocol for sandbox execution adapters.

    Implementations provide isolated environments for running
    LLM-generated Python code.
    """

    def execute(self, code: str) -> ExecutionResultDTO:
        """Execute Python code and return the result.

        Args:
            code: Python source code to execute.

        Returns:
            ExecutionResultDTO with stdout, stderr, exceptions, etc.
        """
        ...

    def reset(self) -> None:
        """Reset the execution environment to a clean state."""
        ...

    def is_healthy(self) -> bool:
        """Check whether the sandbox is operational.

        Returns:
            True if the sandbox can execute code, False otherwise.
        """
        ...

    def set_variable(self, name: str, value: Any) -> None:
        """Inject a variable into the sandbox namespace.

        Args:
            name: Variable name.
            value: Variable value.
        """
        ...

    def get_variable(self, name: str) -> Optional[Any]:
        """Retrieve a variable from the sandbox namespace.

        Args:
            name: Variable name.

        Returns:
            The variable's value, or None if not found.
        """
        ...

"""Docker sandbox adapter wrapping DockerExecutor as a SandboxPort.

Provides the strongest isolation level by running LLM-generated code in
disposable Docker containers with resource limits and network isolation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rlmkit.application.dto import ExecutionResultDTO

logger = logging.getLogger(__name__)


class DockerSandboxAdapter:
    """SandboxPort adapter backed by Docker containers.

    Wraps :class:`rlmkit.envs.sandbox.DockerExecutor` to conform to the
    SandboxPort protocol used by the application layer.

    Args:
        image_name: Docker image to use for execution.
        memory_limit: Memory limit for the container (e.g. "512m").
        cpu_limit: CPU limit (e.g. "1" for 1 core).
        timeout: Execution timeout in seconds.
        network_mode: Docker network mode ("none" for full isolation).
    """

    def __init__(
        self,
        image_name: str = "rlmkit-sandbox",
        memory_limit: str = "512m",
        cpu_limit: str = "1",
        timeout: int = 30,
        network_mode: str = "none",
    ) -> None:
        from rlmkit.envs.sandbox import DockerExecutor

        self._executor = DockerExecutor(
            image_name=image_name,
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
            timeout=timeout,
            network_mode=network_mode,
            auto_build=True,
        )
        self._namespace: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # SandboxPort interface
    # ------------------------------------------------------------------

    def execute(self, code: str) -> ExecutionResultDTO:
        """Execute Python code inside a Docker container.

        Args:
            code: Python source code to execute.

        Returns:
            ExecutionResultDTO with captured output.
        """
        # Inject any namespace variables by prepending assignments
        preamble = ""
        if self._namespace:
            for name, value in self._namespace.items():
                preamble += f"{name} = {value!r}\n"
            code = preamble + code

        raw = self._executor.execute(code)

        stdout = raw.get("output", "") or ""
        error = raw.get("error", "") or ""
        success = raw.get("result", False)
        timed_out = "timed out" in error.lower() if error else False

        return ExecutionResultDTO(
            stdout=stdout,
            stderr=error if not success and not timed_out else "",
            exception=error if not success and not timed_out else None,
            timeout=timed_out,
        )

    def reset(self) -> None:
        """Reset the execution namespace."""
        self._namespace.clear()

    def is_healthy(self) -> bool:
        """Check whether Docker is available and the image exists.

        Returns:
            True if Docker is running and can execute code.
        """
        from rlmkit.envs.sandbox import DockerExecutor

        return DockerExecutor.is_available()

    def set_variable(self, name: str, value: Any) -> None:
        """Inject a variable into the sandbox namespace.

        Args:
            name: Variable name.
            value: Variable value (must be repr-able for Docker transfer).
        """
        self._namespace[name] = value

    def get_variable(self, name: str) -> Optional[Any]:
        """Retrieve a variable from the namespace.

        Note: Docker containers are ephemeral, so this only returns
        variables that were set via :meth:`set_variable`, not those
        created during execution.

        Args:
            name: Variable name.

        Returns:
            The variable's value, or None if not found.
        """
        return self._namespace.get(name)

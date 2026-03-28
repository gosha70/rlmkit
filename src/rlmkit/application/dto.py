"""Data Transfer Objects for crossing layer boundaries.

DTOs are simple dataclasses used to pass data between layers without
creating coupling to domain entities or infrastructure types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequestDTO:
    """Data needed to make an LLM completion request.

    Attributes:
        messages: Chat messages in [{role, content}] format.
        model: Model identifier string.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    messages: list[dict[str, str]]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


@dataclass
class LLMResponseDTO:
    """Data returned from an LLM completion.

    Attributes:
        content: Generated text.
        model: Model that produced the response.
        input_tokens: Input token count (from API, if available).
        output_tokens: Output token count (from API, if available).
        finish_reason: Why generation stopped.
    """

    content: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None


@dataclass
class ExecutionResultDTO:
    """Result from sandbox code execution.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        exception: Exception message, if any.
        timeout: Whether execution timed out.
        truncated: Whether output was truncated.
    """

    stdout: str = ""
    stderr: str = ""
    exception: str | None = None
    timeout: bool = False
    truncated: bool = False

    @property
    def success(self) -> bool:
        """Whether execution completed without error or timeout."""
        return self.exception is None and not self.timeout


@dataclass
class RunConfigDTO:
    """Configuration for a use-case run.

    Attributes:
        mode: Execution mode (direct, rag, rlm, auto, compare).
        provider: LLM provider name.
        model: Model name.
        api_key: Optional API key override.
        max_steps: Maximum execution steps.
        max_tokens: Maximum total tokens.
        max_cost: Maximum cost in USD.
        max_time_seconds: Maximum execution time.
        max_recursion_depth: Maximum recursion depth.
        verbose: Whether to emit progress output.
        extra: Additional configuration passed through.
    """

    mode: str = "auto"
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    max_steps: int = 16
    max_tokens: int | None = None
    max_cost: float | None = None
    max_time_seconds: float | None = None
    max_recursion_depth: int = 1
    verbose: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResultDTO:
    """Result from a use-case run.

    Attributes:
        answer: Final answer text.
        mode_used: Execution mode that was used.
        success: Whether execution succeeded.
        error: Error description if it failed.
        steps: Number of execution steps.
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens consumed.
        total_cost: Total cost in USD.
        elapsed_time: Wall-clock seconds.
        trace: Serialized execution trace.
        metadata: Additional result metadata.
    """

    answer: str = ""
    mode_used: str = ""
    success: bool = True
    error: str | None = None
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    elapsed_time: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens

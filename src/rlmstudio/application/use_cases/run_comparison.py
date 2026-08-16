"""Use case: run multiple strategies in parallel and compare results.

Executes the same query using different modes (direct, rlm, optionally rag)
concurrently and returns all results for comparison.
"""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from rlmstudio.application.dto import RunConfigDTO, RunResultDTO
from rlmstudio.application.ports.llm_port import LLMPort
from rlmstudio.application.ports.sandbox_port import SandboxPort
from rlmstudio.application.use_cases.run_direct import RunDirectUseCase
from rlmstudio.application.use_cases.run_rlm import RunRLMUseCase


@dataclass
class ComparisonResultDTO:
    """Results from running multiple strategies for comparison.

    Attributes:
        results: Mapping of mode name to its RunResultDTO.
        total_elapsed: Wall-clock time from first start to last finish.
    """

    results: dict[str, RunResultDTO] = field(default_factory=dict)
    total_elapsed: float = 0.0

    def get_result(self, mode: str) -> RunResultDTO | None:
        """Get result for a specific mode."""
        return self.results.get(mode)

    @property
    def modes_run(self) -> list[str]:
        """List of modes that were executed."""
        return list(self.results.keys())


class RunComparisonUseCase:
    """Orchestrates running multiple strategies in parallel and comparing them.

    Args:
        llm: LLM port adapter.
        sandbox: Sandbox port adapter (needed for RLM mode).
    """

    def __init__(self, llm: LLMPort, sandbox: SandboxPort) -> None:
        self._llm = llm
        self._sandbox = sandbox

    def _run_direct(
        self, content: str, query: str, config: RunConfigDTO
    ) -> tuple[str, RunResultDTO]:
        # Shallow-copy the adapter so each branch has an independent _active_model.
        return "direct", RunDirectUseCase(copy.copy(self._llm)).execute(content, query, config)

    def _run_rlm(self, content: str, query: str, config: RunConfigDTO) -> tuple[str, RunResultDTO]:
        # Shallow-copy the adapter so RLM's use_recursive_model() doesn't affect direct.
        return "rlm", RunRLMUseCase(copy.copy(self._llm), self._sandbox).execute(
            content, query, config
        )

    def execute(
        self,
        content: str,
        query: str,
        config: RunConfigDTO | None = None,
        modes: list[str] | None = None,
    ) -> ComparisonResultDTO:
        """Run the query using multiple modes in parallel threads and compare results.

        Args:
            content: Document text to analyze.
            query: User question.
            config: Optional run configuration.
            modes: List of modes to run. Defaults to ["direct", "rlm"].

        Returns:
            ComparisonResultDTO with results from each mode.
        """
        config = config or RunConfigDTO(mode="compare")
        if modes is None:
            modes = ["direct", "rlm"]
        start = time.time()

        runners: dict[str, Callable[[], tuple[str, RunResultDTO]]] = {
            "direct": lambda: self._run_direct(content, query, config),
            "rlm": lambda: self._run_rlm(content, query, config),
        }

        comparison = ComparisonResultDTO()

        active = [m for m in modes if m in runners]
        if not active:
            comparison.total_elapsed = time.time() - start
            return comparison

        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures: dict[Future[tuple[str, RunResultDTO]], str] = {
                executor.submit(runners[m]): m for m in active
            }
            for future in as_completed(futures):
                mode_label, result = future.result()
                comparison.results[mode_label] = result

        comparison.total_elapsed = time.time() - start
        return comparison

    async def execute_async(
        self,
        content: str,
        query: str,
        config: RunConfigDTO | None = None,
        modes: list[str] | None = None,
    ) -> ComparisonResultDTO:
        """Run the query using multiple modes concurrently and compare results.

        Uses asyncio.gather so strategies run in parallel rather than sequentially.

        Args:
            content: Document text to analyze.
            query: User question.
            config: Optional run configuration.
            modes: List of modes to run. Defaults to ["direct", "rlm"].

        Returns:
            ComparisonResultDTO with results from each mode.
        """
        config = config or RunConfigDTO(mode="compare")
        if modes is None:
            modes = ["direct", "rlm"]
        start = time.time()

        async def _run(mode: str) -> tuple[str, RunResultDTO]:
            if mode == "direct":
                # Shallow-copy so each branch has an independent _active_model.
                result = await RunDirectUseCase(copy.copy(self._llm)).execute_async(
                    content, query, config
                )
                return "direct", result
            elif mode == "rlm":
                # Shallow-copy so RLM's use_recursive_model() doesn't affect direct.
                result = await RunRLMUseCase(copy.copy(self._llm), self._sandbox).execute_async(
                    content, query, config
                )
                return "rlm", result
            else:
                raise ValueError(f"Unsupported comparison mode: {mode!r}")

        pairs = await asyncio.gather(*[_run(m) for m in modes if m in ("direct", "rlm")])

        comparison = ComparisonResultDTO()
        for mode_label, result in pairs:
            comparison.results[mode_label] = result
        comparison.total_elapsed = time.time() - start
        return comparison

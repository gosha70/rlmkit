"""Use case: run multiple strategies and compare results.

Executes the same query using different modes (direct, rlm, optionally rag)
and returns all results for comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rlmkit.application.dto import RunConfigDTO, RunResultDTO
from rlmkit.application.ports.llm_port import LLMPort
from rlmkit.application.ports.sandbox_port import SandboxPort
from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase


@dataclass
class ComparisonResultDTO:
    """Results from running multiple strategies for comparison.

    Attributes:
        results: Mapping of mode name to its RunResultDTO.
        total_elapsed: Total wall-clock time for the entire comparison.
    """

    results: Dict[str, RunResultDTO] = field(default_factory=dict)
    total_elapsed: float = 0.0

    def get_result(self, mode: str) -> Optional[RunResultDTO]:
        """Get result for a specific mode."""
        return self.results.get(mode)

    @property
    def modes_run(self) -> List[str]:
        """List of modes that were executed."""
        return list(self.results.keys())


class RunComparisonUseCase:
    """Orchestrates running multiple strategies and comparing them.

    Args:
        llm: LLM port adapter.
        sandbox: Sandbox port adapter (needed for RLM mode).
    """

    def __init__(self, llm: LLMPort, sandbox: SandboxPort) -> None:
        self._llm = llm
        self._sandbox = sandbox

    def execute(
        self,
        content: str,
        query: str,
        config: Optional[RunConfigDTO] = None,
        modes: Optional[List[str]] = None,
    ) -> ComparisonResultDTO:
        """Run the query using multiple modes and compare results.

        Args:
            content: Document text to analyze.
            query: User question.
            config: Optional run configuration.
            modes: List of modes to run. Defaults to ["direct", "rlm"].

        Returns:
            ComparisonResultDTO with results from each mode.
        """
        config = config or RunConfigDTO(mode="compare")
        modes = modes or ["direct", "rlm"]
        start = time.time()

        comparison = ComparisonResultDTO()

        for mode in modes:
            if mode == "direct":
                uc = RunDirectUseCase(self._llm)
                result = uc.execute(content, query, config)
                comparison.results["direct"] = result

            elif mode == "rlm":
                uc_rlm = RunRLMUseCase(self._llm, self._sandbox)
                result = uc_rlm.execute(content, query, config)
                comparison.results["rlm"] = result

        comparison.total_elapsed = time.time() - start
        return comparison

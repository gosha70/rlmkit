"""Use case: run a direct LLM query (no RLM exploration).

Sends the full content and query to the LLM in a single request.
This is the simplest execution mode and serves as a baseline.
"""

from __future__ import annotations

import time
from typing import Optional

from rlmkit.application.dto import LLMResponseDTO, RunConfigDTO, RunResultDTO
from rlmkit.application.ports.llm_port import LLMPort


class RunDirectUseCase:
    """Orchestrates a direct (single-call) LLM query.

    Args:
        llm: LLM port adapter for generating completions.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def execute(
        self,
        content: str,
        query: str,
        config: Optional[RunConfigDTO] = None,
    ) -> RunResultDTO:
        """Run a direct query against the LLM.

        Args:
            content: Document text to analyze.
            query: User question about the content.
            config: Optional run configuration.

        Returns:
            RunResultDTO with the LLM answer and metrics.
        """
        config = config or RunConfigDTO(mode="direct")
        start = time.time()

        system_prompt = (
            "You are a helpful assistant. Answer the user's question "
            "based on the provided content."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Content:\n{content}\n\nQuestion: {query}"},
        ]

        try:
            response: LLMResponseDTO = self._llm.complete(messages)
            elapsed = time.time() - start

            return RunResultDTO(
                answer=response.content,
                mode_used="direct",
                success=True,
                steps=0,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                elapsed_time=elapsed,
                trace=[{
                    "step": 0,
                    "role": "assistant",
                    "content": response.content,
                    "mode": "direct",
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                }],
            )
        except Exception as exc:
            elapsed = time.time() - start
            return RunResultDTO(
                answer="",
                mode_used="direct",
                success=False,
                error=str(exc),
                elapsed_time=elapsed,
            )

"""Use case: run a direct LLM query (no RLM exploration).

Sends the full content and query to the LLM in a single request.
This is the simplest execution mode and serves as a baseline.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from rlmkit.application.dto import LLMResponseDTO, RunConfigDTO, RunResultDTO
from rlmkit.application.ports.llm_port import LLMPort
from rlmkit.application.ports.event_port import ExecutionEventEmitter


class RunDirectUseCase:
    """Orchestrates a direct (single-call) LLM query.

    Args:
        llm: LLM port adapter for generating completions.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute cost from adapter pricing."""
        try:
            pricing = self._llm.get_pricing()
            input_cost = input_tokens * pricing.get("input_cost_per_1m", 0) / 1_000_000
            output_cost = output_tokens * pricing.get("output_cost_per_1m", 0) / 1_000_000
            return input_cost + output_cost
        except Exception:
            return 0.0

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

            total_cost = self._compute_cost(response.input_tokens, response.output_tokens)

            return RunResultDTO(
                answer=response.content,
                mode_used="direct",
                success=True,
                steps=0,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_cost=total_cost,
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

    async def execute_async(
        self,
        content: str,
        query: str,
        config: Optional[RunConfigDTO] = None,
        event_emitter: Optional[ExecutionEventEmitter] = None,
    ) -> RunResultDTO:
        """Async direct query with optional token streaming.

        If *event_emitter* is provided and the LLM adapter supports async
        streaming, tokens are emitted in real time.
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
            if event_emitter and hasattr(self._llm, "complete_stream_async"):
                collected: list[str] = []
                async for token in self._llm.complete_stream_async(messages):
                    collected.append(token)
                    await event_emitter.on_token(token)
                answer = "".join(collected)
                approx_input = max(1, sum(len(m["content"]) for m in messages) // 4)
                approx_output = max(1, len(answer) // 4)
                input_tokens = approx_input
                output_tokens = approx_output
            elif hasattr(self._llm, "complete_async"):
                response: LLMResponseDTO = await self._llm.complete_async(messages)
                answer = response.content
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens
            else:
                response = await asyncio.to_thread(self._llm.complete, messages)
                answer = response.content
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens

            elapsed = time.time() - start

            step_entry = {
                "step": 0,
                "role": "assistant",
                "content": answer,
                "mode": "direct",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

            total_cost = self._compute_cost(input_tokens, output_tokens)

            if event_emitter:
                await event_emitter.on_step(step_entry)
                await event_emitter.on_metrics({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": total_cost,
                    "steps": 0,
                    "elapsed_seconds": elapsed,
                })

            return RunResultDTO(
                answer=answer,
                mode_used="direct",
                success=True,
                steps=0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost=total_cost,
                elapsed_time=elapsed,
                trace=[step_entry],
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

# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Direct LLM strategy — single call, full context."""

import time

from rlmkit.core.budget import TokenUsage, estimate_tokens
from rlmkit.core.rlm import LLMClient
from rlmkit.prompts import get_mode_system_prompt

from .base import StrategyResult


class DirectStrategy:
    """Send the full content + query to the LLM in one shot."""

    def __init__(
        self,
        client: LLMClient,
        system_prompt: str | None = None,
    ):
        self.client = client
        self.system_prompt = system_prompt or get_mode_system_prompt("direct")

    @property
    def name(self) -> str:
        return "direct"

    def run(self, content: str, query: str) -> StrategyResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Content:\n{content}\n\nQuestion: {query}"},
        ]

        start = time.time()
        try:
            answer = self.client.complete(messages)
        except Exception as e:
            return StrategyResult(
                strategy="direct",
                answer="",
                success=False,
                error=str(e),
                elapsed_time=time.time() - start,
            )

        elapsed = time.time() - start

        tokens = TokenUsage()
        tokens.add_input(estimate_tokens(content + query + self.system_prompt))
        tokens.add_output(estimate_tokens(answer))

        return StrategyResult(
            strategy="direct",
            answer=answer,
            steps=1,
            tokens=tokens,
            elapsed_time=elapsed,
            trace=[{"step": 1, "role": "assistant", "content": answer, "mode": "direct"}],
        )

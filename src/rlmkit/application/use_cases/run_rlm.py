"""Use case: run an RLM (Recursive Language Model) execution.

Orchestrates the RLM loop: load content into a sandbox, let the LLM
write code to explore it, execute code, feed results back, repeat
until a final answer or budget exhaustion.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from rlmkit.domain.entities import BudgetConfig, BudgetState
from rlmkit.domain.exceptions import BudgetExceededError
from rlmkit.application.dto import (
    LLMResponseDTO,
    RunConfigDTO,
    RunResultDTO,
)
from rlmkit.application.ports.llm_port import LLMPort
from rlmkit.application.ports.sandbox_port import SandboxPort


class RunRLMUseCase:
    """Orchestrates RLM recursive exploration through ports.

    Args:
        llm: LLM port adapter.
        sandbox: Sandbox port adapter for code execution.
    """

    def __init__(self, llm: LLMPort, sandbox: SandboxPort) -> None:
        self._llm = llm
        self._sandbox = sandbox

    def execute(
        self,
        content: str,
        query: str,
        config: Optional[RunConfigDTO] = None,
    ) -> RunResultDTO:
        """Run the RLM exploration loop.

        Args:
            content: Large document text to analyze.
            query: User question about the content.
            config: Optional run configuration.

        Returns:
            RunResultDTO with the final answer and execution metrics.
        """
        config = config or RunConfigDTO(mode="rlm")
        start = time.time()

        budget_config = BudgetConfig(
            max_steps=config.max_steps,
            max_tokens=config.max_tokens,
            max_cost=config.max_cost,
            max_time_seconds=config.max_time_seconds,
            max_recursion_depth=config.max_recursion_depth,
        )
        budget_state = BudgetState()

        # Initialize sandbox with content
        self._sandbox.set_variable("P", content)

        # Build initial messages
        system_prompt = self._build_system_prompt(len(content))
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        trace: List[Dict[str, Any]] = []
        cumulative_input = 0
        cumulative_output = 0

        try:
            while budget_state.steps < (budget_config.max_steps or 16):
                budget_state.steps += 1
                budget_state.elapsed_seconds = time.time() - start

                if not budget_state.is_within(budget_config):
                    raise BudgetExceededError(
                        f"Budget exceeded at step {budget_state.steps}"
                    )

                # Call LLM
                response: LLMResponseDTO = self._llm.complete(messages)
                cumulative_input += response.input_tokens
                cumulative_output += response.output_tokens
                budget_state.input_tokens = cumulative_input
                budget_state.output_tokens = cumulative_output

                trace.append({
                    "step": budget_state.steps,
                    "role": "assistant",
                    "content": response.content,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                })

                text = response.content

                # Check for FINAL answer
                final = self._extract_final(text)
                if final is not None:
                    elapsed = time.time() - start
                    return RunResultDTO(
                        answer=final,
                        mode_used="rlm",
                        success=True,
                        steps=budget_state.steps,
                        input_tokens=cumulative_input,
                        output_tokens=cumulative_output,
                        elapsed_time=elapsed,
                        trace=trace,
                    )

                # Check for code to execute
                code = self._extract_code(text)
                if code:
                    exec_result = self._sandbox.execute(code)
                    formatted = self._format_execution(exec_result)

                    trace.append({
                        "step": budget_state.steps,
                        "role": "execution",
                        "content": formatted,
                    })

                    messages.append({"role": "assistant", "content": text})
                    messages.append({
                        "role": "user",
                        "content": f"Execution result:\n{formatted}",
                    })
                else:
                    # No code and no FINAL -- nudge the LLM
                    messages.append({"role": "assistant", "content": text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Please provide either:\n"
                            "1. Python code to execute (in a ```python code block), OR\n"
                            "2. A FINAL answer (using FINAL: prefix)"
                        ),
                    })

            # Max steps exhausted
            raise BudgetExceededError(
                f"Maximum steps ({budget_config.max_steps or 16}) exceeded "
                "without finding final answer"
            )

        except BudgetExceededError as exc:
            elapsed = time.time() - start
            return RunResultDTO(
                answer="",
                mode_used="rlm",
                success=False,
                error=str(exc),
                steps=budget_state.steps,
                input_tokens=cumulative_input,
                output_tokens=cumulative_output,
                elapsed_time=elapsed,
                trace=trace,
            )
        except Exception as exc:
            elapsed = time.time() - start
            return RunResultDTO(
                answer="",
                mode_used="rlm",
                success=False,
                error=str(exc),
                steps=budget_state.steps,
                input_tokens=cumulative_input,
                output_tokens=cumulative_output,
                elapsed_time=elapsed,
                trace=trace,
            )

    # -- Private helpers --

    @staticmethod
    def _extract_final(text: str) -> Optional[str]:
        """Extract FINAL: answer from LLM response."""
        import re
        match = re.search(r"^FINAL:\s*(.*)", text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        """Extract Python code block from LLM response."""
        import re
        # Try python-specific block first
        match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fall back to generic block
        match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _format_execution(result: Any) -> str:
        """Format an ExecutionResultDTO for inclusion in the next message."""
        parts = []
        if hasattr(result, "stdout") and result.stdout:
            parts.append(f"Output:\n{result.stdout.strip()}")
        if hasattr(result, "stderr") and result.stderr:
            parts.append(f"Errors:\n{result.stderr.strip()}")
        if hasattr(result, "exception") and result.exception:
            parts.append(f"Exception:\n{result.exception}")
        if hasattr(result, "timeout") and result.timeout:
            parts.append("Execution timed out")
        if not parts:
            parts.append("Code executed successfully (no output)")
        return "\n\n".join(parts)

    @staticmethod
    def _build_system_prompt(content_length: int) -> str:
        """Build the RLM system prompt."""
        return (
            "You are a Recursive Language Model (RLM) agent. "
            "A large document has been loaded into variable P "
            f"({content_length:,} characters). "
            "You have access to tools: peek(start, end), grep(pattern), "
            "chunk(size), select(ranges). "
            "Write Python code in ```python blocks to explore P and "
            "answer the user's question. "
            "When you have the answer, respond with FINAL: <answer>."
        )

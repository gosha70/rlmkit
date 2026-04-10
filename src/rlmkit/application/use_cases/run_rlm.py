"""Use case: run an RLM (Recursive Language Model) execution.

Orchestrates the RLM loop: load content into a sandbox, let the LLM
write code to explore it, execute code, feed results back, repeat
until a final answer or budget exhaustion.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rlmkit.application.dto import (
    LLMResponseDTO,
    RunConfigDTO,
    RunResultDTO,
)
from rlmkit.application.ports.event_port import ExecutionEventEmitter
from rlmkit.application.ports.llm_port import LLMPort
from rlmkit.application.ports.sandbox_port import SandboxPort
from rlmkit.domain.entities import BudgetConfig, BudgetState
from rlmkit.domain.exceptions import BudgetExceededError
from rlmkit.prompts import get_rlm_message

if TYPE_CHECKING:
    from rlmkit.core.parsing import ParsedResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AttachedFileSection:
    """Attached-file metadata parsed from the multi-file document index."""

    file_no: int
    name: str
    file_start: int
    content_start: int
    file_end_exclusive: int


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
        config: RunConfigDTO | None = None,
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
            # max_recursion_depth is enforced by the _make_subcall closure, not BudgetConfig,
            # to avoid is_within() failing at depth 0 when max_recursion_depth=0.
        )
        budget_state = BudgetState()

        # Initialize sandbox with content and bind subcall for recursive use
        subcall_usage: list[dict[str, Any]] = []
        # Mutable snapshot updated before each code execution so the subcall closure
        # can guard against multiple subcalls within one block jointly exceeding the budget.
        parent_budget_snapshot: dict[str, float] = {
            "cost": 0.0,
            "tokens": 0.0,
            "steps": 0.0,
            "time_start": start,  # fixed: the parent's wall-clock start time
        }
        self._sandbox.set_variable("P", content)
        self._sandbox.set_variable(
            "subcall", self._make_subcall(config, subcall_usage, parent_budget_snapshot)
        )

        # Build initial messages
        system_prompt = self._build_system_prompt(len(content))
        if config.system_prompt_extra:
            system_prompt += "\n\n" + config.system_prompt_extra
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        attached_files = self._extract_attached_files(content)
        enforce_multi_file_coverage = self._should_enforce_multi_file_coverage(
            query, attached_files
        )
        inspected_files: set[int] = set()

        trace: list[dict[str, Any]] = []
        trace_seq = 0  # sequential index across all trace entries (assistant + execution)
        cumulative_input = 0
        cumulative_output = 0
        cumulative_cost = 0.0
        step_start = start
        consecutive_no_progress = 0
        stall_limit = config.stall_limit
        last_plain_answer: str = ""  # circuit-breaker: best non-FINAL text seen so far
        last_assistant_text: str = ""  # fallback: last raw LLM response (including code steps)
        inspect_snapshots: list[str] = []  # rolling inspect summary for synthesis fallback
        last_execution_failed = False  # track if previous code execution had errors
        overflow_at_step: int | None = None  # set when context window is exceeded mid-loop
        recent_exec_hashes: list[int] = []  # last N inspect result hashes for repeat detection
        repeat_count = 0  # consecutive duplicate inspect results
        repeat_limit = config.repeat_limit
        nudge_sent = False  # soft convergence nudge (sent once)
        hard_nudge_sent = False  # stronger convergence nudge near the end
        post_coverage_stall = 0  # no-progress turns after all files inspected
        input_cost_per_1m, output_cost_per_1m = self._get_pricing(self._llm)

        self._emit_runtime_fingerprint(
            trace=trace,
            attached_files=attached_files,
            enforce_multi_file_coverage=enforce_multi_file_coverage,
            execution_path="sync",
        )

        # Use root model for the initial reasoning call
        if hasattr(self._llm, "use_root_model"):
            self._llm.use_root_model()

        try:
            while budget_state.steps < (budget_config.max_steps or 16):
                budget_state.steps += 1
                budget_state.elapsed_seconds = time.time() - start
                step_start = time.time()

                if not budget_state.is_within(budget_config):
                    raise BudgetExceededError(f"Budget exceeded at step {budget_state.steps}")

                # Switch to recursive model for exploration subcalls (step > 1)
                if budget_state.steps > 1 and hasattr(self._llm, "use_recursive_model"):
                    self._llm.use_recursive_model()

                # Soft convergence nudge at budget fraction.
                # Floor at step 2: with low max_steps (e.g. 3), int(3*0.6)=1
                # would fire on step 1 before the model has done anything useful.
                nudge_step = self._compute_nudge_step(
                    budget_config.max_steps,
                    config.nudge_at_fraction,
                    minimum_step=2,
                )
                if not nudge_sent and nudge_step is not None and budget_state.steps >= nudge_step:
                    messages.append(
                        {
                            "role": "user",
                            "content": get_rlm_message("soft_nudge").format(
                                steps_used=budget_state.steps,
                                max_steps=budget_config.max_steps,
                            ),
                        }
                    )
                    nudge_sent = True

                hard_nudge_step = self._compute_nudge_step(
                    budget_config.max_steps,
                    0.75,
                    minimum_step=4,
                )
                if (
                    not hard_nudge_sent
                    and hard_nudge_step is not None
                    and budget_state.steps >= hard_nudge_step
                    and budget_state.steps < (budget_config.max_steps or 0)
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": get_rlm_message("hard_nudge").format(
                                steps_used=budget_state.steps,
                                max_steps=budget_config.max_steps,
                            ),
                        }
                    )
                    hard_nudge_sent = True

                # On the last allowed step, nudge the model to produce a final
                # answer instead of another inspect call.
                is_last_step = (
                    budget_config.max_steps is not None
                    and budget_state.steps >= budget_config.max_steps
                )
                call_messages = (
                    messages
                    + [
                        {
                            "role": "user",
                            "content": get_rlm_message("force_final_nudge"),
                        }
                    ]
                    if is_last_step
                    else messages
                )

                # Call LLM — break to synthesis/warning fallback on context window overflow
                try:
                    response: LLMResponseDTO = self._llm.complete(call_messages)
                except Exception as llm_exc:
                    if self._is_context_overflow(llm_exc):
                        overflow_at_step = budget_state.steps
                        logger.warning(
                            "Context window exceeded at step %d; falling back to warning",
                            budget_state.steps,
                        )
                        break
                    raise
                cumulative_input += response.input_tokens
                cumulative_output += response.output_tokens
                budget_state.input_tokens = cumulative_input
                budget_state.output_tokens = cumulative_output
                cumulative_cost += (
                    response.input_tokens * input_cost_per_1m
                    + response.output_tokens * output_cost_per_1m
                ) / 1_000_000
                budget_state.cost = cumulative_cost

                # Re-check budget immediately after consuming tokens — a single
                # expensive LLM call can push cost past max_cost.
                if not budget_state.is_within(budget_config):
                    raise BudgetExceededError(f"Budget exceeded after step {budget_state.steps}")

                text = response.content

                # Parse response: try JSON v2.0 first, fall back to markdown v1.0
                parsed = self._parse_rlm_response(text)

                # Track non-code responses as a last-resort fallback answer.
                # Code-only responses are excluded: a bare code block is not a
                # useful answer to return when max_steps is exhausted.
                if text.strip() and not parsed.has_code:
                    last_assistant_text = text.strip()

                trace_seq += 1
                clamp_info = getattr(self._llm, "last_clamp_info", None) or {}
                trace_entry: dict[str, Any] = {
                    "step": budget_state.steps,
                    "seq": trace_seq,
                    "role": "assistant",
                    "content": response.content,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "code": parsed.code,
                    "model": getattr(self._llm, "active_model", None) or response.model,
                    "elapsed_seconds": time.time() - step_start,
                }
                if clamp_info:
                    trace_entry["clamp"] = clamp_info
                trace.append(trace_entry)

                # Check for FINAL answer (handles both FINAL: and FINAL_VAR:)
                if parsed.is_complete:
                    missing_files = self._get_missing_files(
                        attached_files, inspected_files, enforce_multi_file_coverage
                    )
                    if missing_files:
                        trace_seq, inspect_snapshots, auto_file_no = (
                            self._auto_inspect_missing_file(
                                content=content,
                                config=config,
                                messages=messages,
                                trace=trace,
                                trace_seq=trace_seq,
                                budget_state=budget_state,
                                inspect_snapshots=inspect_snapshots,
                                missing_files=missing_files,
                                assistant_text=text,
                            )
                        )
                        inspected_files.add(auto_file_no)
                        consecutive_no_progress = 0
                        continue
                    if last_execution_failed:
                        trace_seq += 1
                        trace.append(
                            {
                                "step": budget_state.steps,
                                "seq": trace_seq,
                                "role": "system",
                                "content": (
                                    "⚠️ Warning: FINAL provided after execution failure. "
                                    "Model may have used direct reasoning instead of code execution."
                                ),
                            }
                        )
                    # Restore root model for any subsequent use of the adapter
                    if hasattr(self._llm, "use_root_model"):
                        self._llm.use_root_model()
                    elapsed = time.time() - start
                    return RunResultDTO(
                        answer=self._extract_final_answer(parsed),
                        mode_used="rlm",
                        success=True,
                        steps=budget_state.steps,
                        input_tokens=cumulative_input,
                        output_tokens=cumulative_output,
                        total_cost=cumulative_cost,
                        elapsed_time=elapsed,
                        trace=trace,
                    )

                # Check for code to execute
                if parsed.has_code and parsed.code:
                    consecutive_no_progress = 0
                    if parsed.is_inspect:
                        inspected_files.update(
                            self._infer_inspected_files_from_code(parsed.code, attached_files)
                        )
                    # Snapshot current spend so the subcall closure can guard against
                    # multiple subcalls within one block jointly exceeding the budget.
                    parent_budget_snapshot["cost"] = cumulative_cost
                    parent_budget_snapshot["tokens"] = float(cumulative_input + cumulative_output)
                    parent_budget_snapshot["steps"] = float(budget_state.steps)
                    exec_result = self._sandbox.execute(parsed.code)
                    last_execution_failed = exec_result.exception is not None or exec_result.timeout
                    # Fold subcall token/cost/step usage back into parent accumulators
                    for _u in subcall_usage:
                        cumulative_input += _u.get("input_tokens", 0)
                        cumulative_output += _u.get("output_tokens", 0)
                        cumulative_cost += _u.get("total_cost", 0.0)
                        budget_state.steps += _u.get("steps", 0)
                    subcall_usage.clear()
                    budget_state.input_tokens = cumulative_input
                    budget_state.output_tokens = cumulative_output
                    budget_state.cost = cumulative_cost
                    formatted = self._format_execution(exec_result)
                    # Only track output from inspect actions (they all generate print() calls).
                    # Arbitrary Python code executions are not useful for synthesis fallback.
                    if parsed.is_inspect:
                        inspect_snapshots = self._append_inspect_snapshot(
                            inspect_snapshots,
                            step=budget_state.steps,
                            code=parsed.code,
                            execution_output=formatted,
                        )

                    trace_seq += 1
                    trace.append(
                        {
                            "step": budget_state.steps,
                            "seq": trace_seq,
                            "role": "execution",
                            "content": formatted,
                            "code": parsed.code,
                        }
                    )

                    # Repeat detection — only for inspect actions.
                    # Non-inspect code (subcalls, arbitrary computation) can legitimately
                    # produce identical output; only inspect-action results should feed
                    # the repeat detector.
                    repeat_nudge: str | None = None
                    if parsed.is_inspect:
                        exec_hash = hash(formatted[:500])
                        is_repeat = exec_hash in recent_exec_hashes
                        recent_exec_hashes.append(exec_hash)
                        if len(recent_exec_hashes) > 5:
                            recent_exec_hashes.pop(0)

                        if is_repeat:
                            repeat_count += 1
                            if repeat_count >= repeat_limit:
                                repeat_nudge = get_rlm_message("repeat_detected")
                            else:
                                repeat_nudge = get_rlm_message("duplicate_result")
                        else:
                            repeat_count = 0

                    messages.append({"role": "assistant", "content": text})
                    # Truncate large execution outputs before adding to message
                    # history.  Limit is recomputed per step to account for
                    # accumulated context usage.  We include overhead for the
                    # "Execution result:\n" prefix (~20 chars), truncation suffix
                    # (~50 chars), and any repeat nudge so the cap reflects the
                    # full payload about to be appended, not just the raw body.
                    _wrapper_overhead = 70 + (len(repeat_nudge) if repeat_nudge else 0)
                    current_msg_chars = (
                        sum(len(m.get("content", "")) for m in messages) + _wrapper_overhead
                    )
                    exec_result_limit = self._compute_exec_result_limit(
                        len(content),
                        config.max_steps,
                        current_message_chars=current_msg_chars,
                        model_context_limit=getattr(self._llm, "context_length_chars", None),
                        override=config.extra.get("exec_result_limit"),
                    )
                    exec_content = formatted
                    if len(exec_content) > exec_result_limit:
                        exec_content = (
                            exec_content[:exec_result_limit]
                            + f"\n… [truncated, {len(formatted) - exec_result_limit} chars omitted]"
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Execution result:\n{exec_content}",
                        }
                    )
                    # Append repeat-detection nudge AFTER the exec result so the
                    # model sees both the output (for learning) and the guidance.
                    if repeat_nudge is not None:
                        messages.append({"role": "user", "content": repeat_nudge})
                else:
                    # No code and no FINAL -- track the best plain-text response seen
                    stripped = text.strip()
                    missing_files = self._get_missing_files(
                        attached_files, inspected_files, enforce_multi_file_coverage
                    )
                    if missing_files:
                        trace_seq, inspect_snapshots, auto_file_no = (
                            self._auto_inspect_missing_file(
                                content=content,
                                config=config,
                                messages=messages,
                                trace=trace,
                                trace_seq=trace_seq,
                                budget_state=budget_state,
                                inspect_snapshots=inspect_snapshots,
                                missing_files=missing_files,
                                assistant_text=text,
                            )
                        )
                        inspected_files.add(auto_file_no)
                        consecutive_no_progress = 0
                        continue
                    if stripped and not missing_files:
                        last_plain_answer = stripped
                    consecutive_no_progress += 1

                    # Early post-coverage synthesis: when all required files
                    # have been inspected but the model keeps producing prose
                    # without a valid final, synthesize from inspect_snapshots
                    # after 2 stalled turns instead of waiting for full stall
                    # exhaustion.  This rescues weak models that cannot produce
                    # a JSON final but have already gathered enough evidence.
                    if not missing_files and enforce_multi_file_coverage:
                        post_coverage_stall += 1
                    else:
                        post_coverage_stall = 0

                    if (
                        post_coverage_stall >= 2
                        and inspect_snapshots
                        and enforce_multi_file_coverage
                    ):
                        logger.info(
                            "Early post-coverage synthesis at step %d "
                            "(model stalled %d turns after coverage complete)",
                            budget_state.steps,
                            post_coverage_stall,
                        )
                        trace_seq += 1
                        trace.append(
                            {
                                "step": budget_state.steps,
                                "seq": trace_seq,
                                "role": "system",
                                "content": (
                                    "Early synthesis: coverage complete but model "
                                    "did not finalize after 2 no-progress turns."
                                ),
                            }
                        )
                        if hasattr(self._llm, "use_root_model"):
                            self._llm.use_root_model()
                        synth_step_start = time.time()
                        synth_response = self._llm.complete(
                            self._build_synthesis_messages(query, inspect_snapshots)
                        )
                        budget_state.steps += 1
                        cumulative_input += synth_response.input_tokens
                        cumulative_output += synth_response.output_tokens
                        cumulative_cost += (
                            synth_response.input_tokens * input_cost_per_1m
                            + synth_response.output_tokens * output_cost_per_1m
                        ) / 1_000_000
                        trace_seq += 1
                        trace.append(
                            {
                                "step": budget_state.steps,
                                "seq": trace_seq,
                                "role": "assistant",
                                "content": synth_response.content,
                                "input_tokens": synth_response.input_tokens,
                                "output_tokens": synth_response.output_tokens,
                                "model": (
                                    getattr(self._llm, "active_model", None) or synth_response.model
                                ),
                                "elapsed_seconds": time.time() - synth_step_start,
                                "note": "early post-coverage synthesis",
                            }
                        )
                        answer = synth_response.content.strip() or (
                            "The content was inspected but contained no extractable themes."
                        )
                        if not self._looks_like_action(answer):
                            elapsed = time.time() - start
                            return RunResultDTO(
                                answer=answer,
                                mode_used="rlm",
                                success=True,
                                steps=budget_state.steps,
                                input_tokens=cumulative_input,
                                output_tokens=cumulative_output,
                                total_cost=cumulative_cost,
                                elapsed_time=elapsed,
                                trace=trace,
                            )

                    if consecutive_no_progress >= stall_limit:
                        # Circuit breaker: if the model wrote a real answer but skipped
                        # the FINAL: format, accept it rather than discarding it.
                        if (
                            last_plain_answer
                            and not self._looks_like_action(last_plain_answer)
                            and not missing_files
                            and not enforce_multi_file_coverage
                        ):
                            if hasattr(self._llm, "use_root_model"):
                                self._llm.use_root_model()
                            elapsed = time.time() - start
                            answer = self._extract_final(last_plain_answer) or last_plain_answer
                            return RunResultDTO(
                                answer=answer,
                                mode_used="rlm",
                                success=True,
                                steps=budget_state.steps,
                                input_tokens=cumulative_input,
                                output_tokens=cumulative_output,
                                total_cost=cumulative_cost,
                                elapsed_time=elapsed,
                                trace=trace,
                            )
                        raise BudgetExceededError(
                            f"LLM did not make progress after {consecutive_no_progress} "
                            "consecutive steps without code or a FINAL answer"
                        )
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": get_rlm_message("reprompt"),
                        }
                    )

            # Max steps exhausted — use best available text before failing.
            # Prefer an explicit plain-text answer; fall back to last raw LLM
            # response (covers models that always generate code but never FINAL:).
            best_fallback = last_plain_answer or last_assistant_text
            if self._get_missing_files(
                attached_files, inspected_files, enforce_multi_file_coverage
            ):
                best_fallback = ""
            # Don't return raw JSON action blobs as the final answer —
            # fall through to synthesis or limit-warning instead.
            if best_fallback and self._looks_like_action(best_fallback):
                best_fallback = ""
            if best_fallback:
                if hasattr(self._llm, "use_root_model"):
                    self._llm.use_root_model()
                elapsed = time.time() - start
                # Strip FINAL: prefix if the model included it in plain text
                # but the parser missed it (e.g. not at line start, bold markup).
                answer = self._extract_final(best_fallback) or best_fallback
                return RunResultDTO(
                    answer=answer,
                    mode_used="rlm",
                    success=True,
                    steps=budget_state.steps,
                    input_tokens=cumulative_input,
                    output_tokens=cumulative_output,
                    total_cost=cumulative_cost,
                    elapsed_time=elapsed,
                    trace=trace,
                )
            # Synthesis fallback: model exhausted steps without a final answer but
            # we have inspection output. Make one extra call (outside the step budget)
            # asking the model to summarize what it found.
            if inspect_snapshots:
                if hasattr(self._llm, "use_root_model"):
                    self._llm.use_root_model()
                # Honour non-step budgets before making the synthesis call.
                budget_state.elapsed_seconds = time.time() - start
                if not budget_state.is_within(budget_config):
                    raise BudgetExceededError("Budget exceeded before synthesis call")
                synth_step_start = time.time()
                synth_response = self._llm.complete(
                    self._build_synthesis_messages(query, inspect_snapshots)
                )
                budget_state.steps += 1
                cumulative_input += synth_response.input_tokens
                cumulative_output += synth_response.output_tokens
                cumulative_cost += (
                    synth_response.input_tokens * input_cost_per_1m
                    + synth_response.output_tokens * output_cost_per_1m
                ) / 1_000_000
                budget_state.input_tokens = cumulative_input
                budget_state.output_tokens = cumulative_output
                budget_state.cost = cumulative_cost
                budget_state.elapsed_seconds = time.time() - start
                # Synthesis is intentionally one step past max_steps, so only
                # re-check non-step budgets (cost, tokens, time) here.
                _non_step_config = BudgetConfig(
                    max_tokens=budget_config.max_tokens,
                    max_cost=budget_config.max_cost,
                    max_time_seconds=budget_config.max_time_seconds,
                )
                if not budget_state.is_within(_non_step_config):
                    raise BudgetExceededError("Budget exceeded after synthesis call")
                trace_seq += 1
                trace.append(
                    {
                        "step": budget_state.steps,
                        "seq": trace_seq,
                        "role": "assistant",
                        "content": synth_response.content,
                        "input_tokens": synth_response.input_tokens,
                        "output_tokens": synth_response.output_tokens,
                        "model": getattr(self._llm, "active_model", None) or synth_response.model,
                        "elapsed_seconds": time.time() - synth_step_start,
                        "note": "synthesis fallback",
                    }
                )
                answer = synth_response.content.strip() or (
                    "The content was inspected but contained no extractable themes."
                )
                # If the synthesis call itself returned a JSON action blob
                # (model stuck in action mode), fall through to the limit warning.
                if not self._looks_like_action(answer):
                    elapsed = time.time() - start
                    return RunResultDTO(
                        answer=answer,
                        mode_used="rlm",
                        success=True,
                        steps=budget_state.steps,
                        input_tokens=cumulative_input,
                        output_tokens=cumulative_output,
                        total_cost=cumulative_cost,
                        elapsed_time=elapsed,
                        trace=trace,
                    )
            # No answer available — return an actionable warning.
            # Treated as success=True so the result is visible in chat and
            # can be evaluated by Auto-Judge.
            if hasattr(self._llm, "use_root_model"):
                self._llm.use_root_model()
            elapsed = time.time() - start
            return RunResultDTO(
                answer=self._format_limit_warning(
                    overflow_step=overflow_at_step,
                    steps_used=budget_state.steps,
                    config=budget_config,
                ),
                mode_used="rlm",
                success=True,
                steps=budget_state.steps,
                input_tokens=cumulative_input,
                output_tokens=cumulative_output,
                total_cost=cumulative_cost,
                elapsed_time=elapsed,
                trace=trace,
            )

        except BudgetExceededError as exc:
            # Config/limit hit — return a warning, not an error, so the user
            # knows exactly which setting to change. success=True lets Auto-Judge run.
            if hasattr(self._llm, "use_root_model"):
                self._llm.use_root_model()
            elapsed = time.time() - start
            partial = last_plain_answer or last_assistant_text
            if enforce_multi_file_coverage or self._get_missing_files(
                attached_files, inspected_files, enforce_multi_file_coverage
            ):
                partial = ""
            if enforce_multi_file_coverage and inspect_snapshots:
                budget_state.elapsed_seconds = time.time() - start
                if budget_state.is_within(budget_config):
                    synth_step_start = time.time()
                    synth_response = self._llm.complete(
                        self._build_synthesis_messages(query, inspect_snapshots)
                    )
                    budget_state.steps += 1
                    cumulative_input += synth_response.input_tokens
                    cumulative_output += synth_response.output_tokens
                    cumulative_cost += (
                        synth_response.input_tokens * input_cost_per_1m
                        + synth_response.output_tokens * output_cost_per_1m
                    ) / 1_000_000
                    trace_seq += 1
                    trace.append(
                        {
                            "step": budget_state.steps,
                            "seq": trace_seq,
                            "role": "assistant",
                            "content": synth_response.content,
                            "input_tokens": synth_response.input_tokens,
                            "output_tokens": synth_response.output_tokens,
                            "model": getattr(self._llm, "active_model", None)
                            or synth_response.model,
                            "elapsed_seconds": time.time() - synth_step_start,
                            "note": "synthesis fallback",
                        }
                    )
                    answer = synth_response.content.strip() or (
                        "The content was inspected but contained no extractable themes."
                    )
                    prior_plain = (last_plain_answer or last_assistant_text).strip()
                    if not self._looks_like_action(answer) and answer != prior_plain:
                        return RunResultDTO(
                            answer=answer,
                            mode_used="rlm",
                            success=True,
                            steps=budget_state.steps,
                            input_tokens=cumulative_input,
                            output_tokens=cumulative_output,
                            total_cost=cumulative_cost,
                            elapsed_time=time.time() - start,
                            trace=trace,
                        )
            warning = self._format_limit_warning(
                overflow_step=overflow_at_step,
                steps_used=budget_state.steps,
                config=budget_config,
                exc=exc,
            )
            answer = (partial + "\n\n---\n\n" + warning) if partial else warning
            return RunResultDTO(
                answer=answer,
                mode_used="rlm",
                success=True,
                steps=budget_state.steps,
                input_tokens=cumulative_input,
                output_tokens=cumulative_output,
                total_cost=cumulative_cost,
                elapsed_time=elapsed,
                trace=trace,
            )
        except Exception as exc:
            if hasattr(self._llm, "use_root_model"):
                self._llm.use_root_model()
            elapsed = time.time() - start
            # Context window exceeded is a configuration issue — warn, don't error.
            if self._is_context_overflow(exc):
                partial = last_plain_answer or last_assistant_text
                if self._get_missing_files(
                    attached_files, inspected_files, enforce_multi_file_coverage
                ):
                    partial = ""
                warning = self._format_limit_warning(
                    overflow_step=budget_state.steps,
                    steps_used=budget_state.steps,
                    config=budget_config,
                )
                answer = (partial + "\n\n---\n\n" + warning) if partial else warning
                return RunResultDTO(
                    answer=answer,
                    mode_used="rlm",
                    success=True,
                    steps=budget_state.steps,
                    input_tokens=cumulative_input,
                    output_tokens=cumulative_output,
                    total_cost=cumulative_cost,
                    elapsed_time=elapsed,
                    trace=trace,
                )
            return RunResultDTO(
                answer="",
                mode_used="rlm",
                success=False,
                error=str(exc),
                steps=budget_state.steps,
                input_tokens=cumulative_input,
                output_tokens=cumulative_output,
                total_cost=cumulative_cost,
                elapsed_time=elapsed,
                trace=trace,
            )

    async def execute_async(
        self,
        content: str,
        query: str,
        config: RunConfigDTO | None = None,
        event_emitter: ExecutionEventEmitter | None = None,
    ) -> RunResultDTO:
        """Async RLM loop with real-time event streaming.

        If *event_emitter* is provided, emits ``on_token``, ``on_step``,
        and ``on_metrics`` events as execution progresses.  Falls back to
        the sync ``complete`` path when the LLM adapter lacks async methods.
        """
        config = config or RunConfigDTO(mode="rlm")
        start = time.time()

        budget_config = BudgetConfig(
            max_steps=config.max_steps,
            max_tokens=config.max_tokens,
            max_cost=config.max_cost,
            max_time_seconds=config.max_time_seconds,
            # max_recursion_depth is enforced by the _make_subcall closure, not BudgetConfig,
            # to avoid is_within() failing at depth 0 when max_recursion_depth=0.
        )
        budget_state = BudgetState()

        subcall_usage: list[dict[str, Any]] = []
        parent_budget_snapshot: dict[str, float] = {
            "cost": 0.0,
            "tokens": 0.0,
            "steps": 0.0,
            "time_start": start,  # fixed: the parent's wall-clock start time
        }
        self._sandbox.set_variable("P", content)
        self._sandbox.set_variable(
            "subcall", self._make_subcall(config, subcall_usage, parent_budget_snapshot)
        )

        system_prompt = self._build_system_prompt(len(content))
        if config.system_prompt_extra:
            system_prompt += "\n\n" + config.system_prompt_extra
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        attached_files = self._extract_attached_files(content)
        enforce_multi_file_coverage = self._should_enforce_multi_file_coverage(
            query, attached_files
        )
        inspected_files: set[int] = set()

        trace: list[dict[str, Any]] = []
        trace_seq = 0  # sequential index across all trace entries (assistant + execution)
        cumulative_input = 0
        cumulative_output = 0
        cumulative_cost = 0.0
        step_start = start
        consecutive_no_progress = 0
        stall_limit = config.stall_limit
        last_plain_answer: str = ""  # circuit-breaker: best non-FINAL text seen so far
        last_assistant_text: str = ""  # fallback: last raw LLM response (including code steps)
        inspect_snapshots: list[str] = []  # rolling inspect summary for synthesis fallback
        last_execution_failed = False  # track if previous code execution had errors
        overflow_at_step: int | None = None  # set when context window is exceeded mid-loop
        recent_exec_hashes: list[int] = []  # last N inspect result hashes for repeat detection
        repeat_count = 0  # consecutive duplicate inspect results
        repeat_limit = config.repeat_limit
        nudge_sent = False  # soft convergence nudge (sent once)
        hard_nudge_sent = False  # stronger convergence nudge near the end
        post_coverage_stall = 0  # no-progress turns after all files inspected
        input_cost_per_1m, output_cost_per_1m = self._get_pricing(self._llm)

        self._emit_runtime_fingerprint(
            trace=trace,
            attached_files=attached_files,
            enforce_multi_file_coverage=enforce_multi_file_coverage,
            execution_path="async",
        )

        if hasattr(self._llm, "use_root_model"):
            self._llm.use_root_model()

        try:
            while budget_state.steps < (budget_config.max_steps or 16):
                budget_state.steps += 1
                budget_state.elapsed_seconds = time.time() - start
                step_start = time.time()

                if not budget_state.is_within(budget_config):
                    raise BudgetExceededError(f"Budget exceeded at step {budget_state.steps}")

                if budget_state.steps > 1 and hasattr(self._llm, "use_recursive_model"):
                    self._llm.use_recursive_model()

                # Soft convergence nudge at budget fraction.
                nudge_step = self._compute_nudge_step(
                    budget_config.max_steps,
                    config.nudge_at_fraction,
                    minimum_step=2,
                )
                if not nudge_sent and nudge_step is not None and budget_state.steps >= nudge_step:
                    messages.append(
                        {
                            "role": "user",
                            "content": get_rlm_message("soft_nudge").format(
                                steps_used=budget_state.steps,
                                max_steps=budget_config.max_steps,
                            ),
                        }
                    )
                    nudge_sent = True

                hard_nudge_step = self._compute_nudge_step(
                    budget_config.max_steps,
                    0.75,
                    minimum_step=4,
                )
                if (
                    not hard_nudge_sent
                    and hard_nudge_step is not None
                    and budget_state.steps >= hard_nudge_step
                    and budget_state.steps < (budget_config.max_steps or 0)
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": get_rlm_message("hard_nudge").format(
                                steps_used=budget_state.steps,
                                max_steps=budget_config.max_steps,
                            ),
                        }
                    )
                    hard_nudge_sent = True

                # On the last allowed step, nudge the model to produce a final answer.
                is_last_step = (
                    budget_config.max_steps is not None
                    and budget_state.steps >= budget_config.max_steps
                )
                call_messages = (
                    messages
                    + [
                        {
                            "role": "user",
                            "content": get_rlm_message("force_final_nudge"),
                        }
                    ]
                    if is_last_step
                    else messages
                )

                # Stream tokens if emitter and async streaming are available.
                # Break to synthesis fallback on context window overflow.
                try:
                    if event_emitter and hasattr(self._llm, "complete_stream_async"):
                        collected: list[str] = []
                        async for token in self._llm.complete_stream_async(call_messages):
                            collected.append(token)
                            await event_emitter.on_token(token)
                        text = "".join(collected)
                        # Approximate token counts for streamed responses
                        approx_input = max(1, sum(len(m["content"]) for m in call_messages) // 4)
                        approx_output = max(1, len(text) // 4)
                        response = LLMResponseDTO(
                            content=text,
                            model=getattr(self._llm, "active_model", ""),
                            input_tokens=approx_input,
                            output_tokens=approx_output,
                        )
                    elif hasattr(self._llm, "complete_async"):
                        response = await self._llm.complete_async(call_messages)
                    else:
                        response = await asyncio.to_thread(self._llm.complete, call_messages)
                except Exception as llm_exc:
                    if self._is_context_overflow(llm_exc):
                        overflow_at_step = budget_state.steps
                        logger.warning(
                            "Context window exceeded at step %d; attempting synthesis fallback",
                            budget_state.steps,
                        )
                        break
                    raise

                cumulative_input += response.input_tokens
                cumulative_output += response.output_tokens
                budget_state.input_tokens = cumulative_input
                budget_state.output_tokens = cumulative_output
                cumulative_cost += (
                    response.input_tokens * input_cost_per_1m
                    + response.output_tokens * output_cost_per_1m
                ) / 1_000_000
                budget_state.cost = cumulative_cost

                # Re-check budget immediately after consuming tokens
                if not budget_state.is_within(budget_config):
                    raise BudgetExceededError(f"Budget exceeded after step {budget_state.steps}")

                text = response.content

                # Parse response: try JSON v2.0 first, fall back to markdown v1.0
                parsed = self._parse_rlm_response(text)

                # Track non-code responses as a last-resort fallback answer.
                if text.strip() and not parsed.has_code:
                    last_assistant_text = text.strip()

                trace_seq += 1
                clamp_info = getattr(self._llm, "last_clamp_info", None) or {}
                step_entry: dict[str, Any] = {
                    "step": budget_state.steps,
                    "seq": trace_seq,
                    "role": "assistant",
                    "content": response.content,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "code": parsed.code,
                    "model": getattr(self._llm, "active_model", None) or response.model,
                    "elapsed_seconds": time.time() - step_start,
                }
                if clamp_info:
                    step_entry["clamp"] = clamp_info
                trace.append(step_entry)

                # Emit step event
                if event_emitter:
                    await event_emitter.on_step(step_entry)
                    await event_emitter.on_metrics(
                        {
                            "input_tokens": cumulative_input,
                            "output_tokens": cumulative_output,
                            "total_tokens": cumulative_input + cumulative_output,
                            "steps": budget_state.steps,
                            "elapsed_seconds": time.time() - start,
                        }
                    )

                # Check for FINAL answer (handles both FINAL: and FINAL_VAR:)
                if parsed.is_complete:
                    missing_files = self._get_missing_files(
                        attached_files, inspected_files, enforce_multi_file_coverage
                    )
                    if missing_files:
                        prev_trace_len = len(trace)
                        trace_seq, inspect_snapshots, auto_file_no = (
                            self._auto_inspect_missing_file(
                                content=content,
                                config=config,
                                messages=messages,
                                trace=trace,
                                trace_seq=trace_seq,
                                budget_state=budget_state,
                                inspect_snapshots=inspect_snapshots,
                                missing_files=missing_files,
                                assistant_text=text,
                            )
                        )
                        inspected_files.add(auto_file_no)
                        consecutive_no_progress = 0
                        if event_emitter:
                            for entry in trace[prev_trace_len:]:
                                await event_emitter.on_step(entry)
                        continue
                    if last_execution_failed:
                        trace_seq += 1
                        trace.append(
                            {
                                "step": budget_state.steps,
                                "seq": trace_seq,
                                "role": "system",
                                "content": (
                                    "⚠️ Warning: FINAL provided after execution failure. "
                                    "Model may have used direct reasoning instead of code execution."
                                ),
                            }
                        )
                    if hasattr(self._llm, "use_root_model"):
                        self._llm.use_root_model()
                    elapsed = time.time() - start
                    return RunResultDTO(
                        answer=self._extract_final_answer(parsed),
                        mode_used="rlm",
                        success=True,
                        steps=budget_state.steps,
                        input_tokens=cumulative_input,
                        output_tokens=cumulative_output,
                        total_cost=cumulative_cost,
                        elapsed_time=elapsed,
                        trace=trace,
                    )

                # Check for code to execute
                if parsed.has_code and parsed.code:
                    consecutive_no_progress = 0
                    if parsed.is_inspect:
                        inspected_files.update(
                            self._infer_inspected_files_from_code(parsed.code, attached_files)
                        )
                    parent_budget_snapshot["cost"] = cumulative_cost
                    parent_budget_snapshot["tokens"] = float(cumulative_input + cumulative_output)
                    parent_budget_snapshot["steps"] = float(budget_state.steps)
                    exec_result = await asyncio.to_thread(self._sandbox.execute, parsed.code)
                    last_execution_failed = exec_result.exception is not None or exec_result.timeout
                    # Fold subcall token/cost/step usage back into parent accumulators
                    for _u in subcall_usage:
                        cumulative_input += _u.get("input_tokens", 0)
                        cumulative_output += _u.get("output_tokens", 0)
                        cumulative_cost += _u.get("total_cost", 0.0)
                        budget_state.steps += _u.get("steps", 0)
                    subcall_usage.clear()
                    budget_state.input_tokens = cumulative_input
                    budget_state.output_tokens = cumulative_output
                    budget_state.cost = cumulative_cost
                    formatted = self._format_execution(exec_result)
                    if parsed.is_inspect:
                        inspect_snapshots = self._append_inspect_snapshot(
                            inspect_snapshots,
                            step=budget_state.steps,
                            code=parsed.code,
                            execution_output=formatted,
                        )

                    trace_seq += 1
                    trace.append(
                        {
                            "step": budget_state.steps,
                            "seq": trace_seq,
                            "role": "execution",
                            "content": formatted,
                            "code": parsed.code,
                        }
                    )

                    # Repeat detection — only for inspect actions.
                    repeat_nudge: str | None = None
                    if parsed.is_inspect:
                        exec_hash = hash(formatted[:500])
                        is_repeat = exec_hash in recent_exec_hashes
                        recent_exec_hashes.append(exec_hash)
                        if len(recent_exec_hashes) > 5:
                            recent_exec_hashes.pop(0)

                        if is_repeat:
                            repeat_count += 1
                            if repeat_count >= repeat_limit:
                                repeat_nudge = get_rlm_message("repeat_detected")
                            else:
                                repeat_nudge = get_rlm_message("duplicate_result")
                        else:
                            repeat_count = 0

                    messages.append({"role": "assistant", "content": text})
                    # Truncate large execution outputs before adding to message
                    # history.  Limit is recomputed per step to account for
                    # accumulated context usage.  We include overhead for the
                    # "Execution result:\n" prefix (~20 chars), truncation suffix
                    # (~50 chars), and any repeat nudge so the cap reflects the
                    # full payload about to be appended, not just the raw body.
                    _wrapper_overhead = 70 + (len(repeat_nudge) if repeat_nudge else 0)
                    current_msg_chars = (
                        sum(len(m.get("content", "")) for m in messages) + _wrapper_overhead
                    )
                    exec_result_limit = self._compute_exec_result_limit(
                        len(content),
                        config.max_steps,
                        current_message_chars=current_msg_chars,
                        model_context_limit=getattr(self._llm, "context_length_chars", None),
                        override=config.extra.get("exec_result_limit"),
                    )
                    exec_content = formatted
                    if len(exec_content) > exec_result_limit:
                        exec_content = (
                            exec_content[:exec_result_limit]
                            + f"\n… [truncated, {len(formatted) - exec_result_limit} chars omitted]"
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Execution result:\n{exec_content}",
                        }
                    )
                    # Append repeat-detection nudge AFTER the exec result so the
                    # model sees both the output (for learning) and the guidance.
                    if repeat_nudge is not None:
                        messages.append({"role": "user", "content": repeat_nudge})
                else:
                    stripped = text.strip()
                    missing_files = self._get_missing_files(
                        attached_files, inspected_files, enforce_multi_file_coverage
                    )
                    if missing_files:
                        prev_trace_len = len(trace)
                        trace_seq, inspect_snapshots, auto_file_no = (
                            self._auto_inspect_missing_file(
                                content=content,
                                config=config,
                                messages=messages,
                                trace=trace,
                                trace_seq=trace_seq,
                                budget_state=budget_state,
                                inspect_snapshots=inspect_snapshots,
                                missing_files=missing_files,
                                assistant_text=text,
                            )
                        )
                        inspected_files.add(auto_file_no)
                        consecutive_no_progress = 0
                        if event_emitter:
                            for entry in trace[prev_trace_len:]:
                                await event_emitter.on_step(entry)
                        continue
                    if stripped and not missing_files:
                        last_plain_answer = stripped
                    consecutive_no_progress += 1

                    # Early post-coverage synthesis (async path).
                    if not missing_files and enforce_multi_file_coverage:
                        post_coverage_stall += 1
                    else:
                        post_coverage_stall = 0

                    if (
                        post_coverage_stall >= 2
                        and inspect_snapshots
                        and enforce_multi_file_coverage
                    ):
                        logger.info(
                            "Early post-coverage synthesis at step %d "
                            "(model stalled %d turns after coverage complete)",
                            budget_state.steps,
                            post_coverage_stall,
                        )
                        trace_seq += 1
                        synth_note_entry = {
                            "step": budget_state.steps,
                            "seq": trace_seq,
                            "role": "system",
                            "content": (
                                "Early synthesis: coverage complete but model "
                                "did not finalize after 2 no-progress turns."
                            ),
                        }
                        trace.append(synth_note_entry)
                        if event_emitter:
                            await event_emitter.on_step(synth_note_entry)
                        if hasattr(self._llm, "use_root_model"):
                            self._llm.use_root_model()
                        synth_step_start = time.time()
                        if hasattr(self._llm, "complete_async"):
                            synth_response = await self._llm.complete_async(
                                self._build_synthesis_messages(query, inspect_snapshots)
                            )
                        else:
                            synth_response = await asyncio.to_thread(
                                self._llm.complete,
                                self._build_synthesis_messages(query, inspect_snapshots),
                            )
                        budget_state.steps += 1
                        cumulative_input += synth_response.input_tokens
                        cumulative_output += synth_response.output_tokens
                        cumulative_cost += (
                            synth_response.input_tokens * input_cost_per_1m
                            + synth_response.output_tokens * output_cost_per_1m
                        ) / 1_000_000
                        trace_seq += 1
                        synth_entry = {
                            "step": budget_state.steps,
                            "seq": trace_seq,
                            "role": "assistant",
                            "content": synth_response.content,
                            "input_tokens": synth_response.input_tokens,
                            "output_tokens": synth_response.output_tokens,
                            "model": (
                                getattr(self._llm, "active_model", None) or synth_response.model
                            ),
                            "elapsed_seconds": time.time() - synth_step_start,
                            "note": "early post-coverage synthesis",
                        }
                        trace.append(synth_entry)
                        if event_emitter:
                            await event_emitter.on_step(synth_entry)
                        answer = synth_response.content.strip() or (
                            "The content was inspected but contained no extractable themes."
                        )
                        if not self._looks_like_action(answer):
                            elapsed = time.time() - start
                            return RunResultDTO(
                                answer=answer,
                                mode_used="rlm",
                                success=True,
                                steps=budget_state.steps,
                                input_tokens=cumulative_input,
                                output_tokens=cumulative_output,
                                total_cost=cumulative_cost,
                                elapsed_time=elapsed,
                                trace=trace,
                            )

                    if consecutive_no_progress >= stall_limit:
                        if (
                            last_plain_answer
                            and not self._looks_like_action(last_plain_answer)
                            and not missing_files
                            and not enforce_multi_file_coverage
                        ):
                            if hasattr(self._llm, "use_root_model"):
                                self._llm.use_root_model()
                            elapsed = time.time() - start
                            answer = self._extract_final(last_plain_answer) or last_plain_answer
                            return RunResultDTO(
                                answer=answer,
                                mode_used="rlm",
                                success=True,
                                steps=budget_state.steps,
                                input_tokens=cumulative_input,
                                output_tokens=cumulative_output,
                                total_cost=cumulative_cost,
                                elapsed_time=elapsed,
                                trace=trace,
                            )
                        raise BudgetExceededError(
                            f"LLM did not make progress after {consecutive_no_progress} "
                            "consecutive steps without code or a FINAL answer"
                        )
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": get_rlm_message("reprompt"),
                        }
                    )

            best_fallback = last_plain_answer or last_assistant_text
            if self._get_missing_files(
                attached_files, inspected_files, enforce_multi_file_coverage
            ):
                best_fallback = ""
            # Don't return raw JSON action blobs as the final answer —
            # fall through to synthesis or limit-warning instead.
            if best_fallback and self._looks_like_action(best_fallback):
                best_fallback = ""
            if best_fallback:
                if hasattr(self._llm, "use_root_model"):
                    self._llm.use_root_model()
                elapsed = time.time() - start
                answer = self._extract_final(best_fallback) or best_fallback
                return RunResultDTO(
                    answer=answer,
                    mode_used="rlm",
                    success=True,
                    steps=budget_state.steps,
                    input_tokens=cumulative_input,
                    output_tokens=cumulative_output,
                    total_cost=cumulative_cost,
                    elapsed_time=elapsed,
                    trace=trace,
                )
            if inspect_snapshots:
                if hasattr(self._llm, "use_root_model"):
                    self._llm.use_root_model()
                budget_state.elapsed_seconds = time.time() - start
                if not budget_state.is_within(budget_config):
                    raise BudgetExceededError("Budget exceeded before synthesis call")
                synth_step_start = time.time()
                if hasattr(self._llm, "complete_async"):
                    synth_response = await self._llm.complete_async(
                        self._build_synthesis_messages(query, inspect_snapshots)
                    )
                else:
                    synth_response = await asyncio.to_thread(
                        self._llm.complete,
                        self._build_synthesis_messages(query, inspect_snapshots),
                    )
                budget_state.steps += 1
                cumulative_input += synth_response.input_tokens
                cumulative_output += synth_response.output_tokens
                cumulative_cost += (
                    synth_response.input_tokens * input_cost_per_1m
                    + synth_response.output_tokens * output_cost_per_1m
                ) / 1_000_000
                budget_state.input_tokens = cumulative_input
                budget_state.output_tokens = cumulative_output
                budget_state.cost = cumulative_cost
                budget_state.elapsed_seconds = time.time() - start
                _non_step_config = BudgetConfig(
                    max_tokens=budget_config.max_tokens,
                    max_cost=budget_config.max_cost,
                    max_time_seconds=budget_config.max_time_seconds,
                )
                if not budget_state.is_within(_non_step_config):
                    raise BudgetExceededError("Budget exceeded after synthesis call")
                trace_seq += 1
                trace.append(
                    {
                        "step": budget_state.steps,
                        "seq": trace_seq,
                        "role": "assistant",
                        "content": synth_response.content,
                        "input_tokens": synth_response.input_tokens,
                        "output_tokens": synth_response.output_tokens,
                        "model": getattr(self._llm, "active_model", None) or synth_response.model,
                        "elapsed_seconds": time.time() - synth_step_start,
                        "note": "synthesis fallback",
                    }
                )
                answer = synth_response.content.strip() or (
                    "The content was inspected but contained no extractable themes."
                )
                # If the synthesis call itself returned a JSON action blob
                # (model stuck in action mode), fall through to the limit warning.
                if not self._looks_like_action(answer):
                    elapsed = time.time() - start
                    return RunResultDTO(
                        answer=answer,
                        mode_used="rlm",
                        success=True,
                        steps=budget_state.steps,
                        input_tokens=cumulative_input,
                        output_tokens=cumulative_output,
                        total_cost=cumulative_cost,
                        elapsed_time=elapsed,
                        trace=trace,
                    )
            # No answer available — return an actionable warning.
            # Treated as success=True so the result is visible in chat and
            # can be evaluated by Auto-Judge.
            if hasattr(self._llm, "use_root_model"):
                self._llm.use_root_model()
            elapsed = time.time() - start
            return RunResultDTO(
                answer=self._format_limit_warning(
                    overflow_step=overflow_at_step,
                    steps_used=budget_state.steps,
                    config=budget_config,
                ),
                mode_used="rlm",
                success=True,
                steps=budget_state.steps,
                input_tokens=cumulative_input,
                output_tokens=cumulative_output,
                total_cost=cumulative_cost,
                elapsed_time=elapsed,
                trace=trace,
            )

        except BudgetExceededError as exc:
            # Config/limit hit — return a warning, not an error, so the user
            # knows exactly which setting to change. success=True lets Auto-Judge run.
            if hasattr(self._llm, "use_root_model"):
                self._llm.use_root_model()
            elapsed = time.time() - start
            partial = last_plain_answer or last_assistant_text
            if enforce_multi_file_coverage or self._get_missing_files(
                attached_files, inspected_files, enforce_multi_file_coverage
            ):
                partial = ""
            if enforce_multi_file_coverage and inspect_snapshots:
                budget_state.elapsed_seconds = time.time() - start
                if budget_state.is_within(budget_config):
                    synth_step_start = time.time()
                    if hasattr(self._llm, "complete_async"):
                        synth_response = await self._llm.complete_async(
                            self._build_synthesis_messages(query, inspect_snapshots)
                        )
                    else:
                        synth_response = await asyncio.to_thread(
                            self._llm.complete,
                            self._build_synthesis_messages(query, inspect_snapshots),
                        )
                    budget_state.steps += 1
                    cumulative_input += synth_response.input_tokens
                    cumulative_output += synth_response.output_tokens
                    cumulative_cost += (
                        synth_response.input_tokens * input_cost_per_1m
                        + synth_response.output_tokens * output_cost_per_1m
                    ) / 1_000_000
                    trace_seq += 1
                    synth_entry = {
                        "step": budget_state.steps,
                        "seq": trace_seq,
                        "role": "assistant",
                        "content": synth_response.content,
                        "input_tokens": synth_response.input_tokens,
                        "output_tokens": synth_response.output_tokens,
                        "model": getattr(self._llm, "active_model", None) or synth_response.model,
                        "elapsed_seconds": time.time() - synth_step_start,
                        "note": "synthesis fallback",
                    }
                    trace.append(synth_entry)
                    if event_emitter:
                        await event_emitter.on_step(synth_entry)
                    answer = synth_response.content.strip() or (
                        "The content was inspected but contained no extractable themes."
                    )
                    prior_plain = (last_plain_answer or last_assistant_text).strip()
                    if not self._looks_like_action(answer) and answer != prior_plain:
                        return RunResultDTO(
                            answer=answer,
                            mode_used="rlm",
                            success=True,
                            steps=budget_state.steps,
                            input_tokens=cumulative_input,
                            output_tokens=cumulative_output,
                            total_cost=cumulative_cost,
                            elapsed_time=time.time() - start,
                            trace=trace,
                        )
            warning = self._format_limit_warning(
                overflow_step=overflow_at_step,
                steps_used=budget_state.steps,
                config=budget_config,
                exc=exc,
            )
            answer = (partial + "\n\n---\n\n" + warning) if partial else warning
            return RunResultDTO(
                answer=answer,
                mode_used="rlm",
                success=True,
                steps=budget_state.steps,
                input_tokens=cumulative_input,
                output_tokens=cumulative_output,
                total_cost=cumulative_cost,
                elapsed_time=elapsed,
                trace=trace,
            )
        except Exception as exc:
            if hasattr(self._llm, "use_root_model"):
                self._llm.use_root_model()
            elapsed = time.time() - start
            # Context window exceeded is a configuration issue — warn, don't error.
            if self._is_context_overflow(exc):
                partial = last_plain_answer or last_assistant_text
                if self._get_missing_files(
                    attached_files, inspected_files, enforce_multi_file_coverage
                ):
                    partial = ""
                warning = self._format_limit_warning(
                    overflow_step=budget_state.steps,
                    steps_used=budget_state.steps,
                    config=budget_config,
                )
                answer = (partial + "\n\n---\n\n" + warning) if partial else warning
                return RunResultDTO(
                    answer=answer,
                    mode_used="rlm",
                    success=True,
                    steps=budget_state.steps,
                    input_tokens=cumulative_input,
                    output_tokens=cumulative_output,
                    total_cost=cumulative_cost,
                    elapsed_time=elapsed,
                    trace=trace,
                )
            return RunResultDTO(
                answer="",
                mode_used="rlm",
                success=False,
                error=str(exc),
                steps=budget_state.steps,
                input_tokens=cumulative_input,
                output_tokens=cumulative_output,
                total_cost=cumulative_cost,
                elapsed_time=elapsed,
                trace=trace,
            )

    # -- Private helpers --

    def _emit_runtime_fingerprint(
        self,
        *,
        trace: list[dict[str, Any]],
        attached_files: dict[int, _AttachedFileSection],
        enforce_multi_file_coverage: bool,
        execution_path: str,
    ) -> None:
        """Emit a seq=0 trace entry with runtime metadata for diagnostics.

        Helps identify path/deployment mismatches (e.g. vLLM vs Ollama hitting
        different code paths or stale adapter classes).
        """
        adapter_class = type(self._llm).__name__
        active_model = getattr(self._llm, "active_model", None) or getattr(
            self._llm, "_active_model", None
        )
        root_model = getattr(self._llm, "_root_model", None)
        api_base = getattr(self._llm, "_api_base", None)

        # Context window enforcement config (static, set once at adapter creation)
        context_window = getattr(self._llm, "_context_window", None)
        configured_max_tokens = getattr(self._llm, "_max_tokens", None)

        fingerprint = {
            "step": 0,
            "seq": 0,
            "role": "system",
            "content": "Runtime fingerprint",
            "fingerprint": {
                "adapter_class": adapter_class,
                "active_model": str(active_model) if active_model else None,
                "root_model": str(root_model) if root_model else None,
                "api_base": str(api_base) if api_base else None,
                "execution_path": execution_path,
                "coverage_enforced": enforce_multi_file_coverage,
                "attached_file_count": len(attached_files),
                "attached_files": [
                    {"file_no": f.file_no, "name": f.name} for f in attached_files.values()
                ],
                "context_window": context_window,
                "configured_max_tokens": configured_max_tokens,
            },
        }
        trace.insert(0, fingerprint)
        logger.info(
            "RLM runtime fingerprint: adapter=%s model=%s path=%s coverage=%s files=%d",
            adapter_class,
            active_model,
            execution_path,
            enforce_multi_file_coverage,
            len(attached_files),
        )

    def _make_subcall(
        self,
        config: RunConfigDTO,
        subcall_usage: list[dict[str, Any]],
        parent_budget_snapshot: dict[str, float],
    ) -> Any:
        """Create a subcall closure for LLM-generated code to call recursively.

        *subcall_usage* is a mutable list that the closure appends to after each
        child execution so the parent loop can fold the child's token and cost
        usage back into its own accumulators.
        """
        import copy

        parent_llm = self._llm
        parent_sandbox = self._sandbox

        def subcall(content: str, query: str, max_steps: int | None = None) -> str:
            if config.max_recursion_depth <= 0:
                return "Error: Maximum recursion depth exceeded"

            # Compute usage already accumulated by earlier subcalls in this block.
            # parent_budget_snapshot holds the parent's totals at the start of this
            # code block; subcall_usage accumulates each prior sibling subcall's usage.
            block_cost = sum(u.get("total_cost", 0.0) for u in subcall_usage)
            block_tokens = sum(
                u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in subcall_usage
            )
            block_steps = sum(u.get("steps", 0) for u in subcall_usage)

            # Guard: don't spawn when earlier siblings have already consumed the headroom.
            if config.max_cost is not None:
                if parent_budget_snapshot["cost"] + block_cost >= config.max_cost:
                    return (
                        f"Error: max_cost={config.max_cost} budget exhausted; "
                        "no further subcalls permitted in this code block"
                    )
            if config.max_tokens is not None:
                if parent_budget_snapshot["tokens"] + block_tokens >= config.max_tokens:
                    return (
                        f"Error: max_tokens={config.max_tokens} budget exhausted; "
                        "no further subcalls permitted in this code block"
                    )
            if config.max_steps is not None:
                if int(parent_budget_snapshot["steps"]) + block_steps >= config.max_steps:
                    return (
                        f"Error: max_steps={config.max_steps} budget exhausted; "
                        "no further subcalls permitted in this code block"
                    )
            # Time is measured via the real clock, so we call time.time() here rather
            # than accumulating a block_time: any time spent by previous siblings in
            # this block is already reflected in the elapsed reading.
            if config.max_time_seconds is not None:
                elapsed = time.time() - parent_budget_snapshot["time_start"]
                if elapsed >= config.max_time_seconds:
                    return (
                        f"Error: max_time_seconds={config.max_time_seconds} budget exhausted; "
                        "no further subcalls permitted in this code block"
                    )

            try:
                sub_sandbox = copy.deepcopy(parent_sandbox)
                sub_sandbox.reset()
            except Exception as exc:
                return f"Error: Cannot create subcall environment: {exc}"

            # Give each child only the remaining headroom so it cannot spend beyond
            # what the parent's global budget still allows.
            remaining_cost = (
                config.max_cost - parent_budget_snapshot["cost"] - block_cost
                if config.max_cost is not None
                else None
            )
            remaining_tokens = (
                int(config.max_tokens - parent_budget_snapshot["tokens"] - block_tokens)
                if config.max_tokens is not None
                else None
            )
            # max_steps=N allows N-1 LLM calls (step N increments the counter and hits
            # is_within before the LLM call).  So to allow R remaining calls, the child
            # needs max_steps=R+1.
            remaining_steps = (
                config.max_steps - int(parent_budget_snapshot["steps"]) - block_steps
                if config.max_steps is not None
                else None
            )
            child_remaining = remaining_steps + 1 if remaining_steps is not None else None
            # Explicit max_steps arg further constrains the child; take the tighter bound.
            if max_steps is not None:
                child_max_steps = (
                    max_steps if child_remaining is None else min(max_steps, child_remaining)
                )
            else:
                child_max_steps = child_remaining
            # Remaining wall-clock headroom: measured via the real clock so it
            # automatically accounts for time spent by previous siblings in this block.
            remaining_time = (
                config.max_time_seconds - (time.time() - parent_budget_snapshot["time_start"])
                if config.max_time_seconds is not None
                else None
            )
            sub_config = RunConfigDTO(
                mode="rlm",
                max_steps=child_max_steps,
                max_recursion_depth=config.max_recursion_depth - 1,
                stall_limit=config.stall_limit,
                repeat_limit=config.repeat_limit,
                nudge_at_fraction=config.nudge_at_fraction,
                max_tokens=remaining_tokens,
                max_cost=remaining_cost,
                max_time_seconds=remaining_time,
            )
            sub_uc = RunRLMUseCase(parent_llm, sub_sandbox)
            sub_result = sub_uc.execute(content, query, config=sub_config)
            # Report child usage to the parent so it can be folded back in
            subcall_usage.append(
                {
                    "input_tokens": sub_result.input_tokens,
                    "output_tokens": sub_result.output_tokens,
                    "total_cost": sub_result.total_cost,
                    "steps": sub_result.steps,
                }
            )
            return (
                sub_result.answer if sub_result.success else f"Error in subcall: {sub_result.error}"
            )

        return subcall

    @staticmethod
    def _get_pricing(llm: LLMPort) -> tuple[float, float]:
        """Return (input_cost_per_1m, output_cost_per_1m) USD, defaulting to 0.0."""
        try:
            pricing = llm.get_pricing()
            return (
                float(pricing.get("input_cost_per_1m", 0.0)),
                float(pricing.get("output_cost_per_1m", 0.0)),
            )
        except Exception:
            return (0.0, 0.0)

    @staticmethod
    def _extract_attached_files(content: str) -> dict[int, _AttachedFileSection]:
        """Parse attached-file metadata from the multi-file document index."""
        if not content.startswith("[DOCUMENT INDEX"):
            return {}

        pattern = re.compile(
            r'^\s*(?P<file_no>\d+)\.\s+"(?P<name>.+?)"\s+\('
            r"file_start=(?P<file_start>\d+), "
            r"content_start=(?P<content_start>\d+), "
            r"file_end_exclusive=(?P<file_end>\d+)\)$",
            flags=re.MULTILINE,
        )
        attached_files: dict[int, _AttachedFileSection] = {}
        for match in pattern.finditer(content):
            file_no = int(match.group("file_no"))
            attached_files[file_no] = _AttachedFileSection(
                file_no=file_no,
                name=match.group("name"),
                file_start=int(match.group("file_start")),
                content_start=int(match.group("content_start")),
                file_end_exclusive=int(match.group("file_end")),
            )
        return attached_files

    @staticmethod
    def _should_enforce_multi_file_coverage(
        query: str,
        attached_files: dict[int, _AttachedFileSection],
    ) -> bool:
        """Return True when the query implies a cross-document summary/comparison."""
        if len(attached_files) <= 1:
            return False

        lowered = query.lower()
        coverage_terms = (
            "compare",
            "common",
            "commonal",
            "both",
            "all files",
            "all documents",
            "uploaded documents",
            "uploaded files",
            "documents",
            "files",
            "summarize",
            "summary",
            "differences",
            "shared",
            "similarities",
        )
        return any(term in lowered for term in coverage_terms)

    @staticmethod
    def _infer_inspected_files_from_code(
        code: str,
        attached_files: dict[int, _AttachedFileSection],
    ) -> set[int]:
        """Infer which files were inspected by a tool call."""
        inspected_files: set[int] = set()

        for match in re.finditer(r"(?:outline_file|peek_file|grep_file)\(file_no=(\d+)", code):
            file_no = int(match.group(1))
            if file_no in attached_files:
                inspected_files.add(file_no)

        for match in re.finditer(r"peek\(start=(\d+),\s*end=", code):
            start = int(match.group(1))
            for file_no, section in attached_files.items():
                if section.content_start <= start < section.file_end_exclusive:
                    inspected_files.add(file_no)
                    break

        return inspected_files

    @staticmethod
    def _get_missing_files(
        attached_files: dict[int, _AttachedFileSection],
        inspected_files: set[int],
        enforce_multi_file_coverage: bool,
    ) -> list[_AttachedFileSection]:
        """Return files still missing required inspection coverage."""
        if not enforce_multi_file_coverage:
            return []
        return [
            section
            for file_no, section in sorted(attached_files.items())
            if file_no not in inspected_files
        ]

    @staticmethod
    def _format_missing_files(missing_files: list[_AttachedFileSection]) -> str:
        """Format missing-file metadata for user-facing reprompts."""
        return ", ".join(f'{section.file_no} ("{section.name}")' for section in missing_files)

    @classmethod
    def _format_coverage_gate_note(cls, missing_files: list[_AttachedFileSection]) -> str:
        """Build a trace-visible note when multi-file coverage blocks finalization."""
        return (
            "Coverage gate blocked early finalization; inspect missing files first: "
            f"{cls._format_missing_files(missing_files)}"
        )

    def _auto_inspect_missing_file(
        self,
        *,
        content: str,
        config: RunConfigDTO,
        messages: list[dict[str, str]],
        trace: list[dict[str, Any]],
        trace_seq: int,
        budget_state: BudgetState,
        inspect_snapshots: list[str],
        missing_files: list[_AttachedFileSection],
        assistant_text: str,
    ) -> tuple[int, list[str], int]:
        """Automatically inspect one missing file when coverage blocks finalization.

        Small local models often fail to comply with a reprompt telling them to
        inspect another file. When that happens, proactively run a lightweight
        `outline_file()` for the next missing file and feed the result back into
        the loop so the model can continue from a complete structural view.
        """
        target = missing_files[0]
        coverage_note = self._format_coverage_gate_note(missing_files)

        trace_seq += 1
        trace.append(
            {
                "step": budget_state.steps,
                "seq": trace_seq,
                "role": "system",
                "content": coverage_note,
            }
        )

        code = f"print(outline_file(file_no={target.file_no}, max_lines=25, max_chars=8000))"
        exec_result = self._sandbox.execute(code)
        formatted = self._format_execution(exec_result)
        inspect_snapshots = self._append_inspect_snapshot(
            inspect_snapshots,
            step=budget_state.steps,
            code=code,
            execution_output=formatted,
        )

        trace_seq += 1
        trace.append(
            {
                "step": budget_state.steps,
                "seq": trace_seq,
                "role": "execution",
                "content": formatted,
                "code": code,
                "note": f"auto coverage inspect for file {target.file_no}",
            }
        )

        messages.append({"role": "assistant", "content": assistant_text})
        current_msg_chars = sum(len(m.get("content", "")) for m in messages) + 120
        exec_result_limit = self._compute_exec_result_limit(
            len(content),
            config.max_steps,
            current_message_chars=current_msg_chars,
            model_context_limit=getattr(self._llm, "context_length_chars", None),
            override=config.extra.get("exec_result_limit"),
        )
        exec_content = formatted
        if len(exec_content) > exec_result_limit:
            exec_content = (
                exec_content[:exec_result_limit]
                + f"\n… [truncated, {len(formatted) - exec_result_limit} chars omitted]"
            )
        messages.append({"role": "user", "content": f"Execution result:\n{exec_content}"})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Coverage guard auto-inspected missing file {target.file_no}. "
                    "Continue using all inspected files. Do not ask the user to inspect files manually."
                ),
            }
        )
        return trace_seq, inspect_snapshots, target.file_no

    def _parse_rlm_response(self, text: str) -> ParsedResponse:
        """Parse LLM response: try JSON v2.0 first, fall back to markdown v1.0."""
        from rlmkit.core.actions import ParseError, parse_action
        from rlmkit.core.parsing import ParsedResponse, parse_response

        try:
            action_type, action_obj = parse_action(text)
            if action_type == "final":
                return ParsedResponse(final_answer=action_obj.answer, raw_text=text)
            elif action_type == "inspect":
                code = self._inspect_to_code(action_obj)
                if code:
                    return ParsedResponse(code=code, raw_text=text, is_inspect=True)
                return ParsedResponse(raw_text=text)
            elif action_type == "subcall":
                # Translate to executable Python; subcall() is bound in sandbox globals
                code = (
                    f"_sub_result = subcall("
                    f"content={repr(action_obj.prompt)}, "
                    f"query={repr(action_obj.query)})\n"
                    f"print(_sub_result)"
                )
                return ParsedResponse(code=code, raw_text=text)
            # unknown action type: fall through to markdown
        except (ParseError, Exception):
            pass

        # v1.0 markdown / plain-text fallback.
        # Re-use is_inspect so the synthesis fallback works for models that
        # emit raw Python with peek/grep/chunk/select instead of JSON actions.
        result = parse_response(text)
        if (
            result.has_code
            and result.code
            and any(
                f"{tool}(" in result.code
                for tool in (
                    "peek",
                    "peek_file",
                    "grep",
                    "grep_file",
                    "outline_file",
                    "chunk",
                    "select",
                )
            )
        ):
            result.is_inspect = True
        return result

    @staticmethod
    def _inspect_to_code(action_obj: Any) -> str | None:
        """Convert a JSON inspect action to executable Python code."""
        tool: str = action_obj.tool
        args: dict[str, Any] = action_obj.args
        if tool == "grep":
            return (
                f"print(grep("
                f"pattern={repr(args.get('pattern'))}, "
                f"context_lines={args.get('context_lines', 2)}, "
                f"max_matches={args.get('max_matches', 100)}, "
                f"ignore_case={args.get('ignore_case', False)}, "
                f"use_regex={args.get('use_regex', False)}))"
            )
        elif tool == "peek":
            return (
                f"print(peek("
                f"start={args.get('start', 0)}, "
                f"end={args.get('end')}, "
                f"max_chars={args.get('max_chars', 10000)}))"
            )
        elif tool == "peek_file":
            return (
                f"print(peek_file("
                f"file_no={args.get('file_no')}, "
                f"start={args.get('start', 0)}, "
                f"end={args.get('end')}, "
                f"max_chars={args.get('max_chars', 10000)}))"
            )
        elif tool == "grep_file":
            return (
                f"print(grep_file("
                f"file_no={args.get('file_no')}, "
                f"pattern={repr(args.get('pattern'))}, "
                f"context_lines={args.get('context_lines', 2)}, "
                f"max_matches={args.get('max_matches', 100)}, "
                f"ignore_case={args.get('ignore_case', False)}, "
                f"use_regex={args.get('use_regex', False)}))"
            )
        elif tool == "outline_file":
            return (
                f"print(outline_file("
                f"file_no={args.get('file_no')}, "
                f"max_lines={args.get('max_lines', 40)}, "
                f"max_chars={args.get('max_chars', 8000)}))"
            )
        elif tool == "select":
            return f"print(select(ranges={args.get('ranges')}))"
        elif tool == "chunk":
            return (
                f"print(chunk("
                f"size={args.get('size', 1000)}, "
                f"overlap={args.get('overlap', 0)}, "
                f"by={repr(args.get('by', 'chars'))}, "
                f"max_chunks={args.get('max_chunks', 100)}))"
            )
        return None

    def _extract_final_answer(self, parsed: ParsedResponse) -> str:
        """Extract final answer, resolving FINAL_VAR: references via the sandbox."""
        if parsed.final_answer:
            return str(parsed.final_answer)
        if parsed.final_var:
            var_value = self._sandbox.get_variable(parsed.final_var)
            if var_value is not None:
                return str(var_value)
            return f"Error: Variable '{parsed.final_var}' not found in environment"
        return "Error: No final answer found"

    @staticmethod
    def _compute_nudge_step(
        max_steps: int | None,
        fraction: float,
        *,
        minimum_step: int,
    ) -> int | None:
        """Return the step number at which a budget nudge should fire."""
        if max_steps is None:
            return None
        return max(minimum_step, int(max_steps * fraction))

    @staticmethod
    def _append_inspect_snapshot(
        snapshots: list[str],
        *,
        step: int,
        code: str | None,
        execution_output: str,
        per_snapshot_chars: int = 500,
        max_snapshots: int = 12,
    ) -> list[str]:
        """Append a compact inspect snapshot for synthesis fallback."""
        first_code_line = "inspect"
        if code:
            for line in code.splitlines():
                stripped = line.strip()
                if stripped:
                    first_code_line = stripped
                    break

        output = execution_output.strip() or "(no output)"
        if len(output) > per_snapshot_chars:
            output = (
                output[:per_snapshot_chars] + f"\n... (truncated to {per_snapshot_chars} chars)"
            )

        snapshots = list(snapshots)
        snapshots.append(f"[Inspect step {step}]\nTool call: {first_code_line}\n{output}")
        if len(snapshots) > max_snapshots:
            snapshots = snapshots[-max_snapshots:]
        return snapshots

    @staticmethod
    def _build_synthesis_messages(query: str, execution_output: list[str]) -> list[dict[str, str]]:
        """Build messages for a synthesis call when max steps exhausted without a final answer."""
        inspection_summary = "\n\n".join(execution_output)[:6000]
        return [
            {
                "role": "system",
                "content": get_rlm_message("synthesis_system"),
            },
            {
                "role": "user",
                "content": get_rlm_message("synthesis_user").format(
                    query=query,
                    execution_output=inspection_summary,
                ),
            },
        ]

    @staticmethod
    def _is_context_overflow(exc: Exception) -> bool:
        """Return True if *exc* is a context-window-exceeded error from the LLM."""
        msg = str(exc).lower()
        return ("context" in msg or "8192" in msg or "4096" in msg) and any(
            kw in msg
            for kw in ("window", "length", "exceed", "exhausted", "maximum", "too large", "token")
        )

    @staticmethod
    def _format_limit_warning(
        overflow_step: int | None = None,
        steps_used: int = 0,
        config: BudgetConfig | None = None,
        exc: Exception | None = None,
    ) -> str:
        """Return a user-friendly warning for known configuration-driven limits.

        These are *not* bugs — they indicate that the provider's settings need
        adjustment.  The warning includes specific mitigation steps so the user
        knows exactly what to change.
        """
        exc_msg = str(exc).lower() if exc else ""

        if overflow_step is not None:
            return (
                f"⚠️ **Context window exceeded** at step {overflow_step}.\n\n"
                "The model's context limit was reached during recursive exploration "
                "and the response could not be completed.\n\n"
                "**How to fix** — in Settings → Chat Providers → [this provider] → Runtime Settings:\n"
                "- Reduce **Max Output Tokens** (e.g. from 4096 → 512 or 1024). "
                "RLM steps generate small code snippets; reserving 4096 tokens for output "
                "leaves very little room for the growing conversation context.\n"
                "- Or reduce **Max Steps** to limit how deep the exploration goes."
            )
        if "timeout" in exc_msg or ("time" in exc_msg and "budget" in exc_msg):
            timeout = config.max_time_seconds if config else None
            timeout_str = f" ({int(timeout)}s limit)" if timeout else ""
            return (
                f"⚠️ **Execution timed out** after {steps_used} steps{timeout_str}.\n\n"
                "**How to fix** — in Settings → Chat Providers → [this provider]:\n"
                "- Increase **Timeout** to allow more time for analysis."
            )
        if "token" in exc_msg and any(
            kw in exc_msg for kw in ("budget", "exceed", "maximum", "limit")
        ):
            max_tok = config.max_tokens if config else None
            tok_str = f" (limit: {max_tok:,})" if max_tok else ""
            return (
                f"⚠️ **Token budget exceeded** after {steps_used} steps{tok_str}.\n\n"
                "**How to fix** — in Settings → Budget:\n"
                "- Increase **Max Tokens** per execution."
            )
        # Generic steps-exhausted case
        max_steps = config.max_steps if config else None
        steps_str = f" of {max_steps}" if max_steps else ""
        return (
            f"⚠️ **Step budget exhausted** ({steps_used}{steps_str} steps used) "
            "without producing a complete answer.\n\n"
            "**How to fix** — in Settings → Chat Providers → [this provider]:\n"
            "- Increase **Max Steps** to allow deeper document analysis.\n"
            "- Or ask a more focused question."
        )

    @staticmethod
    def _extract_final(text: str) -> str | None:
        """Extract FINAL: answer from LLM response (used by tests and as fallback)."""
        import re

        match = re.search(r"^FINAL:\s*(.*)", text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_code(text: str) -> str | None:
        """Extract Python code block from LLM response (used by tests and as fallback)."""
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
    def _looks_like_action(text: str) -> bool:
        """Return True if *text* looks like a JSON action, not a real answer.

        The stall-detector circuit breaker should not return raw JSON action
        blobs (inspect/final/subcall) as the final answer.  This heuristic
        catches the common case where ``parse_action`` failed on otherwise
        valid-looking JSON.
        """
        stripped = text.strip()
        return stripped.startswith("{") and '"type"' in stripped

    @staticmethod
    def _build_system_prompt(content_length: int) -> str:
        """Build the RLM system prompt from the versioned template file.

        Template files live in ``src/rlmkit/prompts/`` and are easy to discover
        and customize without touching Python source. The default is v2.0
        (``system_prompt_v2_0.yaml``); switch to v1.0 (``system_prompt_v1_0.yaml``)
        by passing ``version="1.0"`` to ``format_system_prompt``.
        """
        from rlmkit.prompts import format_system_prompt

        return str(format_system_prompt(prompt_length=content_length))

    @staticmethod
    def _compute_exec_result_limit(
        content_length: int,
        max_steps: int | None,
        current_message_chars: int = 0,
        model_context_limit: int | None = None,
        override: int | None = None,
    ) -> int:
        """Scale execution result size based on content, budget, and context usage.

        Returns a character limit clamped to [500, 8000].  When *override* is
        set it takes precedence.  Otherwise the heuristic targets 80% content
        coverage spread across the step budget, with an optional context-aware
        cap that avoids pushing past 80% of the model's context window.
        """
        if override is not None:
            return max(override, 500)

        effective_steps = max(max_steps or 16, 4)
        adaptive = int(content_length * 0.8 / effective_steps)
        limit = max(2000, min(adaptive, 8000))

        # Context-aware cap: don't push result + existing messages past 80% of
        # the model's context window.  Never return more than actual headroom.
        if model_context_limit is not None and model_context_limit > 0:
            headroom = int(model_context_limit * 0.8) - current_message_chars
            if headroom > 0:
                limit = min(limit, headroom)
            else:
                # Context already exceeded 80% — return absolute minimum.
                limit = 500
        return limit

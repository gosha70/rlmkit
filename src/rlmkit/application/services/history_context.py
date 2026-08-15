"""Per-provider conversation history assembly.

This module is the pure-logic home of the history feature described in
``doc_internal/rlm_history/SDD_RLM_CONVERSATION_HISTORY.md``.  It has
two independent output paths:

- **In-prompt replay** (Direct/Compare): take prior turns, assemble the
  newest-first slice that fits a token budget, and render a
  ``"Previous conversation:\\n..."`` prefix that gets glued onto the
  current query.  Bounded by the model's context window.

- **REPL variable** (RLM/RAG/Auto): build a Python list of
  ``{"turn": int, "user": str, "assistant": str}`` dicts, capped by a
  byte size derived from system RAM, for binding as the ``history``
  variable inside the subprocess sandbox.  The model reads from it on
  demand via plain Python (``print(history[-1])``) — the prompt does
  not carry the history, so the token cost is zero unless the model
  chooses to inspect it.

The module is intentionally free of FastAPI, AppState, and adapter
dependencies: everything is pure functions over plain dataclasses so
that unit tests do not need an HTTP stack.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rlmkit.branding import env, env_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Auto-derived REPL history cap — see compute_history_cap_bytes().
_HISTORY_CAP_MIN = 256 * 1024 * 1024  # 256 MB floor
_HISTORY_CAP_MAX = 4 * 1024 * 1024 * 1024  # 4 GB ceiling
_HISTORY_CAP_FRACTION = 0.055  # ~5.5% of total RAM
_HISTORY_CAP_FALLBACK = 4 * 1024 * 1024 * 1024  # fallback if sysconf fails
_HISTORY_CAP_ENV_VAR = env_name("HISTORY_MAX_BYTES")

# Truncate any single user/assistant message in the in-prompt replay to
# this many characters before token-counting.  Matches the legacy
# behaviour that the current inline code in chat.py uses (500 char clip
# on any individual message) so the swap is behaviour-preserving.
_INPROMPT_PER_MESSAGE_CHAR_CAP = 500


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryTurn:
    """One completed user ↔ assistant exchange from a prior turn.

    ``index`` is the turn number in the session (0-based), used as the
    ``turn`` key when the turn is materialised into the ``history``
    REPL variable.  ``user_content`` and ``assistant_content`` are the
    final messages for that turn with tool-call traces already
    stripped (the conversation store only holds final messages).
    """

    index: int
    user_content: str
    assistant_content: str


@dataclass
class HistoryAssemblyResult:
    """Result of assembling in-prompt history within a token budget."""

    messages: list[dict[str, str]] = field(default_factory=list)
    tokens_used: int = 0
    turns_used: int = 0
    turns_available: int = 0
    turns_dropped: int = 0


class TokenCounter(Protocol):
    """Minimal typing hook for the adapter's count_tokens(messages=...)."""

    def __call__(self, *, messages: list[dict[str, str]]) -> int: ...


# ---------------------------------------------------------------------------
# Extraction: flat message list → turns
# ---------------------------------------------------------------------------


def extract_final_qa_pairs(prev_msgs: list[dict[str, Any]]) -> list[HistoryTurn]:
    """Pair consecutive user/assistant messages into completed turns.

    The AppState conversation store holds a flat interleaved list of
    ``{"role": ..., "content": ..., ...}`` dicts.  This walks the list
    in order and emits one ``HistoryTurn`` per (user → assistant) pair.
    Messages that do not fit a clean user/assistant alternation, have
    empty content, or are error markers (``content`` starts with
    ``"Error:"``) are skipped.

    The most recent message is **excluded** — callers pass the full
    history list and the current turn's user message is assumed to be
    the tail; we don't want it folded into its own prior context.

    **Intentional behaviour change vs the legacy inline replay in
    ``chat.py``.**  The legacy code took ``eligible[-6:]`` from the
    filtered message list regardless of pairing, which meant an
    unanswered user question (for example, one whose assistant
    response errored and was filtered out) was replayed as a lone
    ``User: ...`` line with no following ``Assistant: ...``.  That is
    both confusing to the model (it tries to answer the question a
    second time) and impossible to materialise into the REPL
    ``history`` variable, which requires ``{turn, user, assistant}``
    dicts.  The strict pairing here drops such orphan messages.  See
    ``test_error_dropped_turn_drops_orphan_user_message`` for the
    explicit pin.

    Args:
        prev_msgs: The full conversation-store list from
            ``AppState.get_conversation``.  May be empty.

    Returns:
        A chronological list of completed turns.  The list is empty
        when there is no complete prior exchange.
    """
    if len(prev_msgs) < 2:
        return []

    # Drop the trailing user message for the current turn.  The list
    # ends with the just-posted user message when this is called from
    # the HTTP/WS handler, so we want everything before it.
    body = prev_msgs[:-1]

    turns: list[HistoryTurn] = []
    i = 0
    turn_index = 0
    while i < len(body):
        user = body[i]
        if user.get("role") != "user" or not _is_usable_content(user.get("content")):
            i += 1
            continue

        # Found a usable user message.  Scan forward through all
        # consecutive assistant messages and pick the *first usable*
        # one.  This handles:
        #
        #  - Normal (1 assistant): pair immediately.
        #  - Compare (2 assistants, both OK): pair with first (RLM).
        #  - Compare (RLM errored, Direct OK): skip the "Error: ..."
        #    and pair with the Direct answer.
        #  - Both errored: no pair — orphan user dropped.
        j = i + 1
        paired_assistant = None
        while j < len(body) and body[j].get("role") == "assistant":
            if paired_assistant is None and _is_usable_content(body[j].get("content")):
                paired_assistant = body[j]
            j += 1

        if paired_assistant is not None:
            turns.append(
                HistoryTurn(
                    index=turn_index,
                    user_content=str(user["content"]),
                    assistant_content=str(paired_assistant["content"]),
                )
            )
            turn_index += 1

        # Advance past the user + all its assistant messages.
        i = max(j, i + 1)

    return turns


def _is_usable_content(content: Any) -> bool:
    """Return True when the message has non-empty, non-error text."""
    if not content:
        return False
    text = str(content)
    if not text.strip():
        return False
    if text.startswith("Error:"):
        return False
    return True


# ---------------------------------------------------------------------------
# In-prompt path: budgeted assembly + prefix composition
# ---------------------------------------------------------------------------


def compute_inprompt_budget(
    *,
    system_prompt_tokens: int,
    current_query_tokens: int,
    reply_reserve: int,
    fraction_cap_tokens: int,
    context_window: int,
) -> int:
    """Return the token budget available for in-prompt history replay.

    Per SDD §6.3, the budget is the smaller of:

    - A fixed fraction of the context window (defaults to 30% but
      passed in by the caller as ``fraction_cap_tokens`` so a future
      per-provider override can tune it).
    - What is actually left over after reserving space for the system
      prompt, the current user query, and the minimum reply budget.

    A non-positive result is clamped to 0 — the caller then emits no
    history prefix and runs the turn as if there were none.
    """
    headroom = context_window - system_prompt_tokens - current_query_tokens - reply_reserve
    return max(0, min(fraction_cap_tokens, headroom))


def assemble_inprompt_history_within_budget(
    prev_turns: list[HistoryTurn],
    budget_tokens: int,
    token_counter: Callable[..., int],
) -> HistoryAssemblyResult:
    """Pack as many prior turns as fit a token budget, newest-first.

    Walks ``prev_turns`` from newest to oldest, rendering each one as
    a pair of chat messages (``user``/``assistant``), measuring the
    cumulative token cost via the injected ``token_counter``.  Stops
    when adding another turn would exceed ``budget_tokens``.  Returns
    the kept messages **in chronological order** so the caller's
    prompt reads correctly.

    Args:
        prev_turns: All available prior turns (oldest to newest).
        budget_tokens: Maximum tokens the in-prompt history may use.
        token_counter: A callable accepting ``messages=`` that returns
            the token count, matching the LiteLLMAdapter signature.

    Returns:
        A :class:`HistoryAssemblyResult` describing exactly what was
        kept and how much of the budget it used.
    """
    result = HistoryAssemblyResult(turns_available=len(prev_turns))
    if not prev_turns or budget_tokens <= 0:
        result.turns_dropped = len(prev_turns)
        return result

    kept_reverse: list[dict[str, str]] = []
    running_tokens = 0

    for turn in reversed(prev_turns):
        pair = _render_turn_as_messages(turn)
        try:
            candidate_tokens = int(token_counter(messages=kept_reverse + pair))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("token_counter failed during history assembly: %s", exc)
            break
        if candidate_tokens > budget_tokens:
            break
        kept_reverse = kept_reverse + pair
        running_tokens = candidate_tokens

    # kept_reverse holds newest-first pairs; reverse back to chronological.
    # Each pair is two messages, so flip in 2-element chunks.
    chronological: list[dict[str, str]] = []
    # Walk from the end of kept_reverse two at a time; but because we
    # appended `pair` (a [user, assistant] list) for each turn, we can
    # rebuild chronological by reversing whole pairs.
    pairs = [kept_reverse[i : i + 2] for i in range(0, len(kept_reverse), 2)]
    for pair in reversed(pairs):
        chronological.extend(pair)

    result.messages = chronological
    result.tokens_used = running_tokens
    result.turns_used = len(pairs)
    result.turns_dropped = len(prev_turns) - result.turns_used
    return result


def _render_turn_as_messages(turn: HistoryTurn) -> list[dict[str, str]]:
    """Render one turn as two chat messages, clipped to the per-msg cap."""
    return [
        {"role": "user", "content": _clip(turn.user_content)},
        {"role": "assistant", "content": _clip(turn.assistant_content)},
    ]


def _clip(text: str) -> str:
    if len(text) <= _INPROMPT_PER_MESSAGE_CHAR_CAP:
        return text
    return text[:_INPROMPT_PER_MESSAGE_CHAR_CAP] + "…"


def compose_inprompt_prefix(
    messages: list[dict[str, str]],
    current_query: str,
) -> str:
    """Build the ``Previous conversation: ...`` prefix string.

    Given the chronological message list produced by
    :func:`assemble_inprompt_history_within_budget`, format it the way
    the original inline code in ``chat.py`` did:

    .. code-block:: text

        Previous conversation:
        User: ...

        Assistant: ...

        Current question: <current_query>

    When ``messages`` is empty, returns ``current_query`` unchanged —
    the caller does not need to branch on the empty case itself.
    """
    if not messages:
        return current_query

    parts: list[str] = []
    for msg in messages:
        prefix = "User" if msg.get("role") == "user" else "Assistant"
        parts.append(f"{prefix}: {msg.get('content', '')}")
    context_str = "\n\n".join(parts)
    return f"Previous conversation:\n{context_str}\n\nCurrent question: {current_query}"


# ---------------------------------------------------------------------------
# REPL-variable path: auto cap + byte-budgeted assembly
# ---------------------------------------------------------------------------


def compute_history_cap_bytes() -> int:
    """Auto-derive the max bytes the ``history`` variable can hold.

    Target ≈5.5% of total system RAM, clamped to sensible bounds:

    - floor: 256 MB (tiny docker containers)
    - ceiling: 4 GB (no user notices caps beyond this for chat text)

    On a 36 GB Mac this works out to ~2 GB, which matches the goal
    of "almost-infinite" chat memory without letting the subprocess
    round-trip become a scalability problem.

    Override by setting the ``RLMKIT_HISTORY_MAX_BYTES`` environment
    variable to an integer byte count.
    """
    override = env("HISTORY_MAX_BYTES")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning("%s=%r is not an integer; ignoring", _HISTORY_CAP_ENV_VAR, override)

    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        total = _HISTORY_CAP_FALLBACK

    raw = int(total * _HISTORY_CAP_FRACTION)
    return max(_HISTORY_CAP_MIN, min(raw, _HISTORY_CAP_MAX))


def build_history_variable(
    prev_turns: list[HistoryTurn],
    cap_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the ``history`` REPL variable with FIFO byte-based eviction.

    Walks ``prev_turns`` newest-first, accumulating turns until the
    JSON-serialised size would exceed ``cap_bytes``.  The most recent
    turn is always kept even when it alone exceeds the cap — the
    alternative is silently dropping the thing the user just said,
    which is worse.  A warning is logged in that degenerate case.

    Args:
        prev_turns: Chronological list of prior turns.
        cap_bytes: Maximum JSON size (as reported by
            ``len(json.dumps(history))``).  Should come from
            :func:`compute_history_cap_bytes` in production.

    Returns:
        A 2-tuple of:

        - ``history_list`` — a list of
          ``{"turn": int, "user": str, "assistant": str}`` dicts
          ready to bind as a sandbox variable.  Chronological order
          (oldest first).
        - ``info`` — a diagnostics dict for the execution trace with
          keys ``history_variable_turn_count``,
          ``history_variable_byte_size``, ``history_turns_evicted``,
          ``history_cap_bytes``, ``history_variable_was_inspected``
          (always ``False`` at build time), ``history_variable_was_read``
          (always ``False``), and ``history_variable_read_bytes``
          (always ``0``).
    """
    info: dict[str, Any] = {
        "history_variable_turn_count": 0,
        "history_variable_byte_size": 0,
        "history_turns_evicted": 0,
        "history_cap_bytes": cap_bytes,
        "history_variable_was_inspected": False,
        "history_variable_was_read": False,
        "history_variable_read_bytes": 0,
    }

    if not prev_turns:
        return [], info

    # Walk newest-to-oldest, accumulating turns until the running JSON
    # size would exceed the cap.  The accumulator holds newest-first
    # and is reversed to chronological at the end.
    #
    # The most-recent turn is always admitted even if it alone exceeds
    # the cap — the alternative is silently dropping the thing the user
    # just said, which is worse.  A warning is logged in that case.
    kept_reverse: list[dict[str, Any]] = []
    for turn in reversed(prev_turns):
        entry = _turn_to_history_entry(turn)
        candidate = kept_reverse + [entry]
        candidate_bytes = len(json.dumps(candidate, ensure_ascii=False))
        if kept_reverse and candidate_bytes > cap_bytes:
            break
        kept_reverse = candidate

    history_list = list(reversed(kept_reverse))
    byte_size = len(json.dumps(history_list, ensure_ascii=False))

    if byte_size > cap_bytes:
        logger.warning(
            "history cap %d B too tight for last turn (%d B); kept it anyway",
            cap_bytes,
            byte_size,
        )

    info["history_variable_turn_count"] = len(history_list)
    info["history_variable_byte_size"] = byte_size
    info["history_turns_evicted"] = len(prev_turns) - len(history_list)
    return history_list, info


def _turn_to_history_entry(turn: HistoryTurn) -> dict[str, Any]:
    """Materialise a ``HistoryTurn`` into the REPL variable entry shape."""
    return {
        "turn": turn.index,
        "user": turn.user_content,
        "assistant": turn.assistant_content,
    }

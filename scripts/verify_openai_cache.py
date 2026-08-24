"""Reproduce OpenAI prompt caching end-to-end, two ways.

Purpose: close the loop on the Phase 2 cache-extraction work by
producing a deterministic scenario where OpenAI's automatic
``prompt_tokens_details.cached_tokens`` is guaranteed to be > 0.

Runs two back-to-back completions with an identical ~2200-token
prompt, first via LiteLLM directly (the library RLM Studio uses) and
then via :class:`LiteLLMAdapter` (the RLM Studio wrapper). Prints
``cached_tokens`` from each. This isolates three scenarios:

1. LiteLLM direct: cached>0, RLM Studio adapter: cached>0
     → pipeline works end-to-end. Any "Cached=0" you see in the UI
       is prompt-drift in the calling code, not a telemetry bug.

2. LiteLLM direct: cached>0, RLM Studio adapter: cached=0
     → bug inside the adapter or _extract_cache_tokens.

3. LiteLLM direct: cached=0, RLM Studio adapter: cached=0
     → OpenAI isn't caching for this account/workload. Nothing RLM Studio
       can do until OpenAI sees the prompt again in a cacheable
       configuration.

Requirements:
    OPENAI_API_KEY env var (falls back to the RLM Studio secret store if
    set there — see rlmstudio/llm/openai_client.py).

Usage:
    uv run python scripts/verify_openai_cache.py

Notes on the prompt:
    OpenAI caches prefixes ≥ 1024 tokens. We send ~2200 tokens of
    byte-identical content to comfortably exceed the threshold and
    give OpenAI a clear caching opportunity on the second call. Cache
    window is ~5-10 minutes; the two calls here are ~1 second apart.
"""

from __future__ import annotations

import asyncio
import os
import sys


def _long_stable_system_prompt() -> str:
    """Build a ~2000-token system prompt guaranteed to be identical
    between calls. Plain English paragraphs — nothing dynamic."""
    paragraph = (
        "You are a careful assistant whose job is to answer questions "
        "about technical software engineering topics. When the user "
        "asks a question, consider the full context before replying, "
        "prefer concrete examples over abstract generalizations, cite "
        "specific code patterns when relevant, and flag any ambiguity "
        "or underspecification rather than guessing. Keep your answers "
        "focused on the question actually asked. Do not invent facts, "
        "and when you are uncertain about a detail, say so. "
    )
    # Repeat the paragraph enough times to comfortably exceed
    # OpenAI's 1024-token minimum for caching. ~35 repetitions of a
    # ~60-token paragraph lands near ~2100 tokens.
    return paragraph * 35


_SHARED_USER_MESSAGE = "What is the single most important property of a good test?"


def _build_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _long_stable_system_prompt()},
        {"role": "user", "content": _SHARED_USER_MESSAGE},
    ]


def _extract_cached(resp: object) -> int:
    """Pull prompt_tokens_details.cached_tokens from any shape."""
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
    if details is None:
        return 0
    cached = getattr(details, "cached_tokens", None)
    if cached is None and isinstance(details, dict):
        cached = details.get("cached_tokens")
    return int(cached or 0)


def _prompt_tokens(resp: object) -> int:
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return 0
    pt = getattr(usage, "prompt_tokens", None)
    if pt is None and isinstance(usage, dict):
        pt = usage.get("prompt_tokens")
    return int(pt or 0)


def run_litellm_direct() -> tuple[int, int, int]:
    """Hit OpenAI twice via plain litellm. Returns (prompt_tokens,
    turn1_cached, turn2_cached)."""
    import litellm

    messages = _build_messages()
    print("  → call 1 (populates cache)…")
    r1 = litellm.completion(model="gpt-4o", messages=messages, max_tokens=50)
    pt1, c1 = _prompt_tokens(r1), _extract_cached(r1)
    print(f"    prompt_tokens={pt1} cached_tokens={c1}")

    print("  → call 2 (expect cache hit)…")
    r2 = litellm.completion(model="gpt-4o", messages=messages, max_tokens=50)
    pt2, c2 = _prompt_tokens(r2), _extract_cached(r2)
    print(f"    prompt_tokens={pt2} cached_tokens={c2}")

    return pt2, c1, c2


async def run_rlmstudio_adapter() -> tuple[int, int, int]:
    """Hit OpenAI twice via LiteLLMAdapter. Same shape of result."""
    from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

    adapter = LiteLLMAdapter(model="gpt-4o", max_tokens=50)
    messages = _build_messages()

    print("  → call 1 (populates cache)…")
    r1 = await adapter.complete_async(messages)
    print(
        f"    input_tokens={r1.input_tokens} cached_tokens={r1.cached_tokens} ttft_ms={r1.ttft_ms}"
    )

    print("  → call 2 (expect cache hit)…")
    r2 = await adapter.complete_async(messages)
    print(
        f"    input_tokens={r2.input_tokens} cached_tokens={r2.cached_tokens} ttft_ms={r2.ttft_ms}"
    )

    return r2.input_tokens, r1.cached_tokens, r2.cached_tokens


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    print(f"Prompt size: ~{len(_long_stable_system_prompt()) // 4} est tokens")
    print(f"Target cache threshold: 1024 tokens")
    print()

    print("=" * 60)
    print("A. Direct LiteLLM (baseline: is OpenAI caching for this key?)")
    print("=" * 60)
    try:
        pt_direct, c1_direct, c2_direct = run_litellm_direct()
    except Exception as exc:
        print(f"❌ Direct LiteLLM failed: {exc}")
        return 3

    print()
    print("=" * 60)
    print("B. RLM Studio LiteLLMAdapter (does our wrapper propagate it?)")
    print("=" * 60)
    try:
        pt_rlm, c1_rlm, c2_rlm = asyncio.run(run_rlmstudio_adapter())
    except Exception as exc:
        print(f"❌ RLM Studio adapter failed: {exc}")
        return 4

    print()
    print("=" * 60)
    print("Verdict")
    print("=" * 60)
    if c2_direct > 0 and c2_rlm > 0:
        print(
            f"✅ End-to-end cache observability works.\n"
            f"   Direct litellm turn 2 cached={c2_direct}\n"
            f"   RLM Studio adapter turn 2 cached={c2_rlm}"
        )
        return 0
    if c2_direct > 0 and c2_rlm == 0:
        print(
            f"❌ Adapter bug: litellm saw cached={c2_direct} but RLM Studio returned 0.\n"
            f"   _extract_cache_tokens or _observe_chunk is dropping it."
        )
        return 10
    if c2_direct == 0:
        print(
            f"⚠️  OpenAI did not cache this request for this account.\n"
            f"   Both direct litellm and RLM Studio saw cached=0.\n"
            f"   Possible causes:\n"
            f"     - First-ever request of this exact prefix to this org\n"
            f"       (OpenAI requires a prior call within ~5-10 min)\n"
            f"     - API key belongs to a deployment/account that\n"
            f"       has caching disabled (rare)\n"
            f"     - OpenAI service quirk\n"
            f"   Re-run this script within 5 minutes — the second run's\n"
            f"   turn 1 should hit cache from this run's turn 2."
        )
        return 20
    return 30


if __name__ == "__main__":
    raise SystemExit(main())

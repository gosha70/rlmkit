"""Unit tests for the pure ``history_context`` service.

Every test here uses plain Python data structures and a deterministic
fake token counter — no FastAPI, AppState, adapters, or litellm.
"""

from __future__ import annotations

from unittest.mock import patch

from rlmstudio.application.services.history_context import (
    HistoryAssemblyResult,
    HistoryTurn,
    assemble_inprompt_history_within_budget,
    build_history_variable,
    compose_inprompt_prefix,
    compute_history_cap_bytes,
    compute_inprompt_budget,
    extract_final_qa_pairs,
)

# ---------------------------------------------------------------------------
# Fake token counter — 1 token per 4 chars, matches litellm fallback heuristic
# ---------------------------------------------------------------------------


def _fake_count_tokens(*, messages: list[dict[str, str]]) -> int:
    """Deterministic: 4 chars ≈ 1 token + 3 per message (chat overhead)."""
    return sum(max(1, len(m.get("content", "")) // 4) + 3 for m in messages)


# ---------------------------------------------------------------------------
# extract_final_qa_pairs
# ---------------------------------------------------------------------------


class TestExtractFinalQAPairs:
    def test_empty_list_returns_empty(self):
        assert extract_final_qa_pairs([]) == []

    def test_single_user_message_returns_empty(self):
        # Only the user's just-posted query is in the list; nothing prior.
        msgs = [{"role": "user", "content": "Hello"}]
        assert extract_final_qa_pairs(msgs) == []

    def test_one_complete_prior_turn(self):
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},  # current turn, excluded
        ]
        turns = extract_final_qa_pairs(msgs)
        assert len(turns) == 1
        assert turns[0] == HistoryTurn(
            index=0,
            user_content="What is 2+2?",
            assistant_content="4",
        )

    def test_multiple_turns_preserve_order(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},  # current, excluded
        ]
        turns = extract_final_qa_pairs(msgs)
        assert [t.user_content for t in turns] == ["Q1", "Q2"]
        assert [t.assistant_content for t in turns] == ["A1", "A2"]
        assert [t.index for t in turns] == [0, 1]

    def test_errored_turn_is_skipped(self):
        msgs = [
            {"role": "user", "content": "broken?"},
            {"role": "assistant", "content": "Error: LLM timeout"},
            {"role": "user", "content": "works"},
            {"role": "assistant", "content": "yes"},
            {"role": "user", "content": "current"},
        ]
        turns = extract_final_qa_pairs(msgs)
        assert len(turns) == 1
        assert turns[0].user_content == "works"
        # The kept turn is numbered 0, not 1 — errored turns don't
        # occupy a turn index.
        assert turns[0].index == 0

    def test_empty_content_skipped(self):
        msgs = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "real"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "current"},
        ]
        turns = extract_final_qa_pairs(msgs)
        assert len(turns) == 1
        assert turns[0].user_content == "real"

    def test_system_role_ignored(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "current"},
        ]
        turns = extract_final_qa_pairs(msgs)
        assert len(turns) == 1
        assert turns[0].user_content == "Q"

    def test_assistant_only_no_pair(self):
        msgs = [
            {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "current"},
        ]
        assert extract_final_qa_pairs(msgs) == []

    def test_error_dropped_turn_drops_orphan_user_message(self):
        """Explicit pin for the intentional behaviour change vs chat.py.

        Legacy code at ``chat.py`` replayed the last 6 eligible messages
        regardless of pairing, so a user question whose assistant
        response errored (and was therefore filtered out) was shown to
        the model as a lone ``User: ...`` line with no answer.  The
        new service drops such orphan messages because:

        1) the REPL ``history`` variable needs ``{turn, user, assistant}``
           dicts and cannot represent a half-turn;
        2) replaying an unanswered question as prior context is
           confusing — the model tries to answer it a second time.

        This test pins the new behaviour.  If you see it failing, the
        swap is no longer behaviour-preserving in a way that deserves
        a separate commit message note.
        """
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},
            {"role": "user", "content": "Q1-unanswered"},
            {"role": "assistant", "content": "Error: timeout"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "current query"},  # excluded
        ]
        turns = extract_final_qa_pairs(msgs)
        # Q0/A0 and Q2/A2 survive; Q1 is orphaned and dropped.
        assert [t.user_content for t in turns] == ["Q0", "Q2"]
        assert [t.assistant_content for t in turns] == ["A0", "A2"]
        # Turn indices are 0 and 1 — errored turns do not occupy a slot.
        assert [t.index for t in turns] == [0, 1]

    def test_compare_mode_double_assistant_keeps_first_skips_second(self):
        """Compare mode stores two assistant messages per user turn (RLM + Direct).

        The extractor must pair the user with the first assistant and
        skip the second, not desync pairing for subsequent turns.
        """
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0-rlm"},
            {"role": "assistant", "content": "A0-direct"},  # second answer
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1-rlm"},
            {"role": "assistant", "content": "A1-direct"},
            {"role": "user", "content": "current query"},  # excluded
        ]
        turns = extract_final_qa_pairs(msgs)
        assert len(turns) == 2
        assert turns[0].user_content == "Q0"
        assert turns[0].assistant_content == "A0-rlm"
        assert turns[1].user_content == "Q1"
        assert turns[1].assistant_content == "A1-rlm"
        assert turns[0].index == 0
        assert turns[1].index == 1

    def test_compare_rlm_error_direct_success_keeps_direct(self):
        """Compare: RLM half errors, Direct half succeeds.

        The extractor must skip the errored first assistant and pair
        the user with the second (Direct) answer instead of dropping
        the entire turn.
        """
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "Error: LLM timeout"},
            {"role": "assistant", "content": "good direct answer"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1-rlm"},
            {"role": "assistant", "content": "A1-direct"},
            {"role": "user", "content": "current"},  # excluded
        ]
        turns = extract_final_qa_pairs(msgs)
        assert len(turns) == 2
        # Turn 0 pairs with the Direct answer (the first usable one)
        assert turns[0].user_content == "Q0"
        assert turns[0].assistant_content == "good direct answer"
        # Turn 1 pairs normally with the RLM answer
        assert turns[1].user_content == "Q1"
        assert turns[1].assistant_content == "A1-rlm"

    def test_compare_both_errors_drops_turn(self):
        """Compare: both RLM and Direct fail → turn is dropped."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "Error: timeout"},
            {"role": "assistant", "content": "Error: rate limit"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "current"},
        ]
        turns = extract_final_qa_pairs(msgs)
        assert len(turns) == 1
        assert turns[0].user_content == "Q1"
        assert turns[0].assistant_content == "A1"

    def test_non_user_assistant_role_mid_stream_resyncs(self):
        """A stray system/tool role in the middle does not corrupt extraction."""
        msgs = [
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},
            {"role": "system", "content": "reminder injected mid-stream"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "current"},  # excluded
        ]
        turns = extract_final_qa_pairs(msgs)
        assert [t.user_content for t in turns] == ["Q0", "Q1"]
        assert [t.assistant_content for t in turns] == ["A0", "A1"]


# ---------------------------------------------------------------------------
# compute_inprompt_budget
# ---------------------------------------------------------------------------


class TestComputeInpromptBudget:
    def test_plenty_of_room_returns_fraction_cap(self):
        # fraction cap of 1000 tokens, context window of 8192
        # fixed = 200 + 50 + 128 = 378, headroom = 8192 - 378 = 7814
        # min(1000, 7814) = 1000
        budget = compute_inprompt_budget(
            system_prompt_tokens=200,
            current_query_tokens=50,
            reply_reserve=128,
            fraction_cap_tokens=1000,
            context_window=8192,
        )
        assert budget == 1000

    def test_headroom_below_fraction_cap(self):
        # Fraction cap 4000, but headroom only 1000
        budget = compute_inprompt_budget(
            system_prompt_tokens=6000,
            current_query_tokens=500,
            reply_reserve=692,
            fraction_cap_tokens=4000,
            context_window=8192,
        )
        assert budget == 1000

    def test_negative_headroom_clamps_to_zero(self):
        budget = compute_inprompt_budget(
            system_prompt_tokens=8000,
            current_query_tokens=500,
            reply_reserve=128,
            fraction_cap_tokens=1000,
            context_window=8192,
        )
        assert budget == 0

    def test_exact_fit(self):
        # headroom exactly equals fraction cap
        budget = compute_inprompt_budget(
            system_prompt_tokens=1000,
            current_query_tokens=500,
            reply_reserve=500,
            fraction_cap_tokens=2000,
            context_window=4000,
        )
        assert budget == 2000


# ---------------------------------------------------------------------------
# assemble_inprompt_history_within_budget
# ---------------------------------------------------------------------------


class TestAssembleInpromptHistory:
    def test_empty_turns_returns_empty(self):
        res = assemble_inprompt_history_within_budget(
            prev_turns=[],
            budget_tokens=1000,
            token_counter=_fake_count_tokens,
        )
        assert isinstance(res, HistoryAssemblyResult)
        assert res.messages == []
        assert res.tokens_used == 0
        assert res.turns_used == 0
        assert res.turns_dropped == 0

    def test_zero_budget_returns_empty(self):
        turns = [HistoryTurn(0, "hi", "there")]
        res = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=0,
            token_counter=_fake_count_tokens,
        )
        assert res.messages == []
        assert res.turns_dropped == 1

    def test_negative_budget_returns_empty(self):
        turns = [HistoryTurn(0, "hi", "there")]
        res = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=-5,
            token_counter=_fake_count_tokens,
        )
        assert res.messages == []
        assert res.turns_dropped == 1

    def test_all_turns_fit_preserve_chronological_order(self):
        turns = [
            HistoryTurn(0, "first q", "first a"),
            HistoryTurn(1, "second q", "second a"),
            HistoryTurn(2, "third q", "third a"),
        ]
        res = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=10000,
            token_counter=_fake_count_tokens,
        )
        assert res.turns_used == 3
        assert res.turns_dropped == 0
        # Chronological order: first, then second, then third
        assert [m["content"] for m in res.messages] == [
            "first q",
            "first a",
            "second q",
            "second a",
            "third q",
            "third a",
        ]
        # Alternating roles
        assert [m["role"] for m in res.messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    def test_oldest_dropped_first_when_budget_tight(self):
        # Each turn costs roughly 1 + 1 + 3 + 3 = 8 tokens via the fake
        # counter.  Budget 20 leaves room for ~2 turns.
        turns = [
            HistoryTurn(0, "q0xxxx", "a0xxxx"),
            HistoryTurn(1, "q1xxxx", "a1xxxx"),
            HistoryTurn(2, "q2xxxx", "a2xxxx"),
        ]
        res = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=20,
            token_counter=_fake_count_tokens,
        )
        # Oldest (turn 0) dropped; turn 1 and 2 kept.
        assert res.turns_used == 2
        assert res.turns_dropped == 1
        assert [m["content"] for m in res.messages] == [
            "q1xxxx",
            "a1xxxx",
            "q2xxxx",
            "a2xxxx",
        ]

    def test_single_turn_larger_than_budget_dropped_whole(self):
        huge_q = "x" * 2000
        huge_a = "y" * 2000
        turns = [HistoryTurn(0, huge_q, huge_a)]
        res = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=10,
            token_counter=_fake_count_tokens,
        )
        assert res.turns_used == 0
        assert res.turns_dropped == 1
        assert res.messages == []

    def test_newest_fits_but_older_does_not(self):
        # Budget fits newest turn only
        turns = [
            HistoryTurn(0, "old" + "x" * 2000, "old" + "x" * 2000),  # huge
            HistoryTurn(1, "new", "short"),  # ~3+3+3+3=12 tokens
        ]
        res = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=15,
            token_counter=_fake_count_tokens,
        )
        assert res.turns_used == 1
        assert res.turns_dropped == 1
        assert [m["content"] for m in res.messages] == ["new", "short"]

    def test_per_message_char_clip(self):
        huge = "x" * 1000
        turns = [HistoryTurn(0, huge, huge)]
        res = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=10000,
            token_counter=_fake_count_tokens,
        )
        # The clip is 500 chars + ellipsis
        assert res.turns_used == 1
        assert len(res.messages[0]["content"]) == 501  # 500 + "…"
        assert res.messages[0]["content"].endswith("…")


# ---------------------------------------------------------------------------
# compose_inprompt_prefix
# ---------------------------------------------------------------------------


class TestComposeInpromptPrefix:
    def test_empty_messages_returns_query_unchanged(self):
        assert compose_inprompt_prefix([], "What now?") == "What now?"

    def test_single_turn(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        out = compose_inprompt_prefix(msgs, "What now?")
        assert out == (
            "Previous conversation:\nUser: Q1\n\nAssistant: A1\n\nCurrent question: What now?"
        )

    def test_two_turns(self):
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        out = compose_inprompt_prefix(msgs, "Q3?")
        assert "Previous conversation:" in out
        assert out.endswith("Current question: Q3?")
        assert "User: Q1" in out
        assert "Assistant: A1" in out
        assert "User: Q2" in out
        assert "Assistant: A2" in out

    def test_matches_legacy_chat_py_format(self):
        """Pin the exact format so the Commit 4 swap is invisible."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        out = compose_inprompt_prefix(msgs, "How are you?")
        expected = (
            "Previous conversation:\n"
            "User: Hello\n\n"
            "Assistant: Hi there\n\n"
            "Current question: How are you?"
        )
        assert out == expected


# ---------------------------------------------------------------------------
# compute_history_cap_bytes
# ---------------------------------------------------------------------------


class TestComputeHistoryCapBytes:
    def test_env_override_int(self, monkeypatch):
        monkeypatch.setenv("RLM_STUDIO_HISTORY_MAX_BYTES", "1048576")  # 1 MB
        assert compute_history_cap_bytes() == 1_048_576

    def test_env_override_non_int_ignored(self, monkeypatch):
        monkeypatch.setenv("RLM_STUDIO_HISTORY_MAX_BYTES", "not-an-int")
        # Falls back to auto-derived value; must be in [256MB, 4GB]
        cap = compute_history_cap_bytes()
        assert 256 * 1024 * 1024 <= cap <= 4 * 1024 * 1024 * 1024

    def test_env_override_zero_clamps_to_minimum_of_one(self, monkeypatch):
        monkeypatch.setenv("RLM_STUDIO_HISTORY_MAX_BYTES", "0")
        # max(1, 0) = 1
        assert compute_history_cap_bytes() == 1

    def test_autoderive_within_bounds(self, monkeypatch):
        monkeypatch.delenv("RLM_STUDIO_HISTORY_MAX_BYTES", raising=False)
        cap = compute_history_cap_bytes()
        assert 256 * 1024 * 1024 <= cap <= 4 * 1024 * 1024 * 1024

    def test_tiny_ram_floors_at_256mb(self, monkeypatch):
        monkeypatch.delenv("RLM_STUDIO_HISTORY_MAX_BYTES", raising=False)
        # Pretend the host only reports 1 GB RAM → 1 GB * 0.055 ≈ 56 MB → floor
        with patch("os.sysconf") as mock_sysconf:
            mock_sysconf.side_effect = lambda key: {
                "SC_PHYS_PAGES": 262144,  # pages
                "SC_PAGE_SIZE": 4096,  # bytes per page → 1 GB total
            }[key]
            cap = compute_history_cap_bytes()
        assert cap == 256 * 1024 * 1024

    def test_huge_ram_ceilings_at_4gb(self, monkeypatch):
        monkeypatch.delenv("RLM_STUDIO_HISTORY_MAX_BYTES", raising=False)
        # Pretend the host has 256 GB RAM → 256 * 0.055 = ~14 GB → ceiling
        with patch("os.sysconf") as mock_sysconf:
            mock_sysconf.side_effect = lambda key: {
                "SC_PHYS_PAGES": 67_108_864,  # pages
                "SC_PAGE_SIZE": 4096,  # → 256 GB
            }[key]
            cap = compute_history_cap_bytes()
        assert cap == 4 * 1024 * 1024 * 1024

    def test_mac_36gb_yields_roughly_2gb(self, monkeypatch):
        """The user's reference point: 36 GB Mac → ~2 GB cap."""
        monkeypatch.delenv("RLM_STUDIO_HISTORY_MAX_BYTES", raising=False)
        with patch("os.sysconf") as mock_sysconf:
            mock_sysconf.side_effect = lambda key: {
                "SC_PHYS_PAGES": 9_437_184,  # 36 GB / 4096
                "SC_PAGE_SIZE": 4096,
            }[key]
            cap = compute_history_cap_bytes()
        # 36 GB * 0.055 = 1.98 GB
        assert 1.9 * 1024**3 < cap < 2.1 * 1024**3

    def test_sysconf_failure_falls_back(self, monkeypatch):
        monkeypatch.delenv("RLM_STUDIO_HISTORY_MAX_BYTES", raising=False)
        with patch("os.sysconf") as mock_sysconf:
            mock_sysconf.side_effect = OSError("not supported")
            cap = compute_history_cap_bytes()
        # Fallback is 4 GB * 0.055 = 225.28 MB → clamped up to 256 MB floor
        assert cap == 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# build_history_variable
# ---------------------------------------------------------------------------


class TestBuildHistoryVariable:
    def test_empty_turns_returns_empty_list(self):
        history, info = build_history_variable([], cap_bytes=1_000_000)
        assert history == []
        assert info["history_variable_turn_count"] == 0
        assert info["history_variable_byte_size"] == 0
        assert info["history_turns_evicted"] == 0
        assert info["history_cap_bytes"] == 1_000_000
        assert info["history_variable_was_inspected"] is False
        assert info["history_variable_was_read"] is False
        assert info["history_variable_read_bytes"] == 0

    def test_single_turn_fits(self):
        turns = [HistoryTurn(0, "hi", "hello")]
        history, info = build_history_variable(turns, cap_bytes=1_000_000)
        assert history == [{"turn": 0, "user": "hi", "assistant": "hello"}]
        assert info["history_variable_turn_count"] == 1
        assert info["history_turns_evicted"] == 0

    def test_multiple_turns_chronological_order(self):
        turns = [
            HistoryTurn(0, "q0", "a0"),
            HistoryTurn(1, "q1", "a1"),
            HistoryTurn(2, "q2", "a2"),
        ]
        history, info = build_history_variable(turns, cap_bytes=1_000_000)
        assert [h["turn"] for h in history] == [0, 1, 2]
        assert [h["user"] for h in history] == ["q0", "q1", "q2"]
        assert info["history_variable_turn_count"] == 3
        assert info["history_turns_evicted"] == 0

    def test_fifo_eviction_oldest_first(self):
        import json as _json

        turns = [
            HistoryTurn(0, "q0", "a0"),
            HistoryTurn(1, "q1", "a1"),
            HistoryTurn(2, "q2", "a2"),
        ]
        # Compute exact bytes for a 2-entry payload so the test is
        # independent of small formatting changes in json.dumps.
        two_entry = [
            {"turn": 1, "user": "q1", "assistant": "a1"},
            {"turn": 2, "user": "q2", "assistant": "a2"},
        ]
        three_entry = [
            {"turn": 0, "user": "q0", "assistant": "a0"},
            *two_entry,
        ]
        cap = len(_json.dumps(two_entry, ensure_ascii=False))
        assert cap < len(_json.dumps(three_entry, ensure_ascii=False))
        history, info = build_history_variable(turns, cap_bytes=cap)
        assert info["history_variable_turn_count"] == 2
        assert info["history_turns_evicted"] == 1
        # The two newest turns survive (1 and 2), in chronological order.
        assert [h["turn"] for h in history] == [1, 2]

    def test_most_recent_turn_kept_when_alone_exceeds_cap(self, caplog):
        huge_user = "x" * 10_000
        huge_assistant = "y" * 10_000
        turns = [HistoryTurn(0, huge_user, huge_assistant)]
        with caplog.at_level("WARNING"):
            history, info = build_history_variable(turns, cap_bytes=100)
        assert len(history) == 1
        assert history[0]["turn"] == 0
        assert info["history_variable_turn_count"] == 1
        assert info["history_turns_evicted"] == 0  # kept despite over cap
        assert "too tight" in caplog.text

    def test_most_recent_turn_kept_when_cap_would_drop_everything(self, caplog):
        """Even when multiple turns exceed the cap, keep the newest one."""
        turns = [
            HistoryTurn(0, "q0", "a0"),  # small
            HistoryTurn(1, "x" * 10_000, "y" * 10_000),  # huge, newest
        ]
        with caplog.at_level("WARNING"):
            history, info = build_history_variable(turns, cap_bytes=80)
        # Turn 1 is the newest; it alone exceeds the cap.  Older is dropped.
        assert len(history) == 1
        assert history[0]["turn"] == 1
        assert info["history_turns_evicted"] == 1

    def test_history_entry_shape(self):
        """Each entry is a plain dict with turn/user/assistant keys."""
        turns = [HistoryTurn(42, "question", "answer")]
        history, _ = build_history_variable(turns, cap_bytes=10_000)
        assert history[0] == {"turn": 42, "user": "question", "assistant": "answer"}
        assert set(history[0].keys()) == {"turn", "user", "assistant"}

    def test_byte_size_tracks_json_dumps(self):
        import json

        turns = [HistoryTurn(0, "hello", "world")]
        history, info = build_history_variable(turns, cap_bytes=10_000)
        expected = len(json.dumps(history, ensure_ascii=False))
        assert info["history_variable_byte_size"] == expected

    def test_not_aliased_across_calls(self):
        """Two calls with the same input must return independent lists."""
        turns = [HistoryTurn(0, "q", "a")]
        h1, _ = build_history_variable(turns, cap_bytes=10_000)
        h2, _ = build_history_variable(turns, cap_bytes=10_000)
        assert h1 is not h2
        assert h1[0] is not h2[0]
        # Mutating one does not affect the other.
        h1[0]["user"] = "mutated"
        assert h2[0]["user"] == "q"

    def test_info_cap_bytes_reflects_argument(self):
        _, info = build_history_variable([], cap_bytes=12345)
        assert info["history_cap_bytes"] == 12345


# ---------------------------------------------------------------------------
# Integration: extract → assemble round trip
# ---------------------------------------------------------------------------


class TestExtractAssembleRoundTrip:
    def test_full_pipeline_direct_mode_shape(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "alpha"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "beta"},
            {"role": "user", "content": "current query"},  # excluded
        ]
        turns = extract_final_qa_pairs(msgs)
        res = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=10_000,
            token_counter=_fake_count_tokens,
        )
        prefix = compose_inprompt_prefix(res.messages, "current query")
        assert "Previous conversation:" in prefix
        assert "User: first" in prefix
        assert "Assistant: alpha" in prefix
        assert "User: second" in prefix
        assert "Assistant: beta" in prefix
        assert prefix.endswith("Current question: current query")

    def test_full_pipeline_repl_mode_shape(self):
        msgs = [
            {"role": "user", "content": "what is X"},
            {"role": "assistant", "content": "X is foo"},
            {"role": "user", "content": "and Y"},
            {"role": "assistant", "content": "Y is bar"},
            {"role": "user", "content": "current"},  # excluded
        ]
        turns = extract_final_qa_pairs(msgs)
        history, info = build_history_variable(turns, cap_bytes=1_000_000)
        assert history == [
            {"turn": 0, "user": "what is X", "assistant": "X is foo"},
            {"turn": 1, "user": "and Y", "assistant": "Y is bar"},
        ]
        assert info["history_variable_turn_count"] == 2
        assert info["history_turns_evicted"] == 0


# ---------------------------------------------------------------------------
# Regression: the assembly uses the token_counter injected by the caller
# ---------------------------------------------------------------------------


class TestTokenCounterIsHonoured:
    def test_counter_called_with_messages_kwarg(self):
        calls = []

        def spy(*, messages):
            calls.append(list(messages))
            return sum(len(m["content"]) for m in messages)

        turns = [HistoryTurn(0, "q", "a")]
        res = assemble_inprompt_history_within_budget(
            prev_turns=turns,
            budget_tokens=1000,
            token_counter=spy,
        )
        assert res.turns_used == 1
        # spy was called at least once with a messages list
        assert len(calls) >= 1
        assert all(isinstance(m, dict) for m in calls[0])

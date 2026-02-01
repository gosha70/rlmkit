"""Tests for ConversationStore CRUD operations."""

import json
import pytest
from datetime import datetime

from rlmkit.storage.database import Database
from rlmkit.storage.conversation_store import (
    ConversationStore,
    _serialize_optional,
    _deserialize_response,
    _deserialize_metrics,
    _deserialize_comparison,
)
from rlmkit.ui.services.models import (
    ChatMessage, Response, ExecutionMetrics, ComparisonMetrics,
)


@pytest.fixture
def store(tmp_path):
    db = Database(tmp_path / "test.db")
    return ConversationStore(db)


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------

class TestConversationCRUD:
    def test_create_and_get(self, store):
        cid = store.create_conversation(name="My Chat", mode="rlm_only")
        conv = store.get_conversation(cid)
        assert conv is not None
        assert conv["name"] == "My Chat"
        assert conv["mode"] == "rlm_only"

    def test_create_defaults(self, store):
        cid = store.create_conversation()
        conv = store.get_conversation(cid)
        assert conv["name"] == "Untitled"
        assert conv["mode"] == "compare"

    def test_list_conversations(self, store):
        store.create_conversation(name="First")
        store.create_conversation(name="Second")
        convs = store.list_conversations()
        assert len(convs) == 2
        # Most recent first
        assert convs[0]["name"] == "Second"

    def test_list_includes_message_count(self, store):
        cid = store.create_conversation(name="Chat")
        msg = ChatMessage(user_query="hello", mode="direct_only")
        msg.direct_response = Response(content="hi", stop_reason="stop")
        store.save_message(cid, msg)
        convs = store.list_conversations()
        assert convs[0]["message_count"] == 1

    def test_rename(self, store):
        cid = store.create_conversation(name="Old")
        store.rename_conversation(cid, "New")
        conv = store.get_conversation(cid)
        assert conv["name"] == "New"

    def test_delete(self, store):
        cid = store.create_conversation(name="ToDelete")
        store.delete_conversation(cid)
        assert store.get_conversation(cid) is None

    def test_delete_cascades_messages(self, store):
        cid = store.create_conversation()
        msg = ChatMessage(user_query="test", mode="direct_only")
        store.save_message(cid, msg)
        assert store.get_message_count(cid) == 1
        store.delete_conversation(cid)
        assert store.get_message_count(cid) == 0

    def test_get_nonexistent(self, store):
        assert store.get_conversation("nope") is None

    def test_metadata_stored(self, store):
        cid = store.create_conversation(metadata={"key": "value"})
        conv = store.get_conversation(cid)
        assert conv["metadata"] == {"key": "value"}

    def test_provider_and_model(self, store):
        cid = store.create_conversation(provider="openai", model="gpt-4")
        conv = store.get_conversation(cid)
        assert conv["provider"] == "openai"
        assert conv["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------

class TestMessagePersistence:
    def test_save_and_load_direct_message(self, store):
        cid = store.create_conversation()
        msg = ChatMessage(
            user_query="What is 2+2?",
            mode="direct_only",
            direct_response=Response(content="4", stop_reason="stop"),
            direct_metrics=ExecutionMetrics(
                input_tokens=10, output_tokens=5, total_tokens=15,
                cost_usd=0.001, cost_breakdown={"input": 0.0005, "output": 0.0005},
                execution_time_seconds=0.5, steps_taken=0,
                memory_used_mb=10.0, memory_peak_mb=12.0,
                success=True, execution_type="direct",
            ),
        )
        store.save_message(cid, msg)
        loaded = store.load_messages(cid)
        assert len(loaded) == 1
        m = loaded[0]
        assert m.user_query == "What is 2+2?"
        assert m.mode == "direct_only"
        assert m.direct_response.content == "4"
        assert m.direct_metrics.total_tokens == 15
        assert m.direct_metrics.cost_usd == 0.001

    def test_save_and_load_rlm_message(self, store):
        cid = store.create_conversation()
        msg = ChatMessage(
            user_query="Explain gravity",
            mode="rlm_only",
            rlm_response=Response(content="Gravity is...", stop_reason="stop"),
            rlm_metrics=ExecutionMetrics(
                input_tokens=100, output_tokens=50, total_tokens=150,
                cost_usd=0.01, cost_breakdown={"input": 0.005, "output": 0.005},
                execution_time_seconds=2.0, steps_taken=3,
                memory_used_mb=20.0, memory_peak_mb=25.0,
                success=True, execution_type="rlm",
            ),
            rlm_trace=[{"step": 1, "role": "assistant", "content": "thinking..."}],
        )
        store.save_message(cid, msg)
        loaded = store.load_messages(cid)
        m = loaded[0]
        assert m.rlm_response.content == "Gravity is..."
        assert m.rlm_metrics.steps_taken == 3
        assert m.rlm_trace[0]["step"] == 1

    def test_save_and_load_compare_message(self, store):
        cid = store.create_conversation()
        msg = ChatMessage(
            user_query="Test compare",
            mode="compare",
            rlm_response=Response(content="rlm answer", stop_reason="stop"),
            rlm_metrics=ExecutionMetrics(
                input_tokens=100, output_tokens=50, total_tokens=150,
                cost_usd=0.01, cost_breakdown={},
                execution_time_seconds=2.0, steps_taken=3,
                memory_used_mb=0, memory_peak_mb=0, success=True,
            ),
            direct_response=Response(content="direct answer", stop_reason="stop"),
            direct_metrics=ExecutionMetrics(
                input_tokens=50, output_tokens=25, total_tokens=75,
                cost_usd=0.005, cost_breakdown={},
                execution_time_seconds=1.0, steps_taken=0,
                memory_used_mb=0, memory_peak_mb=0, success=True,
            ),
            comparison_metrics=ComparisonMetrics(
                rlm_cost_usd=0.01, rlm_time_seconds=2.0, rlm_tokens=150, rlm_steps=3,
                direct_cost_usd=0.005, direct_time_seconds=1.0, direct_tokens=75, direct_steps=0,
                cost_delta_usd=0.005, cost_delta_percent=100.0,
                time_delta_seconds=1.0, time_delta_percent=100.0,
                token_delta=75, recommendation="Direct is cheaper",
            ),
        )
        store.save_message(cid, msg)
        loaded = store.load_messages(cid)
        m = loaded[0]
        assert m.rlm_response.content == "rlm answer"
        assert m.direct_response.content == "direct answer"
        assert m.comparison_metrics.recommendation == "Direct is cheaper"

    def test_message_ordering(self, store):
        cid = store.create_conversation()
        for i in range(5):
            msg = ChatMessage(user_query=f"msg {i}", mode="direct_only")
            store.save_message(cid, msg)
        loaded = store.load_messages(cid)
        assert [m.user_query for m in loaded] == [f"msg {i}" for i in range(5)]

    def test_message_count(self, store):
        cid = store.create_conversation()
        assert store.get_message_count(cid) == 0
        for i in range(3):
            store.save_message(cid, ChatMessage(user_query=f"q{i}", mode="direct_only"))
        assert store.get_message_count(cid) == 3

    def test_message_with_error(self, store):
        cid = store.create_conversation()
        msg = ChatMessage(user_query="bad query", mode="rlm_only", error="timeout")
        store.save_message(cid, msg)
        loaded = store.load_messages(cid)
        assert loaded[0].error == "timeout"

    def test_in_progress_stored_as_false(self, store):
        """Messages are always loaded as not in_progress (completed)."""
        cid = store.create_conversation()
        msg = ChatMessage(user_query="q", mode="direct_only", in_progress=True)
        store.save_message(cid, msg)
        loaded = store.load_messages(cid)
        assert loaded[0].in_progress is False


# ---------------------------------------------------------------------------
# File context dedup
# ---------------------------------------------------------------------------

class TestFileContextDedup:
    def test_save_and_retrieve(self, store):
        h = store.save_file_context("hello world", filename="test.txt")
        content = store.get_file_context(h)
        assert content == "hello world"

    def test_deduplication(self, store):
        h1 = store.save_file_context("same content")
        h2 = store.save_file_context("same content")
        assert h1 == h2

    def test_different_content_different_hash(self, store):
        h1 = store.save_file_context("content A")
        h2 = store.save_file_context("content B")
        assert h1 != h2

    def test_missing_hash(self, store):
        assert store.get_file_context("nonexistent") is None

    def test_message_with_file_context(self, store):
        cid = store.create_conversation()
        msg = ChatMessage(
            user_query="summarize this",
            mode="rag_only",
            file_context="Big document content here...",
            file_info={"name": "doc.pdf", "size_mb": 1.5},
        )
        store.save_message(cid, msg)
        loaded = store.load_messages(cid)
        assert loaded[0].file_context == "Big document content here..."
        assert loaded[0].file_info["name"] == "doc.pdf"


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_serialize_none(self):
        assert _serialize_optional(None) is None

    def test_serialize_response(self):
        r = Response(content="hello", stop_reason="stop")
        s = _serialize_optional(r)
        assert s is not None
        d = json.loads(s)
        assert d["content"] == "hello"

    def test_deserialize_response(self):
        s = json.dumps({"content": "hi", "stop_reason": "stop", "raw_response": None})
        r = _deserialize_response(s)
        assert r.content == "hi"

    def test_deserialize_none(self):
        assert _deserialize_response(None) is None
        assert _deserialize_metrics(None) is None
        assert _deserialize_comparison(None) is None

    def test_metrics_round_trip(self):
        m = ExecutionMetrics(
            input_tokens=10, output_tokens=5, total_tokens=15,
            cost_usd=0.001, cost_breakdown={"input": 0.0005},
            execution_time_seconds=0.5, steps_taken=0,
            memory_used_mb=1.0, memory_peak_mb=2.0, success=True,
        )
        s = _serialize_optional(m)
        m2 = _deserialize_metrics(s)
        assert m2.total_tokens == 15
        assert m2.cost_breakdown == {"input": 0.0005}

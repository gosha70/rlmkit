"""Tests for WebSocket real-time streaming with mocked LLM adapters."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rlmkit.application.dto import LLMResponseDTO
from rlmkit.server.app import create_app
from rlmkit.server.dependencies import get_state, reset_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state():
    reset_state()
    yield
    reset_state()


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


class FakeStreamingLLM:
    """Fake LLM adapter that supports async streaming."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self._call_idx = 0
        self.active_model = "fake-model"

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponseDTO:
        text = self._get_next_response()
        return LLMResponseDTO(
            content=text, model="fake-model",
            input_tokens=10, output_tokens=5,
        )

    async def complete_async(self, messages: List[Dict[str, str]]) -> LLMResponseDTO:
        return self.complete(messages)

    async def complete_stream_async(
        self, messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        text = self._get_next_response()
        for char in text:
            yield char

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> Dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def use_root_model(self) -> None:
        pass

    def use_recursive_model(self) -> None:
        pass

    def _get_next_response(self) -> str:
        if self._call_idx < len(self._responses):
            text = self._responses[self._call_idx]
            self._call_idx += 1
            return text
        return self._responses[-1] if self._responses else ""


class FakeSandbox:
    """Minimal sandbox fake for RLM tests."""

    def set_variable(self, name: str, value: Any) -> None:
        pass

    def execute(self, code: str) -> Any:
        result = MagicMock()
        result.stdout = "output"
        result.stderr = ""
        result.exception = None
        result.timeout = False
        return result


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------


class TestWSConnection:
    def test_connected_event_on_connect(self, client):
        with client.websocket_connect("/ws/chat/test-session") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert msg["session_id"] == "test-session"

    def test_malformed_json_returns_error(self, client):
        with client.websocket_connect("/ws/chat/test-session") as ws:
            ws.receive_json()  # connected
            ws.send_text("not json")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["data"]["code"] == "INVALID_JSON"
            assert err["data"]["recoverable"] is True


# ---------------------------------------------------------------------------
# Direct mode streaming
# ---------------------------------------------------------------------------


class TestDirectStreaming:
    def test_token_events_sent_during_direct_streaming(self, client, monkeypatch):
        """Direct mode should emit token events followed by step, metrics, and complete."""
        fake_llm = FakeStreamingLLM(["Hello world"])

        state = get_state()
        monkeypatch.setattr(state, "create_llm_adapter", lambda: fake_llm)

        with client.websocket_connect("/ws/chat/s1") as ws:
            ws.receive_json()  # connected

            ws.send_json({
                "type": "query",
                "id": "msg-1",
                "query": "Say hello",
                "content": "some text",
                "mode": "direct",
            })

            events: list[dict] = []
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg["type"] in ("complete", "error"):
                    break

            types = [e["type"] for e in events]

            # Token events should appear
            assert "token" in types, f"Expected token events, got: {types}"

            # Step and metrics events should appear
            assert "step" in types, f"Expected step events, got: {types}"
            assert "metrics" in types, f"Expected metrics events, got: {types}"

            # Complete should be last
            assert types[-1] == "complete"

            # All events should carry the same message id
            for ev in events:
                assert ev["id"] == "msg-1"

    def test_complete_event_has_answer(self, client, monkeypatch):
        fake_llm = FakeStreamingLLM(["The answer is 42"])

        state = get_state()
        monkeypatch.setattr(state, "create_llm_adapter", lambda: fake_llm)

        with client.websocket_connect("/ws/chat/s1") as ws:
            ws.receive_json()  # connected

            ws.send_json({
                "type": "query",
                "id": "msg-2",
                "query": "What is the answer?",
                "content": "text",
                "mode": "direct",
            })

            complete_event = None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "complete":
                    complete_event = msg
                    break
                if msg["type"] == "error":
                    pytest.fail(f"Unexpected error: {msg}")

            assert complete_event is not None
            assert complete_event["data"]["answer"] == "The answer is 42"
            assert complete_event["data"]["success"] is True
            assert "metrics" in complete_event["data"]

    def test_token_data_reconstructs_answer(self, client, monkeypatch):
        fake_llm = FakeStreamingLLM(["abc"])

        state = get_state()
        monkeypatch.setattr(state, "create_llm_adapter", lambda: fake_llm)

        with client.websocket_connect("/ws/chat/s1") as ws:
            ws.receive_json()  # connected

            ws.send_json({
                "type": "query",
                "id": "msg-3",
                "query": "test",
                "content": "text",
                "mode": "direct",
            })

            tokens: list[str] = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "token":
                    tokens.append(msg["data"])
                elif msg["type"] in ("complete", "error"):
                    break

            assert "".join(tokens) == "abc"


# ---------------------------------------------------------------------------
# RLM mode streaming
# ---------------------------------------------------------------------------


class TestRLMStreaming:
    def test_rlm_step_events(self, client, monkeypatch):
        """RLM mode should emit step events for each iteration."""
        fake_llm = FakeStreamingLLM([
            '```python\nprint("hello")\n```',
            "FINAL: The answer is found",
        ])
        fake_sandbox = FakeSandbox()

        state = get_state()
        monkeypatch.setattr(state, "create_llm_adapter", lambda: fake_llm)
        monkeypatch.setattr(state, "create_sandbox", lambda: fake_sandbox)

        with client.websocket_connect("/ws/chat/s1") as ws:
            ws.receive_json()  # connected

            ws.send_json({
                "type": "query",
                "id": "msg-rlm",
                "query": "Find the answer",
                "content": "some long document",
                "mode": "rlm",
            })

            events: list[dict] = []
            while True:
                msg = ws.receive_json()
                events.append(msg)
                if msg["type"] in ("complete", "error"):
                    break

            types = [e["type"] for e in events]

            # Should have step events (one per RLM iteration)
            step_events = [e for e in events if e["type"] == "step"]
            assert len(step_events) >= 1, f"Expected step events, got types: {types}"

            # Should have metrics events
            metrics_events = [e for e in events if e["type"] == "metrics"]
            assert len(metrics_events) >= 1

            # Complete event should have the final answer
            complete = events[-1]
            assert complete["type"] == "complete"
            assert "The answer is found" in complete["data"]["answer"]
            assert complete["data"]["success"] is True

    def test_rlm_metrics_contain_running_totals(self, client, monkeypatch):
        fake_llm = FakeStreamingLLM(["FINAL: done"])
        fake_sandbox = FakeSandbox()

        state = get_state()
        monkeypatch.setattr(state, "create_llm_adapter", lambda: fake_llm)
        monkeypatch.setattr(state, "create_sandbox", lambda: fake_sandbox)

        with client.websocket_connect("/ws/chat/s1") as ws:
            ws.receive_json()

            ws.send_json({
                "type": "query",
                "id": "msg-m",
                "query": "test",
                "content": "text",
                "mode": "rlm",
            })

            metrics_event = None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "metrics":
                    metrics_event = msg
                elif msg["type"] in ("complete", "error"):
                    break

            assert metrics_event is not None
            data = metrics_event["data"]
            assert "total_tokens" in data
            assert "steps" in data
            assert "elapsed_seconds" in data


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestWSErrorHandling:
    def test_error_during_streaming(self, client, monkeypatch):
        """If the LLM raises an exception, an error event should be sent."""

        class FailingLLM:
            active_model = "fail"

            def complete(self, messages):
                raise RuntimeError("LLM exploded")

            async def complete_async(self, messages):
                raise RuntimeError("LLM exploded")

            async def complete_stream_async(self, messages):
                raise RuntimeError("LLM exploded")
                yield  # noqa: unreachable - makes this an async generator

            def count_tokens(self, text):
                return 1

            def get_pricing(self):
                return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

            def use_root_model(self):
                pass

            def use_recursive_model(self):
                pass

        state = get_state()
        monkeypatch.setattr(state, "create_llm_adapter", lambda: FailingLLM())

        with client.websocket_connect("/ws/chat/s1") as ws:
            ws.receive_json()  # connected

            ws.send_json({
                "type": "query",
                "id": "msg-err",
                "query": "will fail",
                "content": "text",
                "mode": "direct",
            })

            # Should get either an error event or a complete with success=False
            msg = ws.receive_json()
            # The use case catches exceptions and returns RunResultDTO with success=False
            if msg["type"] == "complete":
                assert msg["data"]["success"] is False
            elif msg["type"] == "error":
                assert msg["data"]["code"] == "INTERNAL_ERROR"
            else:
                # Might get token events before error, drain them
                while msg["type"] not in ("complete", "error"):
                    msg = ws.receive_json()
                assert msg["type"] in ("complete", "error")

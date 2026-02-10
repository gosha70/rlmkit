"""E2E tests for WebSocket streaming endpoints.

All tests are skipped until the WebSocket backend (Bet 2.3) is implemented.
Each test documents the expected streaming protocol so the backend team
can enable tests as features land.
"""

import pytest


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skip(reason="Waiting for WebSocket streaming (Bet 2.3)"),
]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


class TestWebSocketConnection:
    """WebSocket connect/disconnect lifecycle."""

    def test_websocket_connect(self, ws_base_url):
        """Client can establish a WebSocket connection."""
        # async with websockets.connect(f"{ws_base_url}/stream") as ws:
        #     # Server sends an initial ack
        #     msg = await ws.recv()
        #     data = json.loads(msg)
        #     assert data["type"] == "connected"
        #     assert "session_id" in data


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreamingTokens:
    """Real-time token streaming over WebSocket."""

    def test_streaming_tokens(self, ws_base_url, sample_document):
        """Submitting a query streams token chunks back in real time."""
        # async with websockets.connect(f"{ws_base_url}/stream") as ws:
        #     await ws.send(json.dumps({
        #         "type": "query",
        #         "content": sample_document,
        #         "question": "Summarize this.",
        #         "mode": "direct",
        #     }))
        #
        #     chunks = []
        #     async for msg in ws:
        #         data = json.loads(msg)
        #         if data["type"] == "token":
        #             chunks.append(data["text"])
        #         elif data["type"] == "done":
        #             break
        #
        #     assert len(chunks) > 0
        #     full_text = "".join(chunks)
        #     assert len(full_text) > 0

    def test_streaming_steps(self, ws_base_url, large_document):
        """RLM mode streams step-level progress events."""
        # async with websockets.connect(f"{ws_base_url}/stream") as ws:
        #     await ws.send(json.dumps({
        #         "type": "query",
        #         "content": large_document,
        #         "question": "What are the key topics?",
        #         "mode": "rlm",
        #     }))
        #
        #     step_events = []
        #     async for msg in ws:
        #         data = json.loads(msg)
        #         if data["type"] == "step":
        #             step_events.append(data)
        #         elif data["type"] == "done":
        #             break
        #
        #     assert len(step_events) >= 1
        #     assert all("step_index" in s for s in step_events)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancelExecution:
    """Client-initiated execution cancellation over WebSocket."""

    def test_cancel_execution(self, ws_base_url, large_document):
        """Sending a cancel message stops execution mid-stream."""
        # async with websockets.connect(f"{ws_base_url}/stream") as ws:
        #     await ws.send(json.dumps({
        #         "type": "query",
        #         "content": large_document,
        #         "question": "Analyze everything.",
        #         "mode": "rlm",
        #     }))
        #
        #     # Wait for at least one chunk
        #     msg = await ws.recv()
        #     assert json.loads(msg)["type"] in ("token", "step")
        #
        #     # Send cancel
        #     await ws.send(json.dumps({"type": "cancel"}))
        #
        #     # Server acknowledges cancellation
        #     final = await ws.recv()
        #     data = json.loads(final)
        #     assert data["type"] == "cancelled"


# ---------------------------------------------------------------------------
# Reconnection
# ---------------------------------------------------------------------------


class TestReconnect:
    """WebSocket reconnection after disconnect."""

    def test_reconnect_on_disconnect(self, ws_base_url):
        """Client can reconnect and resume after a dropped connection."""
        # async with websockets.connect(f"{ws_base_url}/stream") as ws:
        #     msg = await ws.recv()
        #     session_id = json.loads(msg)["session_id"]
        #
        # # Reconnect with the same session
        # async with websockets.connect(
        #     f"{ws_base_url}/stream?session_id={session_id}"
        # ) as ws:
        #     msg = await ws.recv()
        #     data = json.loads(msg)
        #     assert data["type"] == "connected"
        #     assert data["session_id"] == session_id

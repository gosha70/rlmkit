"""Focused tests for context_window discovery, clamping, and re-discovery.

Covers:
  1. Provider update re-discovers context_window when model/endpoint changes.
  2. Local UI ChatManager._build_rlm_adapter clamps max_tokens to 75% of
     discovered context_window.
  3. OllamaClient accepts but does NOT dynamically clamp per-call — documents
     the gap.

All external calls (litellm, urllib, Ollama HTTP) are mocked.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from rlmstudio.server.app import app
from rlmstudio.server.dependencies import reset_state
from rlmstudio.ui.data.providers_catalog import PROVIDERS_BY_KEY

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    """Reset in-memory state before and after each test."""
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def valid_provider_key() -> str:
    return str(next(iter(PROVIDERS_BY_KEY)))


@pytest.fixture
def valid_model(valid_provider_key: str) -> str:
    entry = PROVIDERS_BY_KEY[valid_provider_key]
    if entry.models:
        return str(entry.models[0].name)
    return "test-model"


# ---------------------------------------------------------------------------
# 1. Provider update re-discovers context_window
# ---------------------------------------------------------------------------


class TestProviderUpdateRediscovery:
    """PUT /api/llm-providers/{id} re-discovers context_window when
    model or endpoint changes, unless an explicit override is given."""

    def _create_provider(
        self,
        client: TestClient,
        backend: str,
        model: str,
        *,
        context_window: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": f"test-{uuid.uuid4().hex[:6]}",
            "backend": backend,
            "model": model,
        }
        if context_window is not None:
            payload["context_window"] = context_window
        resp = client.post("/api/llm-providers", json=payload)
        assert resp.status_code == 201, resp.text
        result: dict[str, Any] = resp.json()
        return result

    @patch("rlmstudio.server.routes.llm_providers._discover_context_window")
    def test_model_change_triggers_rediscovery(
        self, mock_discover: MagicMock, client: TestClient, valid_provider_key: str
    ) -> None:
        """Changing the model should re-discover context_window."""
        # Create with explicit context_window so discovery is not called on create
        mock_discover.return_value = None
        lp = self._create_provider(client, valid_provider_key, "old-model", context_window=4096)
        assert lp["context_window"] == 4096
        mock_discover.reset_mock()

        # Update model — should trigger re-discovery
        mock_discover.return_value = 32768
        resp = client.put(
            f"/api/llm-providers/{lp['id']}",
            json={"model": "new-big-model"},
        )
        assert resp.status_code == 200
        assert resp.json()["context_window"] == 32768
        mock_discover.assert_called_once()

    @patch("rlmstudio.server.routes.llm_providers._discover_context_window")
    def test_endpoint_change_triggers_rediscovery(
        self, mock_discover: MagicMock, client: TestClient, valid_provider_key: str
    ) -> None:
        """Changing the endpoint should re-discover context_window."""
        mock_discover.return_value = None
        lp = self._create_provider(client, valid_provider_key, "some-model", context_window=8192)
        mock_discover.reset_mock()

        mock_discover.return_value = 16384
        resp = client.put(
            f"/api/llm-providers/{lp['id']}",
            json={"endpoint": "http://new-host:8000"},
        )
        assert resp.status_code == 200
        assert resp.json()["context_window"] == 16384

    @patch("rlmstudio.server.routes.llm_providers._discover_context_window")
    def test_explicit_context_window_overrides_rediscovery(
        self, mock_discover: MagicMock, client: TestClient, valid_provider_key: str
    ) -> None:
        """If the user passes context_window explicitly, discovery is skipped."""
        mock_discover.return_value = None
        lp = self._create_provider(client, valid_provider_key, "a-model", context_window=4096)
        mock_discover.reset_mock()

        resp = client.put(
            f"/api/llm-providers/{lp['id']}",
            json={"model": "another-model", "context_window": 65536},
        )
        assert resp.status_code == 200
        assert resp.json()["context_window"] == 65536
        # Discovery should NOT have been called — explicit override wins
        mock_discover.assert_not_called()

    @patch("rlmstudio.server.routes.llm_providers._discover_context_window")
    def test_no_rediscovery_when_model_unchanged(
        self, mock_discover: MagicMock, client: TestClient, valid_provider_key: str
    ) -> None:
        """Updating name-only should NOT trigger re-discovery."""
        mock_discover.return_value = None
        lp = self._create_provider(client, valid_provider_key, "stable-model", context_window=8192)
        mock_discover.reset_mock()

        resp = client.put(
            f"/api/llm-providers/{lp['id']}",
            json={"name": "Renamed-Provider"},
        )
        assert resp.status_code == 200
        # context_window stays untouched
        assert resp.json()["context_window"] == 8192
        mock_discover.assert_not_called()

    @patch("rlmstudio.server.routes.llm_providers._discover_context_window")
    def test_rediscovery_failure_clears_stale_context_window(
        self, mock_discover: MagicMock, client: TestClient, valid_provider_key: str
    ) -> None:
        """If re-discovery returns None, context_window should be cleared
        so a stale value doesn't persist."""
        mock_discover.return_value = None
        lp = self._create_provider(client, valid_provider_key, "model-v1", context_window=8192)
        mock_discover.reset_mock()

        mock_discover.return_value = None  # discovery fails
        resp = client.put(
            f"/api/llm-providers/{lp['id']}",
            json={"model": "model-v2"},
        )
        assert resp.status_code == 200
        assert resp.json()["context_window"] is None


# ---------------------------------------------------------------------------
# 2. Local UI max-token clamping (ChatManager._build_rlm_adapter)
# ---------------------------------------------------------------------------


class TestLocalUIContextWindowClamping:
    """ChatManager._build_rlm_adapter should clamp max_tokens to 75% of
    discovered context_window before constructing the legacy client."""

    def _make_provider_config(
        self,
        *,
        provider: str = "vllm",
        model: str = "qwen2-7b",
        api_endpoint: str = "http://localhost:8000",
        max_tokens: int = 4096,
    ) -> Any:
        """Build a minimal LLMProviderConfig-like object for the local UI path."""
        from rlmstudio.ui.services.models import LLMProviderConfig as UIProviderConfig

        return UIProviderConfig(
            provider=provider,
            model=model,
            api_endpoint=api_endpoint,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=1.0,
            api_key="",
        )

    @patch("rlmstudio.ui.services.chat_manager.get_llm_client")
    @patch.object(
        __import__("rlmstudio.ui.services.chat_manager", fromlist=["ChatManager"]).ChatManager,
        "_discover_context_window",
    )
    def test_clamps_max_tokens_when_context_window_discovered(
        self, mock_discover: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """When context_window is discovered, max_tokens should be capped to 75%."""
        from rlmstudio.ui.services.chat_manager import ChatManager

        mock_discover.return_value = 8192
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mgr = ChatManager(session_state={"default_max_tokens": 4096})
        provider_cfg = self._make_provider_config(max_tokens=4096)

        mgr._build_rlm_adapter(provider_cfg, "fake-key")

        # get_llm_client should have been called with clamped max_tokens
        call_kwargs = mock_get_client.call_args
        actual_max = call_kwargs.kwargs.get("max_tokens") or call_kwargs[1].get("max_tokens")
        # 4096 < int(8192 * 0.75) = 6144, so no clamping needed — max_tokens stays 4096
        assert actual_max == 4096

    @patch("rlmstudio.ui.services.chat_manager.get_llm_client")
    @patch.object(
        __import__("rlmstudio.ui.services.chat_manager", fromlist=["ChatManager"]).ChatManager,
        "_discover_context_window",
    )
    def test_clamps_when_max_tokens_exceeds_safe_limit(
        self, mock_discover: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """When max_tokens > 75% of context_window, it should be reduced."""
        from rlmstudio.ui.services.chat_manager import ChatManager

        mock_discover.return_value = 8192  # 75% = 6144
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Request 7000 output tokens — exceeds 6144
        mgr = ChatManager(session_state={"default_max_tokens": 7000})
        provider_cfg = self._make_provider_config(max_tokens=7000)

        mgr._build_rlm_adapter(provider_cfg, "fake-key")

        call_kwargs = mock_get_client.call_args
        actual_max = call_kwargs.kwargs.get("max_tokens") or call_kwargs[1].get("max_tokens")
        assert actual_max == int(8192 * 0.75)  # 6144

    @patch("rlmstudio.ui.services.chat_manager.get_llm_client")
    @patch.object(
        __import__("rlmstudio.ui.services.chat_manager", fromlist=["ChatManager"]).ChatManager,
        "_discover_context_window",
    )
    def test_no_clamping_when_discovery_fails(
        self, mock_discover: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """When context_window cannot be discovered, max_tokens passes through."""
        from rlmstudio.ui.services.chat_manager import ChatManager

        mock_discover.return_value = None
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mgr = ChatManager(session_state={"default_max_tokens": 4096})
        provider_cfg = self._make_provider_config(max_tokens=4096)

        mgr._build_rlm_adapter(provider_cfg, "fake-key")

        call_kwargs = mock_get_client.call_args
        actual_max = call_kwargs.kwargs.get("max_tokens") or call_kwargs[1].get("max_tokens")
        assert actual_max == 4096

    @patch("rlmstudio.ui.services.chat_manager.get_llm_client")
    @patch.object(
        __import__("rlmstudio.ui.services.chat_manager", fromlist=["ChatManager"]).ChatManager,
        "_discover_context_window",
    )
    def test_session_state_max_tokens_overrides_provider(
        self, mock_discover: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """session_state['default_max_tokens'] takes precedence over provider config,
        but still gets clamped by context_window."""
        from rlmstudio.ui.services.chat_manager import ChatManager

        mock_discover.return_value = 4096  # 75% = 3072
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Provider says 2048, but session_state says 4000
        mgr = ChatManager(session_state={"default_max_tokens": 4000})
        provider_cfg = self._make_provider_config(max_tokens=2048)

        mgr._build_rlm_adapter(provider_cfg, "fake-key")

        call_kwargs = mock_get_client.call_args
        actual_max = call_kwargs.kwargs.get("max_tokens") or call_kwargs[1].get("max_tokens")
        # 4000 > 3072, so clamped to 3072
        assert actual_max == int(4096 * 0.75)  # 3072


# ---------------------------------------------------------------------------
# 3. OllamaClient gap documentation
# ---------------------------------------------------------------------------


class TestOllamaClientStaticMaxTokens:
    """Documents that OllamaClient only honours the static max_tokens
    set at construction — it has no per-call context-aware clamping.

    This is a known gap:  the local UI path applies a 75% static cap
    before creating OllamaClient, which is enough to avoid hard failures
    but does not dynamically shrink max_tokens as the conversation grows.
    """

    @patch("rlmstudio.llm.ollama_client.requests.get")
    @patch("rlmstudio.llm.ollama_client.requests.post")
    def test_ollama_uses_static_max_tokens(self, mock_post: MagicMock, mock_get: MagicMock) -> None:
        """OllamaClient passes the constructor max_tokens to every call unchanged."""
        from rlmstudio.llm.ollama_client import OllamaClient

        # Mock connection check
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": {"content": "hi"}, "done": True},
        )

        client = OllamaClient(model="qwen2:7b", max_tokens=2048)

        # First call with a short prompt
        client.complete([{"role": "user", "content": "hello"}])
        payload_1 = mock_post.call_args[1]["json"]
        assert payload_1["options"]["num_predict"] == 2048

        # Second call with a much longer prompt — still 2048
        big_prompt = "x" * 30000
        client.complete([{"role": "user", "content": big_prompt}])
        payload_2 = mock_post.call_args[1]["json"]
        assert payload_2["options"]["num_predict"] == 2048  # no dynamic clamping

    @patch("rlmstudio.llm.ollama_client.requests.get")
    @patch("rlmstudio.llm.ollama_client.requests.post")
    def test_ollama_no_num_predict_when_max_tokens_none(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        """When max_tokens is None, num_predict should not appear in options."""
        from rlmstudio.llm.ollama_client import OllamaClient

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"models": []})
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": {"content": "hi"}, "done": True},
        )

        client = OllamaClient(model="llama3", max_tokens=None)
        client.complete([{"role": "user", "content": "hello"}])
        payload = mock_post.call_args[1]["json"]
        assert "num_predict" not in payload["options"]


# ---------------------------------------------------------------------------
# 4. Discovery URL normalization and model ID matching
# ---------------------------------------------------------------------------


class TestDiscoveryURLAndModelMatching:
    """Verify _discover_context_window handles:
    - Endpoints that already end with /v1 (no double /v1)
    - LiteLLM-prefixed model names (openai/Qwen/... → Qwen/...)
    """

    def _mock_urlopen(self, model_id: str, max_model_len: int):
        """Build a mock for urllib.request.urlopen that returns a vLLM-style response."""
        import json

        body = json.dumps({"data": [{"id": model_id, "max_model_len": max_model_len}]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("litellm.get_model_info", side_effect=Exception("skip"))
    @patch("urllib.request.urlopen")
    def test_endpoint_with_v1_suffix_no_doubling(
        self, mock_urlopen: MagicMock, mock_litellm: MagicMock
    ) -> None:
        """Endpoint http://host:8000/v1 should query http://host:8000/v1/models, not /v1/v1/models."""
        from rlmstudio.server.routes.llm_providers import _discover_context_window

        mock_urlopen.return_value = self._mock_urlopen("Qwen/Qwen2.5-7B-Instruct", 8192)

        result = _discover_context_window(
            backend="vllm",
            model="Qwen/Qwen2.5-7B-Instruct",
            endpoint="http://192.168.1.23:8000/v1",
        )
        assert result == 8192
        # Verify the URL used
        call_args = mock_urlopen.call_args
        req_obj = call_args[0][0]
        assert req_obj.full_url == "http://192.168.1.23:8000/v1/models"

    @patch("litellm.get_model_info", side_effect=Exception("skip"))
    @patch("urllib.request.urlopen")
    def test_endpoint_without_v1_suffix(
        self, mock_urlopen: MagicMock, mock_litellm: MagicMock
    ) -> None:
        """Endpoint http://host:8000 should query http://host:8000/v1/models."""
        from rlmstudio.server.routes.llm_providers import _discover_context_window

        mock_urlopen.return_value = self._mock_urlopen("some-model", 4096)

        result = _discover_context_window(
            backend="vllm",
            model="some-model",
            endpoint="http://192.168.1.23:8000",
        )
        assert result == 4096
        req_obj = mock_urlopen.call_args[0][0]
        assert req_obj.full_url == "http://192.168.1.23:8000/v1/models"

    @patch("litellm.get_model_info", side_effect=Exception("skip"))
    @patch("urllib.request.urlopen")
    def test_litellm_prefix_stripped_for_matching(
        self, mock_urlopen: MagicMock, mock_litellm: MagicMock
    ) -> None:
        """Model 'openai/Qwen/Qwen2.5-7B-Instruct' should match vLLM id 'Qwen/Qwen2.5-7B-Instruct'."""
        from rlmstudio.server.routes.llm_providers import _discover_context_window

        # vLLM reports model as "Qwen/Qwen2.5-7B-Instruct" (no openai/ prefix)
        mock_urlopen.return_value = self._mock_urlopen("Qwen/Qwen2.5-7B-Instruct", 8192)

        result = _discover_context_window(
            backend="vllm",
            model="openai/Qwen/Qwen2.5-7B-Instruct",
            endpoint="http://192.168.1.23:8000/v1",
        )
        assert result == 8192

    @patch("litellm.get_model_info", side_effect=Exception("skip"))
    @patch("urllib.request.urlopen")
    def test_exact_match_still_works(
        self, mock_urlopen: MagicMock, mock_litellm: MagicMock
    ) -> None:
        """When model name matches exactly, discovery should still work."""
        from rlmstudio.server.routes.llm_providers import _discover_context_window

        mock_urlopen.return_value = self._mock_urlopen("llama3.2", 131072)

        result = _discover_context_window(
            backend="ollama",
            model="llama3.2",
            endpoint="http://localhost:11434",
        )
        assert result == 131072

    @patch("litellm.get_model_info", side_effect=Exception("skip"))
    @patch("urllib.request.urlopen")
    def test_no_match_returns_none(self, mock_urlopen: MagicMock, mock_litellm: MagicMock) -> None:
        """When model doesn't match any reported ID, return None."""
        from rlmstudio.server.routes.llm_providers import _discover_context_window

        mock_urlopen.return_value = self._mock_urlopen("completely-different-model", 8192)

        result = _discover_context_window(
            backend="vllm",
            model="openai/Qwen/Qwen2.5-7B-Instruct",
            endpoint="http://host:8000",
        )
        assert result is None

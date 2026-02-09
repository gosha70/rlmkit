"""Tests for the public client: backward compatibility and high-level API.

Verifies that RLMKitClient wires up adapters correctly and provides
a stable interface for end users.
"""

import pytest

from rlmkit.public.client import RLMKitClient
from rlmkit.public.types import PublicRunResult, PublicInteractResult
from rlmkit.public.errors import (
    BudgetError,
    ConfigError,
    ProviderError,
    RLMKitError,
    SandboxError,
    wrap_domain_error,
)
from rlmkit.domain.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    ExecutionFailedError,
    SecurityViolationError,
    DomainError,
)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


class TestRLMKitClientConstruction:
    """Test client initialization with various configurations."""

    def test_default_construction(self):
        client = RLMKitClient()
        assert client._provider == "mock"

    def test_mock_provider(self):
        client = RLMKitClient(provider="mock")
        assert type(client._llm).__name__ == "MockLLMAdapter"

    def test_litellm_provider(self):
        client = RLMKitClient(provider="litellm", model="gpt-4o")
        assert type(client._llm).__name__ == "LiteLLMAdapter"
        assert client._llm.active_model == "gpt-4o"

    def test_litellm_two_model(self):
        client = RLMKitClient(
            provider="litellm",
            model="gpt-4o",
            root_model="gpt-4o",
            recursive_model="gpt-4o-mini",
        )
        assert client._llm.is_two_model is True

    def test_unknown_provider_raises(self):
        with pytest.raises(ConfigError, match="Unknown provider"):
            RLMKitClient(provider="nonexistent")

    def test_sandbox_default_is_local(self):
        client = RLMKitClient()
        from rlmkit.infrastructure.sandbox.local_sandbox import LocalSandboxAdapter
        assert isinstance(client._sandbox, LocalSandboxAdapter)


# ---------------------------------------------------------------------------
# interact() method
# ---------------------------------------------------------------------------


class TestRLMKitClientInteract:
    """Test the interact() high-level API."""

    def test_direct_mode(self):
        client = RLMKitClient(provider="mock")
        result = client.interact("document text", "question", mode="direct")

        assert isinstance(result, PublicRunResult)
        assert result.success is True
        assert result.mode_used == "direct"
        assert result.answer != ""

    def test_rlm_mode(self):
        client = RLMKitClient(provider="mock")
        result = client.interact("document text", "question", mode="rlm")

        assert isinstance(result, PublicRunResult)
        assert result.success is True
        assert result.mode_used == "rlm"

    def test_auto_mode_short_content(self):
        client = RLMKitClient(provider="mock")
        # Short content (<8000 tokens ~32000 chars) -> direct
        result = client.interact("short", "question", mode="auto")
        assert result.success is True
        assert result.mode_used == "direct"

    def test_auto_mode_long_content(self):
        client = RLMKitClient(provider="mock")
        # Long content (>8000 tokens ~32000 chars) -> rlm
        long_text = "x" * 40000
        result = client.interact(long_text, "question", mode="auto")
        assert result.success is True
        assert result.mode_used == "rlm"

    def test_compare_mode(self):
        client = RLMKitClient(provider="mock")
        result = client.interact("document", "question", mode="compare")
        assert result.success is True

    def test_empty_content_raises(self):
        client = RLMKitClient(provider="mock")
        with pytest.raises(ValueError, match="content cannot be empty"):
            client.interact("", "question")

    def test_empty_query_raises(self):
        client = RLMKitClient(provider="mock")
        with pytest.raises(ValueError, match="query cannot be empty"):
            client.interact("content", "")

    def test_unsupported_mode_raises(self):
        client = RLMKitClient(provider="mock")
        with pytest.raises(ValueError, match="Unsupported mode"):
            client.interact("content", "question", mode="invalid_mode")


# ---------------------------------------------------------------------------
# complete() method
# ---------------------------------------------------------------------------


class TestRLMKitClientComplete:
    """Test the simplified complete() API."""

    def test_returns_string(self):
        client = RLMKitClient(provider="mock")
        answer = client.complete("document", "question")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_passes_kwargs(self):
        client = RLMKitClient(provider="mock")
        answer = client.complete("document", "question", mode="direct")
        assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# Auto-mode logic
# ---------------------------------------------------------------------------


class TestAutoMode:
    """Test automatic mode selection based on content size."""

    def test_short_content_selects_direct(self):
        assert RLMKitClient._determine_auto_mode("short text") == "direct"

    def test_medium_content_selects_rlm(self):
        # > 8000 tokens = > 32000 chars
        content = "x" * 40000
        assert RLMKitClient._determine_auto_mode(content) == "rlm"

    def test_very_long_content_selects_rlm(self):
        content = "x" * 500000
        assert RLMKitClient._determine_auto_mode(content) == "rlm"


# ---------------------------------------------------------------------------
# PublicRunResult
# ---------------------------------------------------------------------------


class TestPublicRunResult:
    """Tests for the public result type."""

    def test_str_returns_answer(self):
        r = PublicRunResult(answer="42")
        assert str(r) == "42"

    def test_to_dict(self):
        r = PublicRunResult(
            answer="42",
            mode_used="direct",
            success=True,
            total_tokens=100,
        )
        d = r.to_dict()
        assert d["answer"] == "42"
        assert d["mode_used"] == "direct"
        assert d["total_tokens"] == 100
        assert d["has_trace"] is False

    def test_to_dict_with_trace(self):
        r = PublicRunResult(answer="42", trace=[{"step": 0}])
        d = r.to_dict()
        assert d["has_trace"] is True


# ---------------------------------------------------------------------------
# PublicInteractResult
# ---------------------------------------------------------------------------


class TestPublicInteractResult:
    """Tests for the backward-compatible result type."""

    def test_str_returns_answer(self):
        r = PublicInteractResult(answer="hello", mode_used="direct")
        assert str(r) == "hello"

    def test_to_dict(self):
        r = PublicInteractResult(answer="hello", mode_used="rlm", metrics={"tokens": 100})
        d = r.to_dict()
        assert d["answer"] == "hello"
        assert d["mode_used"] == "rlm"
        assert d["metrics"]["tokens"] == 100


# ---------------------------------------------------------------------------
# Public error wrapping
# ---------------------------------------------------------------------------


class TestPublicErrors:
    """Tests for the public error types and wrap_domain_error."""

    def test_rlmkit_error_hierarchy(self):
        assert issubclass(ProviderError, RLMKitError)
        assert issubclass(BudgetError, RLMKitError)
        assert issubclass(SandboxError, RLMKitError)
        assert issubclass(ConfigError, RLMKitError)

    def test_wrap_budget_exceeded(self):
        exc = wrap_domain_error(BudgetExceededError("max steps"))
        assert isinstance(exc, BudgetError)
        assert "max steps" in str(exc)

    def test_wrap_execution_failed(self):
        exc = wrap_domain_error(ExecutionFailedError("code crashed"))
        assert isinstance(exc, SandboxError)

    def test_wrap_security_violation(self):
        exc = wrap_domain_error(SecurityViolationError("blocked import"))
        assert isinstance(exc, SandboxError)

    def test_wrap_configuration_error(self):
        exc = wrap_domain_error(ConfigurationError("bad config"))
        assert isinstance(exc, ConfigError)

    def test_wrap_generic_domain_error(self):
        exc = wrap_domain_error(DomainError("generic"))
        assert isinstance(exc, RLMKitError)

    def test_wrap_non_domain_exception(self):
        exc = wrap_domain_error(RuntimeError("unknown"))
        assert isinstance(exc, RLMKitError)

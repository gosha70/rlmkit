"""Tests for prompt_template_name on profiles and _resolve_profile_prompt.

Covers the new features introduced across the last 12 commits:
  1. Profile CRUD accepts and persists prompt_template_name
  2. _resolve_profile_prompt resolves template names at runtime
  3. Backward compatibility: custom system_prompts still take priority
  4. System prompt templates JSON is well-formed
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.app import create_app
from rlmkit.server.dependencies import get_state, reset_state
from rlmkit.server.models import RunProfile
from rlmkit.server.routes.chat import _resolve_profile_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    """Reset shared state before each test."""
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def _make_profile(
    *,
    system_prompts: dict[str, str] | None = None,
    prompt_template_name: str | None = None,
) -> RunProfile:
    """Build a minimal RunProfile for unit tests."""
    return RunProfile(
        id="test-profile",
        name="Test",
        strategy="rlm",
        system_prompts=system_prompts or {},
        prompt_template_name=prompt_template_name,
    )


# ---------------------------------------------------------------------------
# _resolve_profile_prompt — unit tests
# ---------------------------------------------------------------------------


class TestResolveProfilePrompt:
    """Unit tests for the chat route's prompt resolution helper."""

    def test_returns_none_when_no_prompts_and_no_template(self) -> None:
        profile = _make_profile()
        assert _resolve_profile_prompt(profile, "rlm") is None

    def test_returns_custom_system_prompt_when_set(self) -> None:
        profile = _make_profile(system_prompts={"rlm": "Custom RLM prompt"})
        assert _resolve_profile_prompt(profile, "rlm") == "Custom RLM prompt"

    def test_custom_prompt_takes_priority_over_template(self) -> None:
        """When both system_prompts and prompt_template_name are set,
        the explicit custom text wins."""
        profile = _make_profile(
            system_prompts={"rlm": "My custom override"},
            prompt_template_name="Default",
        )
        assert _resolve_profile_prompt(profile, "rlm") == "My custom override"

    def test_resolves_named_template_when_no_custom_prompt(self) -> None:
        profile = _make_profile(prompt_template_name="Default")
        result = _resolve_profile_prompt(profile, "rlm")
        assert result is not None
        assert len(result) > 0
        assert "Additional guidance" in result

    def test_resolves_concise_analyst_template(self) -> None:
        profile = _make_profile(prompt_template_name="Concise analyst")
        result = _resolve_profile_prompt(profile, "rlm")
        assert result is not None
        assert "fewest steps" in result

    def test_resolves_detailed_explainer_template(self) -> None:
        profile = _make_profile(prompt_template_name="Detailed explainer")
        result = _resolve_profile_prompt(profile, "rlm")
        assert result is not None
        assert "reasoning chain" in result

    def test_returns_none_for_unknown_template_name(self) -> None:
        profile = _make_profile(prompt_template_name="Nonexistent Template")
        assert _resolve_profile_prompt(profile, "rlm") is None

    def test_resolves_direct_mode_from_template(self) -> None:
        profile = _make_profile(prompt_template_name="Default")
        result = _resolve_profile_prompt(profile, "direct")
        assert result is not None
        assert "helpful" in result.lower()

    def test_resolves_rag_mode_from_template(self) -> None:
        profile = _make_profile(prompt_template_name="Default")
        result = _resolve_profile_prompt(profile, "rag")
        assert result is not None
        assert "context" in result.lower()

    def test_returns_none_for_missing_mode_in_template(self) -> None:
        """If the template exists but doesn't have the requested mode key."""
        mock_templates = {"TestTemplate": {"rlm": "only rlm"}}
        with patch(
            "rlmkit.server.routes.chat.SYSTEM_PROMPT_TEMPLATES",
            mock_templates,
            create=True,
        ):
            # Patch the import inside the function
            with patch(
                "rlmkit.ui.services.profile_store.SYSTEM_PROMPT_TEMPLATES",
                mock_templates,
            ):
                profile = _make_profile(prompt_template_name="TestTemplate")
                assert _resolve_profile_prompt(profile, "direct") is None

    def test_empty_string_custom_prompt_falls_through_to_template(self) -> None:
        """An empty string in system_prompts should NOT be treated as a
        custom override — the resolver should fall through to the template."""
        profile = _make_profile(
            system_prompts={"rlm": ""},
            prompt_template_name="Default",
        )
        result = _resolve_profile_prompt(profile, "rlm")
        assert result is not None
        assert "Additional guidance" in result


# ---------------------------------------------------------------------------
# Profile CRUD — prompt_template_name via API
# ---------------------------------------------------------------------------


class TestProfileCRUDWithTemplateName:
    """Integration tests for prompt_template_name in the profile API."""

    def test_create_profile_with_template_name(self, client: TestClient) -> None:
        resp = client.post(
            "/api/profiles",
            json={
                "name": "Template Profile",
                "strategy": "rlm",
                "prompt_template_name": "Default",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["prompt_template_name"] == "Default"

    def test_create_profile_without_template_name_defaults_to_null(
        self, client: TestClient
    ) -> None:
        resp = client.post(
            "/api/profiles",
            json={"name": "No Template", "strategy": "direct"},
        )
        assert resp.status_code == 201
        assert resp.json()["prompt_template_name"] is None

    def test_update_profile_sets_template_name(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/profiles",
            json={"name": "Updatable", "strategy": "direct"},
        )
        pid = create_resp.json()["id"]
        resp = client.put(
            f"/api/profiles/{pid}",
            json={"prompt_template_name": "Concise analyst"},
        )
        assert resp.status_code == 200
        assert resp.json()["prompt_template_name"] == "Concise analyst"

    def test_update_profile_clears_template_name(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/profiles",
            json={
                "name": "Will Clear",
                "strategy": "rlm",
                "prompt_template_name": "Default",
            },
        )
        pid = create_resp.json()["id"]
        # Setting to empty string clears it
        resp = client.put(
            f"/api/profiles/{pid}",
            json={"prompt_template_name": ""},
        )
        assert resp.status_code == 200
        # Empty string is stored as-is; the resolver treats it as falsy
        assert resp.json()["prompt_template_name"] == ""

    def test_template_name_persisted_across_get(self, client: TestClient) -> None:
        """Create → GET listing → verify template_name is returned."""
        client.post(
            "/api/profiles",
            json={
                "name": "Persisted",
                "strategy": "rlm",
                "prompt_template_name": "Detailed explainer",
            },
        )
        listing = client.get("/api/profiles").json()
        user_profiles = [p for p in listing if p["name"] == "Persisted"]
        assert len(user_profiles) == 1
        assert user_profiles[0]["prompt_template_name"] == "Detailed explainer"

    def test_builtin_profiles_have_template_name_field(
        self, client: TestClient
    ) -> None:
        """All profiles (including builtins) should include the field."""
        listing = client.get("/api/profiles").json()
        for profile in listing:
            assert "prompt_template_name" in profile


# ---------------------------------------------------------------------------
# System prompt templates — structural validation
# ---------------------------------------------------------------------------


class TestSystemPromptTemplates:
    """Validate the templates JSON loaded at runtime."""

    def test_templates_endpoint_returns_list(self, client: TestClient) -> None:
        resp = client.get("/api/system-prompts/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 3

    def test_each_template_has_required_modes(self, client: TestClient) -> None:
        resp = client.get("/api/system-prompts/templates")
        for tpl in resp.json():
            assert "name" in tpl
            assert "prompts" in tpl
            prompts = tpl["prompts"]
            for mode in ("direct", "rlm", "rag"):
                assert mode in prompts, f"Template '{tpl['name']}' missing '{mode}' mode"
                assert len(prompts[mode]) > 0

    def test_rlm_templates_are_additive_supplements(self, client: TestClient) -> None:
        """RLM template entries should be short additive supplements,
        not full system prompts that conflict with the base v2.0 protocol."""
        resp = client.get("/api/system-prompts/templates")
        for tpl in resp.json():
            rlm_text = tpl["prompts"]["rlm"]
            # Should start with "Additional guidance" framing
            assert "Additional guidance" in rlm_text or len(rlm_text) < 500, (
                f"Template '{tpl['name']}' RLM text looks too long for a supplement"
            )
            # Should NOT contain conflicting instructions
            assert "use your full step budget" not in rlm_text.lower()
            assert "do not finalize early" not in rlm_text.lower()

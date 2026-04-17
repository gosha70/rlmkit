"""E2E tests for GET /api/docs/{slug}.

Verifies the allowlist enforcement, path-traversal / separator
rejection, and the success path for a known doc. Reads the real
``docs/`` tree so the assertion is an end-to-end round trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.routes.docs import _DOCS_ALLOWLIST, _DOCS_DIR

pytestmark = [pytest.mark.e2e]


class TestDocsEndpoint:
    """GET /api/docs/{slug}."""

    def test_returns_200_with_slug_and_content_for_allowlisted_doc(
        self, client: TestClient
    ) -> None:
        resp = client.get("/api/docs/rlm-concepts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "rlm-concepts"
        assert isinstance(data["content"], str)
        assert len(data["content"]) > 0

    def test_content_matches_file_on_disk(self, client: TestClient) -> None:
        expected = (_DOCS_DIR / "rlm-concepts.md").read_text(encoding="utf-8")
        data = client.get("/api/docs/rlm-concepts").json()
        assert data["content"] == expected

    def test_all_allowlisted_slugs_resolve(self, client: TestClient) -> None:
        # Every entry in the allowlist must point at a readable file.
        for slug in _DOCS_ALLOWLIST:
            resp = client.get(f"/api/docs/{slug}")
            assert resp.status_code == 200, f"slug={slug} failed: {resp.text}"
            assert resp.json()["slug"] == slug

    def test_unknown_slug_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/docs/does-not-exist")
        assert resp.status_code == 404

    def test_slug_with_dot_dot_returns_404(self, client: TestClient) -> None:
        # Traversal attempt: /api/docs/..%2F..%2Fetc%2Fpasswd decoded.
        # Starlette decodes the path param before our handler, so we
        # exercise the literal form that would slip past the router.
        resp = client.get("/api/docs/..")
        assert resp.status_code == 404

    def test_slug_with_slash_returns_404(self, client: TestClient) -> None:
        # URL-encoded slash should not match an allowlisted slug.
        resp = client.get("/api/docs/hosts%2Follama")
        assert resp.status_code == 404

    def test_slug_with_uppercase_returns_404(self, client: TestClient) -> None:
        # Slug regex is lowercase-only; protects against case-insensitive
        # filesystem collisions on macOS/Windows.
        resp = client.get("/api/docs/RLM-Concepts")
        assert resp.status_code == 404

    def test_slug_with_dot_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/docs/rlm-concepts.md")
        assert resp.status_code == 404

    def test_allowlisted_filenames_exist_on_disk(self) -> None:
        # Structural invariant — every allowlist entry must be a real file.
        for slug, filename in _DOCS_ALLOWLIST.items():
            path = _DOCS_DIR / filename
            assert path.is_file(), f"allowlist entry {slug!r} -> {filename!r} missing"

    def test_project_root_resolves_to_repo_with_docs_dir(self) -> None:
        # Guards against a regression in the parents[N] math.
        assert _DOCS_DIR.is_dir()
        assert (_DOCS_DIR / "rlm-concepts.md").is_file()

    def test_docs_root_is_not_above_repo(self) -> None:
        # Sanity: resolved docs dir does not escape to the filesystem root.
        assert str(_DOCS_DIR.resolve()) != "/"
        assert "rlmkit" in str(_DOCS_DIR.resolve()).lower()


class TestDocsPathTraversalDefense:
    """The endpoint's second line of defense: resolved-path containment."""

    def test_symlink_pointing_outside_docs_dir_is_rejected(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate a poisoned allowlist entry whose filename resolves,
        # via symlink, outside docs/. The containment check should 404.
        outside = tmp_path / "outside.md"
        outside.write_text("secret content", encoding="utf-8")

        link = _DOCS_DIR / "__test_symlink.md"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlink creation unsupported on this platform")
        try:
            # Poison the allowlist for the duration of this test.
            from rlmkit.server.routes import docs as docs_mod

            monkeypatch.setitem(docs_mod._DOCS_ALLOWLIST, "test-symlink", "__test_symlink.md")
            resp = client.get("/api/docs/test-symlink")
            assert resp.status_code == 404
        finally:
            link.unlink(missing_ok=True)

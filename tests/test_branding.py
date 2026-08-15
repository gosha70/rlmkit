"""Tests for ``rlmkit.branding`` — the single source of product identity.

These exercise the env-var and state-dir accessors, including the legacy
fallback machinery that a future rename relies on.  Legacy tuples are empty
today, so the fallback paths are driven by monkeypatching the constants.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rlmkit import branding


@pytest.fixture(autouse=True)
def _reset_branding_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate module-level caches and env between tests."""
    monkeypatch.setattr(branding, "_warned_legacy_env", set())
    monkeypatch.setattr(branding, "_migration_attempted", set())
    for key in list(branding.os.environ):
        if key.startswith(branding.ENV_PREFIX) or key.startswith("LEGACYX_"):
            monkeypatch.delenv(key, raising=False)


class TestEnv:
    def test_env_name_uses_prefix(self) -> None:
        assert branding.env_name("PORT") == f"{branding.ENV_PREFIX}PORT"

    def test_canonical_value_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(branding.env_name("PORT"), "8123")
        assert branding.env("PORT") == "8123"

    def test_default_when_unset(self) -> None:
        assert branding.env("PORT") is None
        assert branding.env("PORT", "8000") == "8000"

    def test_legacy_prefix_fallback_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(branding, "LEGACY_ENV_PREFIXES", ("LEGACYX_",))
        monkeypatch.setenv("LEGACYX_PORT", "9000")
        with caplog.at_level(logging.WARNING, logger="rlmkit.branding"):
            assert branding.env("PORT") == "9000"
            assert branding.env("PORT") == "9000"
        warnings = [r for r in caplog.records if "deprecated" in r.getMessage()]
        assert len(warnings) == 1
        assert "LEGACYX_PORT" in warnings[0].getMessage()
        assert branding.env_name("PORT") in warnings[0].getMessage()

    def test_canonical_wins_over_legacy(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(branding, "LEGACY_ENV_PREFIXES", ("LEGACYX_",))
        monkeypatch.setenv("LEGACYX_PORT", "9000")
        monkeypatch.setenv(branding.env_name("PORT"), "8123")
        with caplog.at_level(logging.WARNING, logger="rlmkit.branding"):
            assert branding.env("PORT") == "8123"
        assert not [r for r in caplog.records if "deprecated" in r.getMessage()]


class TestStateDir:
    def test_default_under_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert branding.state_dir() == tmp_path / branding.STATE_DIR_NAME
        assert not (tmp_path / branding.STATE_DIR_NAME).exists()

    def test_create_flag_makes_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        target = branding.state_dir(create=True)
        assert target.is_dir()

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        override = tmp_path / "elsewhere"
        monkeypatch.setenv(branding.env_name(branding.STATE_DIR_ENV_SUFFIX), str(override))
        assert branding.state_dir() == override

    def test_legacy_dir_is_copied_forward_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(branding, "LEGACY_STATE_DIR_NAMES", (".legacyx",))
        legacy = tmp_path / ".legacyx"
        (legacy / "sub").mkdir(parents=True)
        (legacy / "config.json").write_text("{}")
        (legacy / "sub" / "telemetry.db").write_bytes(b"db")

        with caplog.at_level(logging.INFO, logger="rlmkit.branding"):
            target = branding.state_dir()

        assert target == tmp_path / branding.STATE_DIR_NAME
        assert (target / "config.json").read_text() == "{}"
        assert (target / "sub" / "telemetry.db").read_bytes() == b"db"
        # Originals untouched.
        assert (legacy / "config.json").exists()
        assert any("Copied" in r.getMessage() for r in caplog.records)

        # Second call in the same process is a no-op even if target is removed.
        caplog.clear()
        branding.state_dir()
        assert not any("Copied" in r.getMessage() for r in caplog.records)

    def test_no_migration_when_target_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(branding, "LEGACY_STATE_DIR_NAMES", (".legacyx",))
        legacy = tmp_path / ".legacyx"
        legacy.mkdir()
        (legacy / "config.json").write_text("legacy")
        target = tmp_path / branding.STATE_DIR_NAME
        target.mkdir()
        (target / "config.json").write_text("current")

        branding.state_dir()
        assert (target / "config.json").read_text() == "current"

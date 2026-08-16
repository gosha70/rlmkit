"""Tests for ``rlmstudio.branding`` — the single source of product identity.

These exercise the env-var and state-dir accessors, including the legacy
fallback machinery that a future rename relies on.  Legacy tuples are empty
today, so the fallback paths are driven by monkeypatching the constants.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rlmstudio import branding


@pytest.fixture(autouse=True)
def _reset_branding_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate module-level caches and env between tests."""
    monkeypatch.setattr(branding, "_warned_legacy_env", set())
    # conftest points RLM_STUDIO_DIR at a temp dir for the whole session;
    # these tests exercise default resolution, so clear it (and any legacy
    # names) per test.
    for key in list(branding.os.environ):
        if key.startswith(branding.ENV_PREFIX) or key.startswith("LEGACYX_"):
            monkeypatch.delenv(key, raising=False)
        for legacy in branding.LEGACY_ENV_PREFIXES:
            if key.startswith(legacy):
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
        with caplog.at_level(logging.WARNING, logger="rlmstudio.branding"):
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
        with caplog.at_level(logging.WARNING, logger="rlmstudio.branding"):
            assert branding.env("PORT") == "8123"
        assert not [r for r in caplog.records if "deprecated" in r.getMessage()]


class TestStateDir:
    def test_default_under_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert branding.state_dir() == tmp_path / branding.STATE_DIR_NAME

    def test_state_dir_is_pure(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Resolving the path must never create or copy anything."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(branding, "LEGACY_STATE_DIR_NAMES", (".legacyx",))
        legacy = tmp_path / ".legacyx"
        legacy.mkdir()
        (legacy / "config.json").write_text("{}")
        target = branding.state_dir()
        assert not target.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == [".legacyx"]

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        override = tmp_path / "elsewhere"
        monkeypatch.setenv(branding.env_name(branding.STATE_DIR_ENV_SUFFIX), str(override))
        assert branding.state_dir() == override


class TestMigrateLegacyState:
    def test_copies_legacy_dir_forward(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(branding, "LEGACY_STATE_DIR_NAMES", (".legacyx",))
        legacy = tmp_path / ".legacyx"
        (legacy / "sub").mkdir(parents=True)
        (legacy / "config.json").write_text("{}")
        (legacy / "sub" / "telemetry.db").write_bytes(b"db")

        with caplog.at_level(logging.INFO, logger="rlmstudio.branding"):
            migrated = branding.migrate_legacy_state()

        target = tmp_path / branding.STATE_DIR_NAME
        assert migrated == legacy
        assert (target / "config.json").read_text() == "{}"
        assert (target / "sub" / "telemetry.db").read_bytes() == b"db"
        # Originals untouched.
        assert (legacy / "config.json").exists()
        assert any("Copied" in r.getMessage() for r in caplog.records)
        # Second call: canonical dir exists → no-op.
        assert branding.migrate_legacy_state() is None

    def test_noop_when_target_exists(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(branding, "LEGACY_STATE_DIR_NAMES", (".legacyx",))
        legacy = tmp_path / ".legacyx"
        legacy.mkdir()
        (legacy / "config.json").write_text("legacy")
        target = tmp_path / branding.STATE_DIR_NAME
        target.mkdir()
        (target / "config.json").write_text("current")

        assert branding.migrate_legacy_state() is None
        assert (target / "config.json").read_text() == "current"

    def test_noop_when_no_legacy_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert branding.migrate_legacy_state() is None
        assert not (tmp_path / branding.STATE_DIR_NAME).exists()

    def test_noop_when_dir_overridden(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        (tmp_path / ".rlmkit").mkdir()
        monkeypatch.setenv(branding.env_name(branding.STATE_DIR_ENV_SUFFIX), str(tmp_path / "x"))
        assert branding.migrate_legacy_state() is None
        assert not (tmp_path / "x").exists()


class TestRealLegacyNames:
    """The rename's actual compatibility contract (RLMKit → RLM Studio)."""

    def test_constants(self) -> None:
        assert branding.ENV_PREFIX == "RLM_STUDIO_"
        assert "RLMKIT_" in branding.LEGACY_ENV_PREFIXES
        assert branding.STATE_DIR_NAME == ".rlm-studio"
        assert ".rlmkit" in branding.LEGACY_STATE_DIR_NAMES
        assert branding.CLI_NAME == "rlm-studio"
        assert branding.DIST_NAME == "rlm-studio"
        assert branding.PACKAGE_NAME == "rlmstudio"

    def test_rlmkit_env_var_is_honoured_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("RLM_STUDIO_PORT", raising=False)
        monkeypatch.setenv("RLMKIT_PORT", "8123")
        with caplog.at_level(logging.WARNING, logger="rlmstudio.branding"):
            assert branding.env("PORT") == "8123"
        assert any("RLMKIT_PORT is deprecated" in r.getMessage() for r in caplog.records)

    def test_rlmkit_state_dir_is_copied_forward(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        legacy = tmp_path / ".rlmkit"
        legacy.mkdir()
        (legacy / "api_keys.json").write_text("{}")
        target = branding.state_dir()
        assert target == tmp_path / ".rlm-studio"
        assert not target.exists()  # resolving is pure
        assert branding.migrate_legacy_state() == legacy
        assert (target / "api_keys.json").exists()
        assert (legacy / "api_keys.json").exists()


class TestConfigSearchPaths:
    def test_canonical_before_legacy_stems(self) -> None:
        from rlmstudio.config import RLMConfig

        paths = RLMConfig.CONFIG_SEARCH_PATHS
        assert paths[0] == "./rlm_studio_config.yaml"
        assert "./rlmkit_config.yaml" in paths
        assert paths.index("./rlm_studio_config.yaml") < paths.index("./rlmkit_config.yaml")
        assert "~/.rlm-studio/config.yaml" in paths


class TestKeyringFallback:
    def test_reads_legacy_service_and_migrates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        from rlmstudio.ui.services import secret_store

        store: dict[tuple[str, str], str] = {("rlmkit", "openai"): "sk-legacy"}
        fake = types.ModuleType("keyring")
        fake.get_password = lambda svc, user: store.get((svc, user))  # type: ignore[attr-defined]
        fake.set_password = lambda svc, user, pw: store.__setitem__((svc, user), pw)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "keyring", fake)

        ks = secret_store.KeyringSecretStore()
        assert ks.get("openai") == "sk-legacy"
        # Migrated forward under the canonical service name.
        assert store[(branding.KEYRING_SERVICE, "openai")] == "sk-legacy"
        assert ks.get("anthropic") is None


class TestBootBoundary:
    def test_import_does_not_migrate(self, tmp_path: Path) -> None:
        """Importing server modules must not copy state; only the app lifespan does."""
        import os
        import subprocess
        import sys

        (tmp_path / ".rlmkit").mkdir()
        (tmp_path / ".rlmkit" / "config.json").write_text("{}")
        env = {k: v for k, v in os.environ.items() if not k.startswith(branding.ENV_PREFIX)}
        env["HOME"] = str(tmp_path)
        code = (
            "import rlmstudio, rlmstudio.server.dependencies, rlmstudio.server.app, "
            "rlmstudio.cli.main; from rlmstudio.cli.main import main; main(['version'])"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        assert not (tmp_path / ".rlm-studio").exists()

    def test_lifespan_migrates(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from rlmstudio.server.app import app

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.delenv(branding.env_name(branding.STATE_DIR_ENV_SUFFIX), raising=False)
        legacy = tmp_path / ".rlmkit"
        legacy.mkdir()
        (legacy / "config.json").write_text("{}")
        with TestClient(app):
            pass
        assert (tmp_path / ".rlm-studio" / "config.json").exists()
        assert (legacy / "config.json").exists()

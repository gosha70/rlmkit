"""Atomicity tests for AppState.save_config.

Per the spec (doc_internal/specs/scheduled-connection-testing.md §Acceptance
criteria), save_config MUST write via NamedTemporaryFile + os.replace so
that a mid-write crash never leaves a partial or corrupted config file on
disk.  This matters because the scheduled-connection-testing feature will
drive save_config at ~1500 writes/day, making any non-atomicity eventually
visible as a reliability regression.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from rlmstudio.server import dependencies
from rlmstudio.server.dependencies import AppState


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect _CONFIG_FILE to a temp location so tests never touch ~/.rlmkit."""
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(dependencies, "_RLMKIT_DIR", tmp_path)
    monkeypatch.setattr(dependencies, "_CONFIG_FILE", config_path)
    return config_path


def test_save_config_writes_valid_json(isolated_config: Path) -> None:
    """Baseline: save_config produces a valid file on disk."""
    state = AppState(load_from_disk=False)
    state.save_config()
    assert isolated_config.exists()
    data = json.loads(isolated_config.read_text())
    assert "config" in data
    assert "system_prompts" in data
    assert "user_profiles" in data


def test_save_config_is_atomic_under_simulated_crash(isolated_config: Path) -> None:
    """Simulate a process abort between temp-file write and os.replace.

    Precondition: a previous valid config exists on disk.
    Action: call save_config() with os.replace patched to raise, so the temp
    file is created but the rename never happens.
    Postcondition: the target file still contains the previous valid
    contents — never partial, never corrupted.
    """
    # Write an initial known-good config.
    state = AppState(load_from_disk=False)
    state.config.active_provider = "openai"
    state.config.active_model = "gpt-4o"
    state.save_config()

    # Snapshot the good contents for comparison.
    good_contents = isolated_config.read_text()
    good_data = json.loads(good_contents)
    assert good_data["config"]["active_provider"] == "openai"

    # Mutate state and attempt a save that crashes mid-rename.
    state.config.active_provider = "anthropic"
    state.config.active_model = "claude-4"

    with patch(
        "rlmstudio.server.dependencies.os.replace",
        side_effect=OSError("simulated abort"),
    ):
        state.save_config()  # save_config swallows exceptions; that's fine.

    # Target file must still be the original contents.
    assert isolated_config.exists()
    post_crash = isolated_config.read_text()
    assert post_crash == good_contents, (
        "save_config left the config file in a changed state despite the "
        "rename failing — atomicity contract broken"
    )
    # And still parseable.
    post_data = json.loads(post_crash)
    assert post_data["config"]["active_provider"] == "openai"


def test_save_config_cleans_up_temp_file_on_write_failure(
    isolated_config: Path,
) -> None:
    """Write-path failure (ENOSPC, EIO, quota exceeded) must not leak
    temp files.  Regression test for commit ddf5339's predecessor which
    only cleaned up on rename failure, not write failure.

    Simulated via patching ``os.fsync`` — it runs after the temp file
    exists on disk and after the write, so a failure here leaves a
    written-but-unsynced temp on disk.  On a real filesystem this is
    exactly what ENOSPC looks like when the write buffered but the
    kernel couldn't flush it.
    """
    state = AppState(load_from_disk=False)
    state.save_config()  # create initial good config

    target_dir = isolated_config.parent
    before = {p.name for p in target_dir.iterdir()}

    with patch(
        "rlmstudio.server.dependencies.os.fsync",
        side_effect=OSError("ENOSPC: simulated disk full"),
    ):
        state.save_config()  # swallows the OSError

    after = {p.name for p in target_dir.iterdir()}
    leaked = after - before
    leaked_temps = [n for n in leaked if n.startswith(".config.") and n.endswith(".tmp")]
    assert leaked_temps == [], (
        f"save_config leaked {len(leaked_temps)} temp file(s) after a "
        f"write failure (ENOSPC simulation): {leaked_temps}"
    )


def test_save_config_cleans_up_temp_file_on_rename_failure(
    isolated_config: Path,
) -> None:
    """A failed os.replace must not leak the temp file.

    Otherwise repeated background-cycle save failures would flood the
    config directory with .config.<random>.tmp files.
    """
    state = AppState(load_from_disk=False)
    state.save_config()  # create initial good config

    target_dir = isolated_config.parent
    before = {p.name for p in target_dir.iterdir()}

    with patch(
        "rlmstudio.server.dependencies.os.replace",
        side_effect=OSError("simulated abort"),
    ):
        state.save_config()

    after = {p.name for p in target_dir.iterdir()}
    leaked = after - before
    # Temp files would match ".config.*.tmp"; good files should not.
    leaked_temps = [n for n in leaked if n.startswith(".config.") and n.endswith(".tmp")]
    assert leaked_temps == [], (
        f"save_config leaked {len(leaked_temps)} temp file(s) after a "
        f"rename failure: {leaked_temps}"
    )


def test_save_config_uses_same_directory_for_temp(isolated_config: Path) -> None:
    """Temp file MUST live in the same dir as the target.

    Cross-filesystem os.replace is not atomic.  If save_config accidentally
    writes the temp to /tmp/<random> and renames across devices, the rename
    can degrade to copy+delete, reintroducing the partial-file window this
    whole mechanism is designed to avoid.
    """
    target_dir = isolated_config.parent
    captured_dirs: list[str] = []
    real_named_tempfile = __import__("tempfile").NamedTemporaryFile

    def _capturing_tempfile(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        d = kwargs.get("dir")
        if d is not None:
            captured_dirs.append(str(d))
        return real_named_tempfile(*args, **kwargs)

    state = AppState(load_from_disk=False)
    with patch("rlmstudio.server.dependencies.tempfile.NamedTemporaryFile", _capturing_tempfile):
        state.save_config()

    assert captured_dirs, "save_config did not pass a dir= to NamedTemporaryFile"
    for d in captured_dirs:
        assert os.path.realpath(d) == os.path.realpath(str(target_dir)), (
            f"save_config wrote temp to {d}, expected {target_dir}"
        )

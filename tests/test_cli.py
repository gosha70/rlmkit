"""Smoke tests for the ``rlmkit`` console script.

These tests deliberately do NOT boot a real uvicorn server. They cover:

* The CLI module imports cleanly.
* ``rlmkit version`` exits 0 and prints a version string.
* ``rlmkit studio --help`` exits 0 and lists the documented flags.
* The bundled-UI helper returns the right value for the current source
  tree state (``None`` when ``_ui/`` is empty in the dev checkout).
* The FastAPI app still constructs without a bundled UI (regression
  guard for the conditional mount added in ``server/app.py``).

The end-to-end "``rlmkit studio`` actually serves the UI" test belongs
in the manual release plan, not in pytest. See
``doc_internal/release/steps/01-pre-cut-gates.md`` § "One-click smoke
test" once that section lands.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout

import pytest

from rlmkit.cli import main as cli_main_module
from rlmkit.cli.main import main
from rlmkit.ui_bundle import get_ui_directory, is_ui_bundled


def test_cli_module_imports() -> None:
    """The CLI module must be importable without side effects."""
    assert callable(cli_main_module.main)


def test_version_command_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("rlmkit ")


def test_studio_help_exits_zero() -> None:
    """`rlmkit studio --help` should print usage and exit 0.

    argparse exits via SystemExit, not return; catch it explicitly.
    """
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, redirect_stdout(buf):
        main(["studio", "--help"])
    assert exc_info.value.code == 0
    help_text = buf.getvalue()
    assert "--host" in help_text
    assert "--port" in help_text
    assert "--no-browser" in help_text
    assert "--reload" in help_text


def test_top_level_help_exits_zero() -> None:
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, redirect_stdout(buf):
        main(["--help"])
    assert exc_info.value.code == 0
    help_text = buf.getvalue()
    assert "studio" in help_text
    assert "version" in help_text


def test_unknown_command_exits_nonzero() -> None:
    """argparse rejects unknown subcommands with exit code 2."""
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, redirect_stderr(buf):
        main(["definitely-not-a-real-subcommand"])
    assert exc_info.value.code != 0


def test_studio_without_bundled_ui_prints_helpful_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the bundled UI is missing, `rlmkit studio` must:

    * exit non-zero (not crash uvicorn)
    * print actionable guidance pointing at the dev workflow
    * NOT attempt to start uvicorn
    """
    # Force the UI lookup to return None regardless of the actual state.
    monkeypatch.setattr("rlmkit.cli.main.get_ui_directory", lambda: None)

    # Sentinel that fails the test if uvicorn.run is invoked.
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("uvicorn.run must not be called when UI is missing")

    monkeypatch.setattr("uvicorn.run", _fail)

    rc = main(["studio", "--no-browser", "--port", "0"])
    captured = capsys.readouterr()

    assert rc != 0
    # Match exact-case to keep the assertion honest; the CLI prints
    # "RLMKit: no bundled UI was found." and the dev workflow hints.
    assert "no bundled UI" in captured.err
    assert "npm run build:bundle" in captured.err
    assert "python -m rlmkit.server" in captured.err


def test_ui_bundle_helpers_consistent() -> None:
    """`is_ui_bundled()` must agree with `get_ui_directory() is not None`."""
    assert is_ui_bundled() is (get_ui_directory() is not None)


def test_server_app_constructs_without_bundled_ui() -> None:
    """Regression guard: the FastAPI app must import cleanly when
    ``src/rlmkit/_ui/`` is empty (no ``index.html``), which is the
    state of every source checkout that has not run the frontend build.

    If this test fails, the conditional mount in
    ``rlmkit.server.app.create_app`` is regressed and the API-only
    workflow (`python -m rlmkit.server` + `npm run dev` on :3000) is
    broken.
    """
    from rlmkit.server.app import app  # noqa: F401 — import is the test.

"""Root pytest conftest — ``--runslow`` gate for heavy tests.

Some tests (notably the subprocess-sandbox fresh-interpreter entrypoint
tests) spawn multiple fresh Python interpreters per case and take 5–15
seconds each.  They are essential for CI but painful in local iteration.

This module wires the already-declared ``slow`` marker in
``pyproject.toml`` to a ``--runslow`` command-line gate using the
standard pytest pattern:

- Default behavior: tests marked ``@pytest.mark.slow`` are **skipped**
  with a clear reason message.  Run all other tests fast.
- ``pytest --runslow``: include slow tests.  CI passes this flag.

This is NOT a CI-only mechanism — contributors can ``pytest --runslow``
locally when they specifically want the full suite.  But ``pytest`` with
no flags stays fast.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--runslow`` CLI flag."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (spawn-heavy, 5-15s each).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip slow-marked tests unless ``--runslow`` was passed."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test — pass --runslow to include (pytest --runslow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

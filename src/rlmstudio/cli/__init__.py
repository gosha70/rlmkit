"""Command-line interface for RLM Studio.

The ``rlm-studio`` console script is registered via
``[project.scripts]`` in ``pyproject.toml``. Subcommands are dispatched
from ``main()`` in ``cli.main``.

Currently implemented:

* ``rlm-studio studio`` — boot RLM Studio (FastAPI + bundled Next.js UI on a
  single process, single port). Auto-opens the browser unless
  ``--no-browser`` is passed.
* ``rlmstudio version`` — print the installed package version and exit.

Planned for later releases (not in v1.1):

* ``rlmstudio serve`` — boot the API without the UI (for production
  deployments behind a reverse proxy).
* ``rlmstudio doctor`` — diagnose common installation / configuration
  problems (missing UI bundle, unreachable providers, port conflicts).
"""

# Deliberately empty — re-exporting `main` here would shadow the
# `rlmstudio.cli.main` submodule (the name collision breaks
# ``monkeypatch.setattr("rlmstudio.cli.main.<attr>", ...)`` in tests and
# any other code that looks up the module via dotted-path resolution).
# Import directly from the submodule:
#
#     from rlmstudio.cli.main import main

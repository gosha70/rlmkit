"""Entry point for ``python -m rlmkit.server``.

CLI flags take precedence over environment variables.

Usage::

    # default: 127.0.0.1:8000
    uv run python -m rlmkit.server

    # CLI flags
    uv run python -m rlmkit.server --port 8080
    uv run python -m rlmkit.server --host 0.0.0.0 --port 9000 --reload

    # env vars (overridden by CLI flags if both are given)
    RLMKIT_PORT=8080 uv run python -m rlmkit.server
    RLMKIT_HOST=0.0.0.0 RLMKIT_PORT=9000 uv run python -m rlmkit.server --reload
"""

from __future__ import annotations

import argparse

import uvicorn

from rlmkit.branding import env, env_name

_parser = argparse.ArgumentParser(
    prog="python -m rlmkit.server",
    description="Start the RLMKit API server.",
)
_parser.add_argument(
    "--host",
    default=env("HOST", "127.0.0.1"),
    help=f"Bind host (default: 127.0.0.1, env: {env_name('HOST')})",
)
_parser.add_argument(
    "--port",
    type=int,
    default=int(env("PORT", "8000") or "8000"),
    help=f"Bind port (default: 8000, env: {env_name('PORT')})",
)
_parser.add_argument(
    "--reload",
    action="store_true",
    help="Enable auto-reload on source changes (development only)",
)

args = _parser.parse_args()

uvicorn.run(
    "rlmkit.server.app:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
)

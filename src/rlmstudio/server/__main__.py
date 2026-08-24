"""Entry point for ``python -m rlmstudio.server``.

CLI flags take precedence over environment variables.

Usage::

    # default: 127.0.0.1:8000
    uv run python -m rlmstudio.server

    # CLI flags
    uv run python -m rlmstudio.server --port 8080
    uv run python -m rlmstudio.server --host 0.0.0.0 --port 9000 --reload

    # env vars (overridden by CLI flags if both are given)
    RLM_STUDIO_PORT=8080 uv run python -m rlmstudio.server
    RLM_STUDIO_HOST=0.0.0.0 RLM_STUDIO_PORT=9000 uv run python -m rlmstudio.server --reload
"""

from __future__ import annotations

import argparse

import uvicorn

from rlmstudio.branding import DEFAULT_HOST, DEFAULT_PORT, PRODUCT_NAME, env, env_name

_parser = argparse.ArgumentParser(
    prog="python -m rlmstudio.server",
    description=f"Start the {PRODUCT_NAME} API server.",
)
_parser.add_argument(
    "--host",
    default=env("HOST", DEFAULT_HOST),
    help=f"Bind host (default: {DEFAULT_HOST}, env: {env_name('HOST')})",
)
_parser.add_argument(
    "--port",
    type=int,
    default=int(env("PORT", str(DEFAULT_PORT)) or DEFAULT_PORT),
    help=f"Bind port (default: {DEFAULT_PORT}, env: {env_name('PORT')})",
)
_parser.add_argument(
    "--reload",
    action="store_true",
    help="Enable auto-reload on source changes (development only)",
)

args = _parser.parse_args()

uvicorn.run(
    "rlmstudio.server.app:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
)

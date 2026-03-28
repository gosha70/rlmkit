"""Entry point for ``python -m rlmkit.server``.

Reads RLMKIT_HOST and RLMKIT_PORT environment variables so the server
can be started on a non-default address without modifying source code.

Usage::

    # default: 127.0.0.1:8000
    uv run python -m rlmkit.server

    # custom port
    RLMKIT_PORT=8080 uv run python -m rlmkit.server

    # custom host + port
    RLMKIT_HOST=0.0.0.0 RLMKIT_PORT=9000 uv run python -m rlmkit.server --reload
"""

from __future__ import annotations

import os
import sys

import uvicorn

HOST = os.environ.get("RLMKIT_HOST", "127.0.0.1")
PORT = int(os.environ.get("RLMKIT_PORT", "8000"))

# Pass any extra CLI args (e.g. --reload) straight through to uvicorn.
extra_args = sys.argv[1:]

uvicorn.run(
    "rlmkit.server.app:app",
    host=HOST,
    port=PORT,
    # --reload is the most common extra arg; uvicorn handles the rest.
    reload="--reload" in extra_args,
)

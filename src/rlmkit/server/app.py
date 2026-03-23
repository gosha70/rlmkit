"""FastAPI application for RLMKit API server."""

from __future__ import annotations

import logging
import os as _os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rlmkit.server.dependencies import get_state
from rlmkit.server.models import HealthResponse
from rlmkit.server.routes import (
    chat,
    chat_providers,
    config,
    evaluations,
    files,
    metrics,
    profiles,
    prompts,
    providers,
    sessions,
    traces,
)

# Snapshot keys already in the real process environment *before* load_dotenv.
# This lets us distinguish user-set shell variables (must not be overridden)
# from .env-loaded values (may be overridden by a newer SecretStore entry).
_real_env_keys: frozenset[str] = frozenset(_os.environ.keys())

# Load .env file (backward-compat: keys saved before the SecretStore migration)
_env_path = Path(".env")
load_dotenv(_env_path)


# Reload keys persisted via SecretStore (keyring / JSON file) into os.environ
# so all server read-paths (os.environ.get) work after restart.
# SecretStore wins over .env; real process-env vars are never overridden.
def _reload_stored_api_keys() -> None:
    from rlmkit.ui.data.providers_catalog import PROVIDERS
    from rlmkit.ui.services.secret_store import FileSecretStore, KeyringSecretStore

    file_store = FileSecretStore()
    keyring_store = KeyringSecretStore() if KeyringSecretStore.is_available() else None

    for p in PROVIDERS:
        if not p.env_var:
            continue
        if p.env_var in _real_env_keys:
            continue  # set in the shell before the server started — never override
        key = (keyring_store.get(p.key) if keyring_store else None) or file_store.get(p.key)
        if key:
            _os.environ[p.env_var] = key


_reload_stored_api_keys()

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Configure logging so all rlmkit messages appear in the terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Set rlmkit loggers to DEBUG so detailed traces are visible
    logging.getLogger("rlmkit").setLevel(logging.DEBUG)

    app = FastAPI(
        title="RLMKit API",
        version="0.1.0",
        description="Recursive Language Model toolkit API server",
    )

    # CORS middleware for frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
            "http://host.docker.internal:3000",
            "http://host.docker.internal:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom exception handler to match ErrorResponse format from spec
    code_map = {
        400: "VALIDATION_ERROR",
        401: "UNAUTHORIZED",
        404: "NOT_FOUND",
        413: "FILE_TOO_LARGE",
        500: "INTERNAL_ERROR",
    }

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: object, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code_map.get(exc.status_code, "ERROR"),
                    "message": str(exc.detail),
                    "details": {},
                }
            },
        )

    # Include route modules
    app.include_router(chat.router)
    app.include_router(files.router)
    app.include_router(sessions.router)
    app.include_router(metrics.router)
    app.include_router(traces.router)
    app.include_router(providers.router)
    app.include_router(profiles.router)
    app.include_router(prompts.router)
    app.include_router(config.router)
    app.include_router(chat_providers.router)
    app.include_router(evaluations.router)

    @app.get("/health")
    async def health_check() -> HealthResponse:
        state = get_state()
        return HealthResponse(
            status="ok",
            version="0.1.0",
            uptime_seconds=round(time.time() - state.start_time, 1),
        )

    return app


app = create_app()

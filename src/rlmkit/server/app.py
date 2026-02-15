"""FastAPI application for RLMKit API server."""

from __future__ import annotations

import logging
import time

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env file so API keys persist across restarts
_env_path = Path(".env")
_loaded = load_dotenv(_env_path)
print(f">>> DOTENV: load_dotenv({_env_path.resolve()}) returned {_loaded}, file exists={_env_path.exists()}")
# Show which API key env vars are set (without revealing values)
for _var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
    _val = os.environ.get(_var, "")
    print(f">>>   {_var}={'set (' + str(len(_val)) + ' chars)' if _val else 'NOT SET'}")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rlmkit.server.dependencies import get_state
from rlmkit.server.models import HealthResponse
from rlmkit.server.routes import chat, config, files, metrics, profiles, prompts, providers, sessions, traces


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
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
    async def http_exception_handler(request, exc):
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

"""FastAPI application for RLMKit API server."""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rlmkit.server.dependencies import get_state
from rlmkit.server.models import HealthResponse
from rlmkit.server.routes import chat, config, files, metrics, providers, sessions, traces


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
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

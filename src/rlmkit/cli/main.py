"""``rlmkit`` console-script entry point.

The CLI is intentionally tiny: ``argparse``, no third-party CLI library.
Each subcommand lives in its own helper function so the dispatcher stays
flat.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence

from rlmkit import __version__
from rlmkit.ui_bundle import get_ui_directory

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlmkit",
        description="RLMKit — Recursive Language Model toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # rlmkit version
    subparsers.add_parser(
        "version",
        help="Print the installed RLMKit version and exit.",
    )

    # rlmkit studio
    studio = subparsers.add_parser(
        "studio",
        help="Start RLM Studio (API + bundled web UI on a single port).",
        description=(
            "Start RLM Studio: the FastAPI backend with the bundled "
            "Next.js UI mounted at /studio. Opens your default browser "
            "to the UI unless --no-browser is passed."
        ),
    )
    studio.add_argument(
        "--host",
        default=os.environ.get("RLMKIT_HOST", "127.0.0.1"),
        help="Bind host (default: 127.0.0.1, env: RLMKIT_HOST)",
    )
    studio.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RLMKIT_PORT", "8000")),
        help="Bind port (default: 8000, env: RLMKIT_PORT)",
    )
    studio.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    studio.add_argument(
        "--browser-delay",
        type=float,
        default=1.5,
        help=(
            "Seconds to wait after server start before opening the "
            "browser (default: 1.5). Increase if the browser opens "
            "before the server is ready."
        ),
    )
    studio.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on source changes (development only).",
    )

    return parser


def _cmd_version() -> int:
    print(f"rlmkit {__version__}")
    return 0


_WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::", "*", ""})


def _browse_host(host: str) -> str:
    """Substitute a browseable loopback for wildcard bind hosts.

    The actual uvicorn bind still uses ``host`` verbatim; this rewrites
    only the URL that is printed and opened in the browser. Browsers do
    not consistently navigate to ``0.0.0.0`` or ``[::]`` — those are
    bind targets, not addressable hosts — so the advertised
    ``rlmkit studio --host 0.0.0.0`` workflow needs a navigable URL
    even though the bind is correct as written.
    """
    return "127.0.0.1" if host in _WILDCARD_BIND_HOSTS else host


def _cmd_studio(args: argparse.Namespace) -> int:
    import uvicorn

    ui_dir = get_ui_directory()
    if ui_dir is None:
        print(
            "RLMKit: no bundled UI was found.\n"
            "\n"
            "  Expected at: rlmkit/_ui/index.html\n"
            "\n"
            "This usually means you installed rlmkit from a source\n"
            "checkout without building the frontend. Two options:\n"
            "\n"
            "  1. Install the published wheel:\n"
            "       pip install --upgrade rlmkit\n"
            "\n"
            "  2. Build the frontend locally:\n"
            "       cd frontend\n"
            "       npm install && npm run build:bundle\n"
            "       # then copy frontend/out → src/rlmkit/_ui/\n"
            "\n"
            "  3. Or run the dev stack with the API and UI as separate\n"
            "     processes:\n"
            "       python -m rlmkit.server           # terminal 1\n"
            "       cd frontend && npm run dev         # terminal 2\n",
            file=sys.stderr,
        )
        return 2

    browse_host = _browse_host(args.host)
    url = f"http://{browse_host}:{args.port}/studio"
    print(f"RLM Studio: {url}")
    print(f"  API:        http://{browse_host}:{args.port}/api")
    print(f"  Health:     http://{browse_host}:{args.port}/health")
    print()

    if not args.no_browser:
        # Open the browser on a delay so the server has time to bind.
        # We do this in a daemon thread so it does not block uvicorn.run().
        def _open_browser() -> None:
            time.sleep(args.browser_delay)
            try:
                webbrowser.open(url)
            except Exception as exc:  # pragma: no cover - depends on host env
                logger.warning("Could not open browser: %s", exc)

        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        "rlmkit.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        return _cmd_version()
    if args.command == "studio":
        return _cmd_studio(args)

    parser.error(f"Unknown command: {args.command!r}")
    return 2  # unreachable; parser.error exits.


if __name__ == "__main__":
    raise SystemExit(main())

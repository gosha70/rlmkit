# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Command-line entrypoint for the wiki backbone.

Subcommands: ``ingest``, ``query``, ``lint``. Exit codes mirror
the typed exception hierarchy in :mod:`rlmkit.wiki.errors`:

* 0 — success (accept *or* reject; both are pipeline successes)
* 2 — backend not found
* 3 — backend invocation failure
* 4 — contract violation
* 5 — source missing
* 6 — output write failure
* 1 — any other ``WikiError``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .backends import IngestBackend, TestBackend
from .errors import BackendNotFound, WikiError
from .ingest import Ingestor
from .linter import lint_wiki
from .proposal import IngestRequest


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_wiki_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    return _project_root() / "knowledge" / "wiki"


def _resolve_schema_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    return _project_root() / "knowledge" / "wiki" / "schema"


def _resolve_output_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    return _project_root() / "doc_internal" / "proposals"


def _build_backend(name: str) -> IngestBackend:
    if name == "test":
        return TestBackend()
    # The ``llm`` and ``rlm`` backends require a configured client
    # the CLI cannot pick on the user's behalf safely. Programmatic
    # callers wire them up explicitly via :class:`Ingestor`.
    raise BackendNotFound(
        f"backend '{name}' requires programmatic instantiation; "
        "the CLI ships only --backend test for v1. Use rlmkit.wiki.Ingestor "
        "from Python with a configured LLMBackend or RLMBackend."
    )


def _cmd_ingest(args: argparse.Namespace) -> int:
    backend = _build_backend(args.backend)
    ingestor = Ingestor(
        backend=backend,
        schema_dir=_resolve_schema_dir(args.schema_dir),
        output_dir=_resolve_output_dir(args.output_dir),
    )
    request = IngestRequest(
        source_path=Path(args.source),
        mode="rlm" if args.backend == "rlm" else "direct",
        backend_name=args.backend,
    )
    path = ingestor.run(request)
    print(path)
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    report = lint_wiki(_resolve_wiki_dir(args.wiki_dir))
    for v in report.violations:
        print(f"  ✗ {v.page}: [{v.rule}] {v.detail}", file=sys.stderr)
    print(report.summary())
    return 0 if report.ok else 1


def _cmd_query(args: argparse.Namespace) -> int:
    print(
        "wiki query CLI requires a configured LLMClient; "
        "call rlmkit.wiki.query_wiki(...) from Python instead.",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rlmkit-wiki")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="ingest a source into a wiki proposal")
    ing.add_argument("source", help="path to the source file (or directory for --backend rlm)")
    ing.add_argument("--backend", default="test", help="backend name (default: test)")
    ing.add_argument("--output-dir", default=None, help="override doc_internal/proposals/")
    ing.add_argument("--schema-dir", default=None, help="override knowledge/wiki/schema/")
    ing.set_defaults(func=_cmd_ingest)

    lin = sub.add_parser("lint", help="lint the wiki tree")
    lin.add_argument("--wiki-dir", default=None, help="override knowledge/wiki/")
    lin.set_defaults(func=_cmd_lint)

    q = sub.add_parser("query", help="query the wiki (programmatic only in v1)")
    q.add_argument("question", help="question to ask")
    q.set_defaults(func=_cmd_query)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WikiError as err:
        print(f"error: {err}", file=sys.stderr)
        return err.exit_code


if __name__ == "__main__":
    sys.exit(main())

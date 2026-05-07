# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""``rlmkit wiki`` CLI — dispatches the four operations.

Usage::

    python -m rlmkit.cli.wiki ingest <source>           [--backend test]
    python -m rlmkit.cli.wiki promote <proposal-dir>
    python -m rlmkit.cli.wiki query "<question>"        [--file-back]
    python -m rlmkit.cli.wiki lint [--health] [--strict]

Common flags::

    --wiki-root knowledge/wiki      override the wiki dir
    --proposals-root doc_internal/proposals
    --backend test                  use the deterministic in-process backend
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rlmkit.strategies.wiki.backends import (
    DeterministicTestBackend,
    LLMClientWikiBackend,
)
from rlmkit.strategies.wiki.errors import WikiError
from rlmkit.strategies.wiki.health_lint import format_findings, lint_health
from rlmkit.strategies.wiki.ingestor import ingest, load_patch_from_dir
from rlmkit.strategies.wiki.promoter import promote
from rlmkit.strategies.wiki.querier import query
from rlmkit.strategies.wiki.structural_lint import format_violations, lint


def _build_backend(args, *, need_llm: bool):
    """Resolve the backend per --backend flag."""
    if args.backend == "test":
        return DeterministicTestBackend()
    if not need_llm:
        return None
    # Lazy import — avoids requiring litellm for non-LLM verbs.
    from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter

    client = LiteLLMAdapter(model=args.model)
    return LLMClientWikiBackend(client, name=args.model or "litellm")


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--wiki-root",
        default="knowledge/wiki",
        help="path to the wiki dir (default: knowledge/wiki)",
    )
    p.add_argument(
        "--proposals-root",
        default="doc_internal/proposals",
        help="path to the proposals root (default: doc_internal/proposals)",
    )
    p.add_argument(
        "--backend",
        default="test",
        choices=["test", "llm"],
        help="backend to use (default: test)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="model identifier when --backend llm",
    )


def cmd_ingest(args) -> int:
    backend = _build_backend(args, need_llm=True)
    patch, proposal_dir = ingest(
        source_path=Path(args.source),
        wiki_dir=Path(args.wiki_root),
        proposals_root=Path(args.proposals_root),
        backend=backend,
    )
    print(f"proposal: {proposal_dir}")
    print(f"  edits: {len(patch.edits)}")
    for e in patch.edits:
        print(f"    [{e.action}] {e.path}")
    return 0


def cmd_promote(args) -> int:
    result = promote(
        proposal_dir=Path(args.proposal_dir),
        wiki_dir=Path(args.wiki_root),
        archive_root=Path(args.proposals_root) / ".applied",
        dry_run=args.dry_run,
    )
    if result.dry_run:
        print(f"dry-run promote: {len(result.applied_paths)} path(s) "
              f"would be applied")
    else:
        print(f"promoted: {len(result.applied_paths)} path(s)")
        for rel in result.applied_paths:
            print(f"  {rel}")
        if result.archived_dir:
            print(f"  archived: {result.archived_dir}")
    return 0


def cmd_query(args) -> int:
    backend = _build_backend(args, need_llm=True)
    audit = Path("doc_internal/wiki-query-log.jsonl")
    ans = query(
        question=args.question,
        wiki_dir=Path(args.wiki_root),
        backend=backend,
        audit_log_path=audit,
    )
    print("--- answer ---")
    print(ans.answer or "(empty)")
    print("--- citations ---")
    for c in ans.citations:
        print(f"  - {c.page}: {c.fragment}")
    print(f"--- pages loaded: {len(ans.pages_loaded)} ---", file=sys.stderr)
    for rel in ans.pages_loaded:
        print(f"  + {rel}", file=sys.stderr)
    if args.file_back:
        # Run an ingest of the question+answer as a synthesised source.
        synthetic = Path("/tmp/_rlmkit_query_filebacked.md")
        synthetic.write_text(
            f"# Query: {args.question}\n\n{ans.answer}\n",
            encoding="utf-8",
        )
        backend2 = _build_backend(args, need_llm=True)
        patch, proposal_dir = ingest(
            source_path=synthetic,
            wiki_dir=Path(args.wiki_root),
            proposals_root=Path(args.proposals_root),
            backend=backend2,
        )
        print(f"--- file-back proposal: {proposal_dir} ---", file=sys.stderr)
    return 0


def cmd_lint(args) -> int:
    wiki_dir = Path(args.wiki_root)
    violations = lint(wiki_dir)
    print(format_violations(violations))
    structural_failed = bool(violations)
    health_failed = False
    if args.health:
        backend = _build_backend(args, need_llm=False)
        findings = lint_health(wiki_dir, backend=backend)
        print(format_findings(findings))
        health_failed = any(f.severity == "error" for f in findings) or (
            args.strict and findings
        )
    if structural_failed:
        return 1
    if health_failed:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rlmkit-wiki")
    sub = parser.add_subparsers(dest="verb", required=True)

    p_ingest = sub.add_parser("ingest", help="produce a multi-page patch-set")
    p_ingest.add_argument("source")
    _add_common_flags(p_ingest)
    p_ingest.set_defaults(func=cmd_ingest)

    p_promote = sub.add_parser("promote", help="apply a proposal directory")
    p_promote.add_argument("proposal_dir")
    p_promote.add_argument("--dry-run", action="store_true")
    _add_common_flags(p_promote)
    p_promote.set_defaults(func=cmd_promote)

    p_query = sub.add_parser("query", help="index-first wiki query")
    p_query.add_argument("question")
    p_query.add_argument(
        "--file-back",
        action="store_true",
        help="generate a patch-set capturing the answer",
    )
    _add_common_flags(p_query)
    p_query.set_defaults(func=cmd_query)

    p_lint = sub.add_parser("lint", help="structural + optional health lint")
    p_lint.add_argument("--health", action="store_true")
    p_lint.add_argument("--strict", action="store_true")
    _add_common_flags(p_lint)
    p_lint.set_defaults(func=cmd_lint)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WikiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

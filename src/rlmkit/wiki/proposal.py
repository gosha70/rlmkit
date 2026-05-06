# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Domain types for ingest requests, proposals, and proposal-file rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

INGESTOR_VERSION = 1


@dataclass(frozen=True)
class IngestRequest:
    """A single ingest invocation.

    ``source_path`` may point at a file (Mode A / B) or a directory
    (Mode C — the recursive controller will load the corpus into
    its REPL environment).
    """

    source_path: Path
    mode: Literal["direct", "rlm"] = "direct"
    backend_name: str = "test"


@dataclass(frozen=True)
class IngestProposal:
    """The structured output of one ingest call.

    On ``disposition == "reject"``, ``page_type`` / ``slug`` /
    ``title`` / ``draft_markdown`` are ``None`` and ``reason``
    explains which gate question failed.
    """

    disposition: Literal["accept", "reject"]
    reason: str
    page_type: str | None = None
    slug: str | None = None
    title: str | None = None
    draft_markdown: str | None = None
    sources: list[dict] = field(default_factory=list)


def render_proposal_file(
    proposal: IngestProposal,
    *,
    source_path: Path,
    backend_name: str,
    proposal_date: str,
) -> str:
    """Render a proposal to the markdown file body.

    The frontmatter block describes the *proposal*; for an accept
    proposal, the body that follows is the candidate wiki page
    (which itself carries its own wiki-page-shaped frontmatter).
    """
    fm_lines = [
        "---",
        f"proposal_kind: {proposal.disposition}",
        f"proposal_date: {proposal_date}",
        f"source_path: {source_path}",
        f"backend: {backend_name}",
        f"ingestor_version: {INGESTOR_VERSION}",
        f"gate_disposition: {proposal.disposition}",
        f"gate_reason: {proposal.reason}",
        f"target_slug: {proposal.slug or ''}",
        f"target_page_type: {proposal.page_type or ''}",
        "---",
        "",
    ]
    if proposal.disposition == "accept" and proposal.draft_markdown:
        body = proposal.draft_markdown
    else:
        body = (
            f"# Rejected: {source_path.name}\n\n"
            f"{proposal.reason}\n"
        )
    return "\n".join(fm_lines) + body

# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki ingest orchestrator.

Composes prompt → backend → two-layer validation → proposal-file
write. The two-layer validation is borrowed from
``code-copilot-team/specs/wiki-ingest-pipeline/spec.md`` §Interface
and protects the curator from proposals that pass JSON-shape
checks but would fail the wiki linter or the schema templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path

from .backends import IngestBackend
from .errors import ContractViolation, OutputWriteFailure, SourceMissing
from .proposal import IngestProposal, IngestRequest, render_proposal_file
from .schema import (
    PAGE_TYPE_DIRS,
    SOURCES_EXEMPT_TYPES,
    is_kebab_case,
    parse_frontmatter,
)


@dataclass
class Ingestor:
    """Drive a single ingest invocation end-to-end.

    The wiki schema directory (``knowledge/wiki/schema/``) is read
    on every invocation rather than embedded in source — that
    matches cct's design and keeps the prompt in sync with whatever
    the schema currently says.
    """

    backend: IngestBackend
    schema_dir: Path
    output_dir: Path
    schema_excerpts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.schema_excerpts:
            self.schema_excerpts = self._load_schema()

    def _load_schema(self) -> dict[str, str]:
        excerpts: dict[str, str] = {}
        for key, filename in (
            ("ingest_rules", "ingest-rules.md"),
            ("page_types", "page-types.md"),
            ("citation_rules", "citation-rules.md"),
        ):
            path = self.schema_dir / filename
            if path.exists():
                excerpts[key] = path.read_text(encoding="utf-8")
            else:
                excerpts[key] = ""
        return excerpts

    def run(self, request: IngestRequest) -> Path:
        """Run the pipeline; return the proposal-file path on success.

        Raises typed :class:`WikiError` subclasses on every documented
        failure mode (see ``rlmkit.wiki.errors``).
        """
        if not request.source_path.exists():
            raise SourceMissing(f"source path does not exist: {request.source_path}")

        proposal = self.backend.ingest(request, self.schema_excerpts)
        validate_proposal(proposal)
        return self._write_proposal(proposal, request)

    def _write_proposal(self, proposal: IngestProposal, request: IngestRequest) -> Path:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OutputWriteFailure(
                f"could not create output dir {self.output_dir}: {exc}"
            ) from exc

        proposal_date = _date.today().isoformat()
        slug = proposal.slug or "rejected"
        filename = f"{proposal_date}-{slug}.md"
        path = self.output_dir / filename
        body = render_proposal_file(
            proposal,
            source_path=request.source_path,
            backend_name=self.backend.name,
            proposal_date=proposal_date,
        )
        try:
            path.write_text(body, encoding="utf-8")
        except OSError as exc:
            raise OutputWriteFailure(
                f"could not write proposal file {path}: {exc}"
            ) from exc
        return path


def validate_proposal(proposal: IngestProposal) -> None:
    """Run the two-layer validation.

    Layer 1 (shape) — every accept proposal must carry the
    structured fields the linter will need.

    Layer 2 (semantic cross-consistency) — if a draft is present,
    its embedded YAML frontmatter must agree with the structured
    fields and the (page_type, slug) pair must satisfy the
    directory-placement rule the wiki linter enforces.
    """
    if proposal.disposition == "reject":
        if not proposal.reason:
            raise ContractViolation("reject proposal missing reason")
        return

    # accept ─ layer 1
    for field_name in ("page_type", "slug", "title", "draft_markdown"):
        if not getattr(proposal, field_name):
            raise ContractViolation(f"accept proposal missing '{field_name}'")
    if proposal.page_type not in PAGE_TYPE_DIRS:
        raise ContractViolation(
            f"accept proposal page_type '{proposal.page_type}' is not a canonical type"
        )
    assert proposal.slug is not None
    if not is_kebab_case(proposal.slug):
        raise ContractViolation(f"slug '{proposal.slug}' is not kebab-case")
    if (
        proposal.page_type not in SOURCES_EXEMPT_TYPES
        and not proposal.sources
    ):
        raise ContractViolation(
            f"accept proposal of page_type '{proposal.page_type}' has no sources"
        )

    # accept ─ layer 2 (semantic cross-consistency)
    assert proposal.draft_markdown is not None
    fm, _ = parse_frontmatter(proposal.draft_markdown)
    if not fm:
        raise ContractViolation("draft_markdown does not begin with valid frontmatter")
    for key, expected in (
        ("page_type", proposal.page_type),
        ("slug", proposal.slug),
        ("title", proposal.title),
    ):
        if fm.get(key) != expected:
            raise ContractViolation(
                f"draft frontmatter {key}={fm.get(key)!r} ≠ structured {key}={expected!r}"
            )
    fm_sources = fm.get("sources") or []
    if not fm_sources:
        raise ContractViolation("draft frontmatter sources are empty")

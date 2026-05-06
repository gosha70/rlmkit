# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""End-to-end test: TestBackend → Ingestor → proposal file → linter parity.

This is the anchor test the spec calls out. It exercises ingest
through the deterministic backend, asserts the proposal file is
written with the documented frontmatter, and confirms the embedded
draft would lint clean if dropped into the wiki tree.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rlmkit.wiki.backends import TestBackend
from rlmkit.wiki.ingest import Ingestor, validate_proposal
from rlmkit.wiki.linter import lint_wiki
from rlmkit.wiki.proposal import IngestRequest
from rlmkit.wiki.schema import parse_frontmatter

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "knowledge" / "wiki" / "schema"
FIXTURE = REPO_ROOT / "tests" / "wiki" / "fixtures" / "sample-incident.md"


def test_ingest_with_test_backend_writes_valid_proposal(tmp_path: Path) -> None:
    out_dir = tmp_path / "proposals"
    ingestor = Ingestor(
        backend=TestBackend(),
        schema_dir=SCHEMA_DIR,
        output_dir=out_dir,
    )
    request = IngestRequest(source_path=FIXTURE, mode="direct", backend_name="test")
    proposal_path = ingestor.run(request)

    assert proposal_path.exists()
    body = proposal_path.read_text(encoding="utf-8")
    fm, close = parse_frontmatter(body)
    assert fm["proposal_kind"] == "accept"
    assert fm["gate_disposition"] == "accept"
    assert fm["target_page_type"] == "concept"
    assert fm["target_slug"]
    # The body after the proposal frontmatter is the candidate wiki page.
    page_body = "\n".join(body.splitlines()[close:])
    page_fm, _ = parse_frontmatter(page_body)
    assert page_fm["page_type"] == "concept"
    assert page_fm["slug"] == fm["target_slug"]


def test_test_backend_proposal_lints_clean_when_dropped_into_wiki(tmp_path: Path) -> None:
    """Dropping the embedded draft into a fresh wiki tree must lint clean."""
    out_dir = tmp_path / "proposals"
    ingestor = Ingestor(
        backend=TestBackend(),
        schema_dir=SCHEMA_DIR,
        output_dir=out_dir,
    )
    request = IngestRequest(source_path=FIXTURE, mode="direct", backend_name="test")
    proposal_path = ingestor.run(request)

    body = proposal_path.read_text(encoding="utf-8")
    fm, close = parse_frontmatter(body)
    page_body = "\n".join(body.splitlines()[close:])
    target_slug = fm["target_slug"]

    # Build a minimal wiki that contains only the seeded pages plus
    # the candidate. Linter must report 0 violations.
    wiki = tmp_path / "wiki"
    src_wiki = REPO_ROOT / "knowledge" / "wiki"
    shutil.copytree(src_wiki, wiki)

    # Drop the candidate page in the right directory and link it
    # from the index so the orphan check passes.
    target_dir = wiki / "concepts"
    target_dir.mkdir(exist_ok=True)
    (target_dir / f"{target_slug}.md").write_text(page_body, encoding="utf-8")

    # The seeded index already lists llm-wiki-as-knowledge-layer;
    # append the new bullet so the new page is reachable.
    index_path = wiki / "index.md"
    index_text = index_path.read_text(encoding="utf-8")
    index_path.write_text(
        index_text + f"- [{target_slug}](concepts/{target_slug}.md)\n",
        encoding="utf-8",
    )

    report = lint_wiki(wiki)
    assert report.ok, "\n".join(f"{v.page}: [{v.rule}] {v.detail}" for v in report.violations)


def test_validate_proposal_rejects_layer2_mismatch() -> None:
    """A draft whose frontmatter disagrees with structured fields must fail."""
    from rlmkit.wiki.errors import ContractViolation
    from rlmkit.wiki.proposal import IngestProposal

    # page_type in the structured field says 'concept' but the
    # embedded frontmatter says 'incident' — layer 2 must catch this.
    bad_draft = (
        "---\npage_type: incident\nslug: foo\ntitle: Foo\n"
        "status: draft\nlast_reviewed: 2026-05-05\n"
        "sources:\n  - issue: 1\n---\n# Foo\n"
    )
    proposal = IngestProposal(
        disposition="accept",
        reason="ok",
        page_type="concept",
        slug="foo",
        title="Foo",
        draft_markdown=bad_draft,
        sources=[{"issue": 1}],
    )
    with pytest.raises(ContractViolation):
        validate_proposal(proposal)

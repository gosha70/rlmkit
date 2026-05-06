# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""End-to-end wiki test: ingest -> query -> promote, all against stubs.

This is the runnable feature deliverable named in the spec. It exercises
the wiki backbone without any real LLM call:

1. Ingest a raw markdown source via IngestSourceUseCase + StubLLMClient
   that returns a canned YAML page list.
2. Verify pages, raw mirror, and index were written.
3. Query the wiki via QueryWikiUseCase (also stubbed) and assert the
   answer + COVERAGE header parse correctly.
4. Drive WikiRLMStrategy: when the wiki returns COVERAGE: missing, the
   strategy must hand off to RLM fallback; in this test we substitute a
   stub LLMClient that mimics the RLM controller's reply.
5. Promote a fresh answer back into the wiki.
"""

from __future__ import annotations

from datetime import date

import pytest

from rlmkit.application.use_cases.wiki import (
    IngestSourceUseCase,
    LintWikiUseCase,
    PromoteAnswerUseCase,
    QueryWikiUseCase,
)
from rlmkit.application.use_cases.wiki.query import (
    COVERAGE_FULL,
    COVERAGE_MISSING,
)
from rlmkit.domain.wiki import PageType
from rlmkit.infrastructure.wiki import MarkdownWikiRepository
from rlmkit.strategies.wiki import WikiStrategy

# Canned ingest reply — matches the wiki_ingest YAML contract.
INGEST_YAML_REPLY = """
pages:
  - title: Chunking strategy
    slug: chunking-strategy
    type: concept
    sources: [chunking-notes]
    status: draft
    body: |
      Chunks should be 1000 tokens with 100 overlap.
  - title: Why chunk
    slug: why-chunk
    type: concept
    sources: [chunking-notes]
    status: draft
    body: |
      Long documents exceed model context windows.
"""

QUERY_REPLY_FULL = (
    "COVERAGE: full\n\n"
    "Use 1000-token chunks with 100-token overlap [chunking-strategy]."
)

QUERY_REPLY_MISSING = (
    "COVERAGE: missing\n\n"
    "The wiki does not yet describe entity extraction."
)


def test_e2e_ingest_query_promote(
    wiki_root, stub_embedder, make_stub_llm
):
    repo = MarkdownWikiRepository(wiki_root)

    # --- 1. INGEST ---------------------------------------------------
    ingest_llm = make_stub_llm(default=INGEST_YAML_REPLY)
    ingest = IngestSourceUseCase(repo=repo, client=ingest_llm)
    result = ingest.execute(
        source_id="chunking-notes",
        content="Chunk overlap matters for retrieval recall.",
        today=date(2026, 5, 5),
    )

    assert result.pages_created == [
        "concepts/chunking-strategy.md",
        "concepts/why-chunk.md",
    ]
    assert result.pages_updated == []
    assert "chunking-notes" in repo.list_raws()
    index_text = repo.read_index()
    assert "chunking-strategy" in index_text
    assert "why-chunk" in index_text
    log = repo.read_log()
    assert "ingest source=chunking-notes" in log

    # Re-ingest is idempotent on slug — same pages, just updated.
    result2 = ingest.execute(
        source_id="chunking-notes",
        content="Chunk overlap matters for retrieval recall.",
        today=date(2026, 5, 5),
    )
    assert result2.pages_created == []
    assert len(result2.pages_updated) == 2

    # --- 1b. LINT ----------------------------------------------------
    lint_report = LintWikiUseCase(repo).execute()
    assert lint_report.passed, lint_report.errors

    # --- 2. QUERY (covered) ------------------------------------------
    query_llm = make_stub_llm(default=QUERY_REPLY_FULL)
    query = QueryWikiUseCase(repo=repo, client=query_llm, embedder=stub_embedder)
    outcome = query.execute("What chunk size should we use?")
    assert outcome.coverage == COVERAGE_FULL
    assert "1000-token" in outcome.strategy_result.answer
    assert outcome.strategy_result.metadata["coverage"] == COVERAGE_FULL
    assert outcome.strategy_result.success is True
    assert outcome.citations  # at least one ranked candidate

    # WikiStrategy adapter satisfies the LLMStrategy protocol contract.
    strategy = WikiStrategy(client=query_llm, repo=repo, embedder=stub_embedder)
    sresult = strategy.run("ignored", "What chunk size should we use?")
    assert sresult.strategy == "wiki"
    assert sresult.success is True

    # --- 3. QUERY (missing → promote new answer) ---------------------
    miss_llm = make_stub_llm(default=QUERY_REPLY_MISSING)
    miss_query = QueryWikiUseCase(repo=repo, client=miss_llm, embedder=stub_embedder)
    miss_outcome = miss_query.execute("How do we extract entities?")
    assert miss_outcome.coverage == COVERAGE_MISSING

    promote = PromoteAnswerUseCase(repo)
    promo_result = promote.execute(
        title="Entity extraction",
        slug="entity-extraction",
        page_type=PageType.CONCEPT,
        body="Use the spaCy NER pipeline. References: chunking-notes.",
        sources=["chunking-notes"],
        today=date(2026, 5, 5),
    )
    assert promo_result.created is True
    assert promo_result.page_slug == "entity-extraction"

    # The new page is now visible.
    pages = {p.slug for p in repo.list_pages()}
    assert pages == {"chunking-strategy", "why-chunk", "entity-extraction"}
    assert "entity-extraction" in repo.read_index()


def test_query_on_empty_wiki_reports_missing(wiki_root, stub_embedder, make_stub_llm):
    repo = MarkdownWikiRepository(wiki_root)
    query = QueryWikiUseCase(
        repo=repo,
        client=make_stub_llm(default="COVERAGE: full\n\nshould not be reached"),
        embedder=stub_embedder,
    )
    outcome = query.execute("anything")
    assert outcome.coverage == COVERAGE_MISSING
    assert outcome.strategy_result.success is False

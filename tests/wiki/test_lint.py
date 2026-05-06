# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""LintWikiUseCase tests — frontmatter strict, link-rot warning."""

from __future__ import annotations

from datetime import date

from rlmkit.application.use_cases.wiki.lint import (
    LINT_BROKEN_LINK,
    LINT_ORPHAN_PAGE,
    LINT_STALE_SOURCE,
    LintWikiUseCase,
)
from rlmkit.domain.wiki import LintSeverity, PageStatus, PageType, WikiPage
from rlmkit.infrastructure.wiki import MarkdownWikiRepository


def _page(**overrides):
    base = dict(
        title="Thing",
        slug="thing",
        type=PageType.CONCEPT,
        sources=["src1"],
        status=PageStatus.DRAFT,
        created=date(2026, 5, 5),
        updated=date(2026, 5, 5),
        body="body",
    )
    base.update(overrides)
    return WikiPage(**base)


def test_clean_wiki_passes(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.write_raw("src1", "raw")
    repo.write_page(_page())
    report = LintWikiUseCase(repo).execute()
    assert report.passed
    assert not report.errors


def test_orphan_page_warns(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.write_page(_page(sources=[]))
    report = LintWikiUseCase(repo).execute()
    assert report.passed  # warnings don't block
    codes = {i.code for i in report.warnings}
    assert LINT_ORPHAN_PAGE in codes


def test_stale_source_warns(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.write_page(_page(sources=["nonexistent"]))
    report = LintWikiUseCase(repo).execute()
    assert report.passed
    codes = {i.code for i in report.warnings}
    assert LINT_STALE_SOURCE in codes


def test_broken_intra_wiki_link_warns(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.write_raw("src1", "raw")
    repo.write_page(_page(body="See [other](concepts/missing.md) for details."))
    report = LintWikiUseCase(repo).execute()
    assert report.passed
    codes = {i.code for i in report.warnings}
    assert LINT_BROKEN_LINK in codes


def test_external_link_does_not_warn(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.write_raw("src1", "raw")
    repo.write_page(_page(body="See [paper](https://example.com/paper.pdf)."))
    report = LintWikiUseCase(repo).execute()
    assert all(i.code != LINT_BROKEN_LINK for i in report.warnings)


def test_severity_split(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.write_page(_page(sources=[]))
    report = LintWikiUseCase(repo).execute()
    assert all(i.severity is LintSeverity.WARNING for i in report.warnings)
    assert not report.errors

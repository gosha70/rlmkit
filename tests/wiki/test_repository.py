# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""MarkdownWikiRepository tests against tmp_path."""

from __future__ import annotations

from datetime import date

from rlmkit.application.ports.wiki_port import WikiRepositoryPort
from rlmkit.domain.wiki import PageStatus, PageType, WikiPage
from rlmkit.infrastructure.wiki import MarkdownWikiRepository


def _page(slug="thing", body="hello"):
    return WikiPage(
        title="Thing",
        slug=slug,
        type=PageType.CONCEPT,
        sources=["src1"],
        status=PageStatus.DRAFT,
        created=date(2026, 5, 5),
        updated=date(2026, 5, 5),
        body=body,
    )


def test_repository_satisfies_port_protocol(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    assert isinstance(repo, WikiRepositoryPort)


def test_write_page_creates_file_and_lists_it(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    is_new = repo.write_page(_page())
    assert is_new is True
    pages = repo.list_pages()
    assert len(pages) == 1
    assert pages[0].slug == "thing"


def test_write_page_overwrite_returns_false(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.write_page(_page())
    assert repo.write_page(_page(body="updated")) is False


def test_raw_round_trip(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.write_raw("alpha", "raw text")
    assert repo.read_raw("alpha") == "raw text"
    assert repo.list_raws() == ["alpha"]


def test_log_appends(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.append_log("first")
    repo.append_log("second")
    log = repo.read_log()
    assert "first" in log
    assert "second" in log
    assert log.count("\n") == 2


def test_list_pages_skips_index_and_log(wiki_root):
    repo = MarkdownWikiRepository(wiki_root)
    repo.write_page(_page())
    repo.write_index("# index")
    repo.append_log("entry")
    assert {p.slug for p in repo.list_pages()} == {"thing"}

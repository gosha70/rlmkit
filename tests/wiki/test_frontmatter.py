# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Frontmatter parser/serializer round-trip tests."""

from __future__ import annotations

from datetime import date

import pytest

from rlmkit.domain.wiki import PageStatus, PageType, WikiPage
from rlmkit.infrastructure.wiki.frontmatter import (
    FrontmatterError,
    page_from_text,
    parse_frontmatter,
    serialize_page,
)


def test_parse_frontmatter_separates_yaml_and_body():
    text = """---
title: Hello
slug: hello
type: concept
sources: [a]
status: draft
created: 2026-05-05
updated: 2026-05-05
---
Body text here.
"""
    fm, body = parse_frontmatter(text)
    assert fm["title"] == "Hello"
    assert fm["slug"] == "hello"
    assert body.strip() == "Body text here."


def test_parse_frontmatter_missing_close_delim_raises():
    bad = "---\ntitle: oops\n"
    with pytest.raises(FrontmatterError):
        parse_frontmatter(bad)


def test_round_trip_preserves_fields():
    original = WikiPage(
        title="Chunking strategy",
        slug="chunking-strategy",
        type=PageType.CONCEPT,
        sources=["chunking-notes"],
        status=PageStatus.DRAFT,
        created=date(2026, 5, 5),
        updated=date(2026, 5, 5),
        body="Chunks should be 1000 tokens with 100 overlap.",
    )
    text = serialize_page(original)
    parsed = page_from_text(text)
    assert parsed.title == original.title
    assert parsed.slug == original.slug
    assert parsed.type is PageType.CONCEPT
    assert parsed.sources == ["chunking-notes"]
    assert parsed.status is PageStatus.DRAFT
    assert parsed.created == date(2026, 5, 5)
    assert parsed.body.strip() == original.body


def test_page_from_text_rejects_missing_required_fields():
    text = "---\ntitle: only\n---\nbody\n"
    with pytest.raises(FrontmatterError):
        page_from_text(text)


def test_page_from_text_rejects_unknown_type():
    text = """---
title: x
slug: x
type: bogus
sources: []
status: draft
created: 2026-05-05
updated: 2026-05-05
---
body
"""
    with pytest.raises(FrontmatterError):
        page_from_text(text)

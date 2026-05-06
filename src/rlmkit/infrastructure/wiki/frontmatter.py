# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""YAML frontmatter parser/serializer for wiki markdown pages.

A wiki page on disk looks like::

    ---
    title: Chunking strategy
    slug: chunking-strategy
    type: concept
    sources: [chunking-notes]
    status: draft
    created: 2026-05-05
    updated: 2026-05-05
    ---
    Body markdown goes here...

This module is the only place that knows about that exact serialization.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import yaml

from rlmkit.domain.wiki import PageStatus, PageType, WikiPage

FRONTMATTER_DELIM = "---"

REQUIRED_FRONTMATTER_FIELDS = (
    "title",
    "slug",
    "type",
    "sources",
    "status",
    "created",
    "updated",
)


class FrontmatterError(ValueError):
    """Raised when a page's frontmatter cannot be parsed or is missing fields."""


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown file into its YAML frontmatter dict and body text.

    Returns:
        (frontmatter_dict, body) — frontmatter_dict is empty if no
        frontmatter block is present.
    """
    stripped = text.lstrip()
    if not stripped.startswith(FRONTMATTER_DELIM):
        return {}, text

    # Locate the closing delimiter.
    rest = stripped[len(FRONTMATTER_DELIM):].lstrip("\n")
    end = rest.find(f"\n{FRONTMATTER_DELIM}")
    if end == -1:
        raise FrontmatterError("Unterminated frontmatter block (missing closing '---').")

    yaml_block = rest[:end]
    body = rest[end + len(FRONTMATTER_DELIM) + 1:].lstrip("\n")
    try:
        data = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as exc:
        raise FrontmatterError(f"Invalid YAML frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        raise FrontmatterError("Frontmatter must be a YAML mapping.")
    return data, body


def page_from_text(text: str) -> WikiPage:
    """Parse a full markdown file into a WikiPage. Raises FrontmatterError."""
    data, body = parse_frontmatter(text)
    missing = [f for f in REQUIRED_FRONTMATTER_FIELDS if f not in data]
    if missing:
        raise FrontmatterError(f"Missing required frontmatter fields: {missing}")
    try:
        page_type = PageType(data["type"])
    except ValueError as exc:
        raise FrontmatterError(f"Unknown page type: {data['type']!r}") from exc
    try:
        page_status = PageStatus(data["status"])
    except ValueError as exc:
        raise FrontmatterError(f"Unknown status: {data['status']!r}") from exc

    sources = data["sources"] or []
    if not isinstance(sources, list):
        raise FrontmatterError("`sources` must be a list.")

    return WikiPage(
        title=str(data["title"]),
        slug=str(data["slug"]),
        type=page_type,
        sources=[str(s) for s in sources],
        status=page_status,
        created=_coerce_date(data["created"]),
        updated=_coerce_date(data["updated"]),
        body=body,
    )


def serialize_page(page: WikiPage) -> str:
    """Render a WikiPage as a markdown file with YAML frontmatter."""
    fm: dict[str, Any] = {
        "title": page.title,
        "slug": page.slug,
        "type": page.type.value,
        "sources": list(page.sources),
        "status": page.status.value,
        "created": _coerce_date(page.created).isoformat() if page.created else None,
        "updated": _coerce_date(page.updated).isoformat() if page.updated else None,
    }
    yaml_block = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip()
    body = page.body.rstrip("\n")
    return f"{FRONTMATTER_DELIM}\n{yaml_block}\n{FRONTMATTER_DELIM}\n{body}\n"


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise FrontmatterError(f"Cannot interpret date value: {value!r}")

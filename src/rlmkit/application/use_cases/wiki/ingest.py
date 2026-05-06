# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Ingest a raw source into the wiki layer.

The use case:
1. Mirrors the raw content under ``knowledge/raw/<source-id>.md``.
2. Asks the LLM to draft / update wiki pages, returning a YAML document.
3. Parses the YAML, writes/updates each page, and rebuilds the index.
4. Appends an entry to ``log.md``.

The LLM contract is the simplest possible YAML reply:

    pages:
      - title: ...
        slug: ...
        type: concept
        sources: [<source-id>]
        status: draft
        body: |
          ...
"""

from __future__ import annotations

from datetime import date
from typing import Any

import yaml

from rlmkit.application.ports.wiki_port import WikiRepositoryPort
from rlmkit.core.rlm import LLMClient
from rlmkit.domain.wiki import (
    IngestResult,
    PageStatus,
    PageType,
    WikiPage,
)
from rlmkit.infrastructure.wiki.index_writer import build_index
from rlmkit.prompts.wiki import get_wiki_prompt

WIKI_INGEST_PROMPT_NAME = "wiki_ingest"


class IngestSourceUseCase:
    """Mirror a raw source into the wiki layer."""

    def __init__(self, repo: WikiRepositoryPort, client: LLMClient):
        self.repo = repo
        self.client = client

    def execute(
        self,
        source_id: str,
        content: str,
        *,
        today: date | None = None,
    ) -> IngestResult:
        today = today or date.today()
        # 1. Mirror raw.
        self.repo.write_raw(source_id, content)

        # 2. Ask the LLM for a YAML page list.
        messages = [
            {"role": "system", "content": get_wiki_prompt(WIKI_INGEST_PROMPT_NAME)},
            {
                "role": "user",
                "content": (
                    f"Source id: {source_id}\n\n"
                    f"--- Raw content ---\n{content}\n--- End raw content ---\n\n"
                    "Reply with the YAML page list now."
                ),
            },
        ]
        reply = self.client.complete(messages)
        page_dicts = _parse_pages_yaml(reply)

        # 3. Write each page.
        result = IngestResult(source_id=source_id)
        for raw_page in page_dicts:
            page = _coerce_page(raw_page, source_id, today)
            is_new = self.repo.write_page(page)
            if is_new:
                result.pages_created.append(page.relative_path())
            else:
                result.pages_updated.append(page.relative_path())

        # 4. Rebuild the index.
        self.repo.write_index(build_index(self.repo.list_pages()))

        # 5. Log entry.
        result.log_entry = (
            f"{today.isoformat()} ingest source={source_id} "
            f"created={len(result.pages_created)} updated={len(result.pages_updated)}"
        )
        self.repo.append_log(result.log_entry)
        return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_pages_yaml(reply: str) -> list[dict[str, Any]]:
    text = _strip_code_fence(reply)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"LLM reply is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or "pages" not in data:
        raise ValueError("LLM reply must contain a top-level `pages:` key.")
    pages = data["pages"]
    if not isinstance(pages, list):
        raise ValueError("`pages` must be a list.")
    return pages


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # drop the opening fence
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.endswith("```"):
            s = s.rsplit("\n```", 1)[0]
    return s


def _coerce_page(raw: dict[str, Any], source_id: str, today: date) -> WikiPage:
    required = ("title", "slug", "type", "body")
    missing = [f for f in required if f not in raw]
    if missing:
        raise ValueError(f"LLM page dict missing fields: {missing}")
    page_type = PageType(raw["type"])
    sources = list(raw.get("sources") or [])
    if source_id not in sources:
        sources.append(source_id)
    status = PageStatus(raw.get("status", "draft"))
    return WikiPage(
        title=str(raw["title"]),
        slug=str(raw["slug"]),
        type=page_type,
        sources=sources,
        status=status,
        created=today,
        updated=today,
        body=str(raw["body"]).rstrip(),
    )

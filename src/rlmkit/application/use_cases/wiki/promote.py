# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Promote an answer (typically from a wiki_rlm fallback) into a wiki page."""

from __future__ import annotations

from datetime import date

from rlmkit.application.ports.wiki_port import WikiRepositoryPort
from rlmkit.domain.wiki import (
    PageStatus,
    PageType,
    PromoteResult,
    WikiPage,
)
from rlmkit.infrastructure.wiki.index_writer import build_index


class PromoteAnswerUseCase:
    """Turn a free-form answer into a structured wiki page."""

    def __init__(self, repo: WikiRepositoryPort):
        self.repo = repo

    def execute(
        self,
        *,
        title: str,
        slug: str,
        page_type: PageType,
        body: str,
        sources: list[str],
        today: date | None = None,
    ) -> PromoteResult:
        today = today or date.today()
        page = WikiPage(
            title=title,
            slug=slug,
            type=page_type,
            sources=sources,
            status=PageStatus.DRAFT,
            created=today,
            updated=today,
            body=body.rstrip(),
        )
        is_new = self.repo.write_page(page)
        self.repo.write_index(build_index(self.repo.list_pages()))
        log_entry = (
            f"{today.isoformat()} promote slug={slug} type={page_type.value} "
            f"new={is_new} sources={sources}"
        )
        self.repo.append_log(log_entry)
        return PromoteResult(
            page_slug=slug,
            page_type=page_type,
            created=is_new,
            log_entry=log_entry,
        )

# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki-only strategy — answer from the persisted wiki layer.

Adapter around ``QueryWikiUseCase`` so the wiki mode satisfies the same
``LLMStrategy`` protocol as DirectStrategy / RAGStrategy / RLMStrategy.
"""

from __future__ import annotations

from rlmkit.application.ports.wiki_port import WikiRepositoryPort
from rlmkit.application.use_cases.wiki.query import MODE_WIKI, QueryWikiUseCase
from rlmkit.core.rlm import LLMClient

from .base import StrategyResult
from .embeddings import EmbeddingProvider


class WikiStrategy:
    """LLMStrategy that queries a persisted wiki layer.

    `content` is ignored — the wiki is the source of truth. Pass the wiki
    root via the repo argument at construction time.
    """

    def __init__(
        self,
        client: LLMClient,
        repo: WikiRepositoryPort,
        embedder: EmbeddingProvider,
        top_k: int = 5,
    ):
        self._use_case = QueryWikiUseCase(
            repo=repo,
            client=client,
            embedder=embedder,
            top_k=top_k,
        )

    @property
    def name(self) -> str:
        return MODE_WIKI

    def run(self, content: str, query: str) -> StrategyResult:
        # `content` is intentionally ignored; the wiki is on disk.
        del content
        return self._use_case.execute(query).strategy_result

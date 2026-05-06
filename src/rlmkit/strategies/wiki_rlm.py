# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki + RLM fallback strategy.

Tries the wiki first; if coverage is `missing` (or `partial` and below the
configured threshold), it falls back to the existing RLM controller over the
raw sources linked from the candidate wiki pages, or all raw sources if the
wiki returned no candidates.
"""

from __future__ import annotations

from rlmkit.application.ports.wiki_port import WikiRepositoryPort
from rlmkit.application.use_cases.wiki.query import (
    COVERAGE_FULL,
    COVERAGE_MISSING,
    QueryWikiUseCase,
)
from rlmkit.config import RLMConfig
from rlmkit.core.rlm import LLMClient

from .base import StrategyResult
from .embeddings import EmbeddingProvider
from .rlm_strategy import RLMStrategy

MODE_WIKI_RLM = "wiki_rlm"

FALLBACK_BACKEND_KEY = "fallback_backend"
FALLBACK_NONE = "none"
FALLBACK_RLM = "rlm"


class WikiRLMStrategy:
    """Wiki-first; RLM fallback when coverage is insufficient."""

    def __init__(
        self,
        client: LLMClient,
        repo: WikiRepositoryPort,
        embedder: EmbeddingProvider,
        rlm_config: RLMConfig | None = None,
        top_k: int = 5,
        fallback_on_partial: bool = True,
    ):
        self.client = client
        self.repo = repo
        self.embedder = embedder
        self.rlm_config = rlm_config or RLMConfig()
        self.top_k = top_k
        self.fallback_on_partial = fallback_on_partial
        self._wiki = QueryWikiUseCase(
            repo=repo,
            client=client,
            embedder=embedder,
            top_k=top_k,
        )

    @property
    def name(self) -> str:
        return MODE_WIKI_RLM

    def run(self, content: str, query: str) -> StrategyResult:
        # `content` is mostly ignored, but if the caller provided extra raw
        # text it is honoured by being prepended to the RLM-fallback content.
        wiki_outcome = self._wiki.execute(query)

        if self._wiki_sufficient(wiki_outcome.coverage):
            wiki_outcome.strategy_result.strategy = MODE_WIKI_RLM
            wiki_outcome.strategy_result.metadata[FALLBACK_BACKEND_KEY] = FALLBACK_NONE
            return wiki_outcome.strategy_result

        # Wiki was insufficient → RLM fallback over the linked raw sources.
        raw_text = self._collect_raws_for_fallback(wiki_outcome.candidate_pages, content)
        rlm = RLMStrategy(client=self.client, config=self.rlm_config)
        rlm_result = rlm.run(raw_text, query)

        # Decorate with fallback metadata.
        rlm_result.strategy = MODE_WIKI_RLM
        rlm_result.metadata = {
            **rlm_result.metadata,
            FALLBACK_BACKEND_KEY: FALLBACK_RLM,
            "wiki_coverage": wiki_outcome.coverage,
            "wiki_candidates": [p.slug for p in wiki_outcome.candidate_pages],
        }
        return rlm_result

    # ------------------------------------------------------------------

    def _wiki_sufficient(self, coverage: str) -> bool:
        if coverage == COVERAGE_FULL:
            return True
        if coverage == COVERAGE_MISSING:
            return False
        # partial → governed by config
        return not self.fallback_on_partial

    def _collect_raws_for_fallback(self, candidates, extra: str) -> str:
        ids: list[str] = []
        for page in candidates:
            for src in page.sources:
                if src not in ids:
                    ids.append(src)
        if not ids:
            ids = self.repo.list_raws()
        chunks: list[str] = []
        if extra:
            chunks.append(extra)
        for src_id in ids:
            try:
                chunks.append(f"# Source: {src_id}\n\n{self.repo.read_raw(src_id)}")
            except FileNotFoundError:
                continue
        return "\n\n---\n\n".join(chunks)

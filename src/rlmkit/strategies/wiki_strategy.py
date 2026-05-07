# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki strategies — first-class entries in the strategy registry.

Two classes that satisfy the LLMStrategy protocol:

  - WikiStrategy     — runs an index-first wiki query through a
                       WikiBackend; returns the answer as a
                       StrategyResult so it slots into
                       MultiStrategyEvaluator alongside Direct/RAG/RLM.

  - WikiRLMStrategy  — first runs WikiStrategy.query; if the wiki
                       answer is empty / has too few citations / the
                       caller passes ``force_rlm=True``, falls back
                       to the recursive RLM controller with the
                       loaded wiki pages joined as the document
                       substrate. This is the issue #37 mode-D
                       differentiator: "use RLMKit where it is
                       strongest" — large, cross-document synthesis
                       — when the distilled wiki layer is thin.
"""

from __future__ import annotations

import time
from pathlib import Path

from rlmkit.application.sandbox_vars import (
    MODE_WIKI,
    MODE_WIKI_RLM,
    TRACE_KEY_CONTENT,
    TRACE_KEY_MODE,
    TRACE_KEY_ROLE,
    TRACE_KEY_STEP,
)
from rlmkit.config import RLMConfig
from rlmkit.core.budget import TokenUsage, estimate_tokens
from rlmkit.core.rlm import RLM, LLMClient
from rlmkit.strategies.wiki.backends import (
    DeterministicTestBackend,
    LLMClientWikiBackend,
    WikiBackend,
)
from rlmkit.strategies.wiki.querier import (
    DEFAULT_MAX_PAGES,
    _select_pages,
    query as wiki_query,
)

from .base import StrategyResult


class WikiStrategy:
    """Index-first wiki query as a StrategyResult-producing strategy."""

    def __init__(
        self,
        wiki_dir: Path,
        backend: WikiBackend | None = None,
        client: LLMClient | None = None,
        max_pages: int = DEFAULT_MAX_PAGES,
        audit_log_path: Path | None = None,
    ) -> None:
        if backend is None and client is None:
            raise ValueError(
                "WikiStrategy requires either a WikiBackend or an LLMClient"
            )
        if backend is None:
            backend = LLMClientWikiBackend(client, name="llm")
        self.wiki_dir = Path(wiki_dir)
        self.backend = backend
        self.max_pages = max_pages
        self.audit_log_path = audit_log_path

    @property
    def name(self) -> str:
        return MODE_WIKI

    def run(self, content: str, query: str) -> StrategyResult:
        """LLMStrategy entry point.

        ``content`` is ignored — the wiki itself is the substrate.
        Strategy comparison frameworks pass the same ``content`` to
        every strategy, so we accept it and discard it. The query
        operation reads index.md + N relevant pages.
        """
        del content
        start = time.time()
        try:
            ans = wiki_query(
                question=query,
                wiki_dir=self.wiki_dir,
                backend=self.backend,
                max_pages=self.max_pages,
                audit_log_path=self.audit_log_path,
            )
        except Exception as exc:  # noqa: BLE001
            return StrategyResult(
                strategy=MODE_WIKI,
                answer="",
                success=False,
                error=str(exc),
                elapsed_time=time.time() - start,
            )
        elapsed = time.time() - start
        tokens = TokenUsage()
        tokens.add_input(estimate_tokens(query))
        tokens.add_output(estimate_tokens(ans.answer))
        return StrategyResult(
            strategy=MODE_WIKI,
            answer=ans.answer,
            success=True,
            steps=1,
            tokens=tokens,
            elapsed_time=elapsed,
            trace=[
                {
                    TRACE_KEY_STEP: 1,
                    TRACE_KEY_ROLE: "retrieval",
                    "pages_loaded": list(ans.pages_loaded),
                    "citation_count": len(ans.citations),
                },
                {
                    TRACE_KEY_STEP: 2,
                    TRACE_KEY_ROLE: "assistant",
                    TRACE_KEY_CONTENT: ans.answer,
                    TRACE_KEY_MODE: MODE_WIKI,
                },
            ],
            metadata={
                "pages_loaded": list(ans.pages_loaded),
                "citation_count": len(ans.citations),
                "citations": [
                    {"page": c.page, "fragment": c.fragment}
                    for c in ans.citations
                ],
                "wiki_dir": str(self.wiki_dir),
            },
        )


class WikiRLMStrategy:
    """Wiki + RLM — wiki query first, recursive controller as fallback.

    Fallback triggers (any of):
      - wiki query returned an empty answer
      - wiki query returned fewer than ``min_citations`` citations
      - ``force_rlm=True`` was passed at construction
    On fallback the loaded wiki pages (or the full wiki if none were
    matched) are joined into a single document and handed to the RLM
    controller as ``content``.
    """

    def __init__(
        self,
        wiki_dir: Path,
        client: LLMClient,
        backend: WikiBackend | None = None,
        max_pages: int = DEFAULT_MAX_PAGES,
        rlm_config: RLMConfig | None = None,
        force_rlm: bool = False,
        min_citations: int = 1,
        audit_log_path: Path | None = None,
    ) -> None:
        self.wiki_dir = Path(wiki_dir)
        self.client = client
        self.backend = backend or LLMClientWikiBackend(client, name="llm")
        self.max_pages = max_pages
        self.rlm_config = rlm_config or RLMConfig()
        self.force_rlm = force_rlm
        self.min_citations = min_citations
        self.audit_log_path = audit_log_path

    @property
    def name(self) -> str:
        return MODE_WIKI_RLM

    def _coverage_is_weak(self, ans) -> bool:
        if not ans.answer.strip():
            return True
        if len(ans.citations) < self.min_citations:
            return True
        return False

    def _gather_substrate(self, query: str) -> str:
        """Fall-back substrate for the RLM controller — the loaded wiki pages.

        If index-first selection found no matches, we fall back to
        index.md alone so the controller still has something to work
        with. The full wiki is intentionally NOT loaded; the
        controller's recursive exploration is what reads beyond the
        initial substrate.
        """
        index_md, _pages_loaded, pages = _select_pages(
            self.wiki_dir, query, self.max_pages
        )
        if not pages:
            return index_md or ""
        parts = [f"# index.md\n\n{index_md}\n"]
        for path, text in pages.items():
            parts.append(f"# {path}\n\n{text}\n")
        return "\n\n".join(parts)

    def run(self, content: str, query: str) -> StrategyResult:
        del content
        start = time.time()
        # Phase 1: wiki query.
        wiki_strat = WikiStrategy(
            wiki_dir=self.wiki_dir,
            backend=self.backend,
            max_pages=self.max_pages,
            audit_log_path=self.audit_log_path,
        )
        wiki_result = wiki_strat.run(content="", query=query)
        used_rlm = False
        rlm_result = None
        if self.force_rlm or self._coverage_is_weak_from(wiki_result):
            used_rlm = True
            substrate = self._gather_substrate(query)
            rlm = RLM(client=self.client, config=self.rlm_config)
            try:
                rlm_result = rlm.run(prompt=substrate, query=query)
            except Exception as exc:  # noqa: BLE001
                return StrategyResult(
                    strategy=MODE_WIKI_RLM,
                    answer=wiki_result.answer,
                    success=False,
                    error=f"RLM fallback failed: {exc}",
                    elapsed_time=time.time() - start,
                    trace=wiki_result.trace,
                    metadata={**wiki_result.metadata, "fallback": "rlm-error"},
                )
        elapsed = time.time() - start

        if used_rlm and rlm_result is not None:
            answer = rlm_result.answer or wiki_result.answer
            steps = wiki_result.steps + rlm_result.steps
            tokens = TokenUsage()
            tokens.add_input(wiki_result.tokens.input_tokens)
            tokens.add_output(wiki_result.tokens.output_tokens)
            tokens.add_input(estimate_tokens(query))
            for item in rlm_result.trace:
                if item.get("role") == "assistant":
                    tokens.add_output(estimate_tokens(item.get("content", "")))
            trace = list(wiki_result.trace) + list(rlm_result.trace)
            metadata = {
                **wiki_result.metadata,
                "fallback": "rlm",
                "rlm_steps": rlm_result.steps,
                "rlm_success": rlm_result.success,
            }
            return StrategyResult(
                strategy=MODE_WIKI_RLM,
                answer=answer,
                success=rlm_result.success and wiki_result.success,
                steps=steps,
                tokens=tokens,
                elapsed_time=elapsed,
                trace=trace,
                metadata=metadata,
                error=rlm_result.error,
            )
        else:
            return StrategyResult(
                strategy=MODE_WIKI_RLM,
                answer=wiki_result.answer,
                success=wiki_result.success,
                steps=wiki_result.steps,
                tokens=wiki_result.tokens,
                elapsed_time=elapsed,
                trace=wiki_result.trace,
                metadata={**wiki_result.metadata, "fallback": "none"},
                error=wiki_result.error,
            )

    def _coverage_is_weak_from(self, sr: StrategyResult) -> bool:
        if not sr.success:
            return True
        if not sr.answer.strip():
            return True
        meta = sr.metadata or {}
        if int(meta.get("citation_count", 0)) < self.min_citations:
            return True
        return False

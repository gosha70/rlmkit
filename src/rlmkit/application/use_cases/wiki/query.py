# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Query the wiki layer (wiki-first retrieval).

The query use case:
1. Embeds the user question.
2. Embeds wiki pages (title + body), ranks by cosine similarity.
3. Picks top-k pages, builds a context block with citations.
4. Calls the LLM and returns a StrategyResult with citations + coverage flag.

Coverage detection: the LLM is asked to start its reply with one of
``COVERAGE: full`` or ``COVERAGE: partial`` or ``COVERAGE: missing``. The
WikiRLMStrategy uses this header to decide whether to fall back.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from rlmkit.application.ports.wiki_port import WikiRepositoryPort
from rlmkit.core.budget import TokenUsage, estimate_tokens
from rlmkit.core.rlm import LLMClient
from rlmkit.domain.wiki import WikiCitation, WikiPage
from rlmkit.prompts.wiki import get_wiki_prompt
from rlmkit.strategies.base import StrategyResult
from rlmkit.strategies.embeddings import EmbeddingProvider

WIKI_QUERY_PROMPT_NAME = "wiki_query"

COVERAGE_HEADER_PREFIX = "COVERAGE:"
COVERAGE_FULL = "full"
COVERAGE_PARTIAL = "partial"
COVERAGE_MISSING = "missing"
VALID_COVERAGE = frozenset({COVERAGE_FULL, COVERAGE_PARTIAL, COVERAGE_MISSING})

# Mode constant — kept here so callers don't depend on sandbox_vars import order.
MODE_WIKI = "wiki"


@dataclass
class WikiQueryResult:
    """Wiki query outcome — wraps StrategyResult with parsed coverage + citations."""

    strategy_result: StrategyResult
    coverage: str  # one of VALID_COVERAGE
    citations: list[WikiCitation]
    candidate_pages: list[WikiPage]


class QueryWikiUseCase:
    """Wiki-first retrieval + LLM answer with coverage flag."""

    def __init__(
        self,
        repo: WikiRepositoryPort,
        client: LLMClient,
        embedder: EmbeddingProvider,
        top_k: int = 5,
    ):
        self.repo = repo
        self.client = client
        self.embedder = embedder
        self.top_k = top_k

    def execute(self, question: str) -> WikiQueryResult:
        start = time.time()
        pages = self.repo.list_pages()
        if not pages:
            empty = StrategyResult(
                strategy=MODE_WIKI,
                answer="",
                success=False,
                error="Wiki is empty.",
                elapsed_time=time.time() - start,
            )
            return WikiQueryResult(empty, COVERAGE_MISSING, [], [])

        ranked = self._rank(question, pages)
        candidates = [page for _score, page in ranked[: self.top_k]]
        context = self._assemble_context(ranked[: self.top_k])
        system_prompt = get_wiki_prompt(WIKI_QUERY_PROMPT_NAME)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Wiki context:\n{context}\n\nQuestion: {question}"},
        ]
        try:
            answer = self.client.complete(messages)
        except Exception as exc:  # noqa: BLE001
            failed = StrategyResult(
                strategy=MODE_WIKI,
                answer="",
                success=False,
                error=str(exc),
                elapsed_time=time.time() - start,
            )
            return WikiQueryResult(failed, COVERAGE_MISSING, [], candidates)

        coverage, body = _parse_coverage_header(answer)
        citations = _build_citations(ranked[: self.top_k])

        tokens = TokenUsage()
        tokens.add_input(estimate_tokens(context + question + system_prompt))
        tokens.add_output(estimate_tokens(answer))

        result = StrategyResult(
            strategy=MODE_WIKI,
            answer=body,
            steps=1,
            tokens=tokens,
            elapsed_time=time.time() - start,
            metadata={
                "coverage": coverage,
                "citations": [c.page_slug for c in citations],
                "candidates": [p.slug for p in candidates],
                "top_scores": [round(s, 4) for s, _ in ranked[: self.top_k]],
            },
        )
        return WikiQueryResult(result, coverage, citations, candidates)

    # ------------------------------------------------------------------

    def _rank(self, question: str, pages: list[WikiPage]) -> list[tuple[float, WikiPage]]:
        page_texts = [f"{p.title}\n\n{p.body}" for p in pages]
        page_embs = self.embedder.embed(page_texts)
        q_emb = self.embedder.embed_query(question)
        scored = [
            (_cosine(q_emb, emb), page)
            for emb, page in zip(page_embs, pages, strict=False)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _assemble_context(self, scored_pages: list[tuple[float, WikiPage]]) -> str:
        out: list[str] = []
        for i, (score, page) in enumerate(scored_pages, 1):
            out.append(
                f"[{i}] slug={page.slug} type={page.type.value} "
                f"score={score:.3f} sources={page.sources}\n"
                f"# {page.title}\n\n{page.body}"
            )
        return "\n\n---\n\n".join(out)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_coverage_header(answer: str) -> tuple[str, str]:
    """Return (coverage, body). Defaults to 'partial' if header missing."""
    head, _sep, rest = answer.partition("\n")
    head = head.strip()
    if head.upper().startswith(COVERAGE_HEADER_PREFIX):
        value = head.split(":", 1)[1].strip().lower()
        if value in VALID_COVERAGE:
            return value, rest.lstrip("\n")
    # No / malformed header → assume partial so the caller can decide.
    return COVERAGE_PARTIAL, answer


def _build_citations(scored_pages: list[tuple[float, WikiPage]]) -> list[WikiCitation]:
    return [
        WikiCitation(
            page_slug=page.slug,
            page_type=page.type,
            raw_sources=list(page.sources),
            score=score,
        )
        for score, page in scored_pages
    ]

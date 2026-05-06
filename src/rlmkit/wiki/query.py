# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki-first query path with optional RLM fallback (Mode B / D).

Algorithm (per ``specs/llm-wiki-backbone/spec.md``):

1. Score every wiki page against the question by simple keyword
   overlap on slug / title / H1 / body.
2. Concatenate the top-K page bodies and ask the LLM to answer
   from them, with explicit instructions to emit
   ``INSUFFICIENT_INFORMATION`` when wiki coverage is weak.
3. If the LLM signals insufficient information *and* a raw-source
   directory plus an :class:`RLM` instance are provided, escalate
   to recursive synthesis against the raw corpus (Mode D).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import parse_frontmatter

INSUFFICIENT = "INSUFFICIENT_INFORMATION"
WORD_RE = re.compile(r"[A-Za-z0-9_\-]+")


@dataclass
class WikiQueryResult:
    """The trail of one query through the wiki-first pipeline."""

    answer: str
    pages_consulted: list[str] = field(default_factory=list)
    fell_back_to_rlm: bool = False
    rlm_trace: list[dict] | None = None


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in WORD_RE.findall(text) if len(tok) > 2}


def _score_page(question_tokens: set[str], page_path: Path) -> tuple[int, str, str]:
    """Return ``(score, page_text, slug)`` for ``page_path``.

    Score is the count of question tokens that appear in the
    union of the page's slug, title, H1, and body. Bigger = better.
    """
    text = page_path.read_text(encoding="utf-8")
    fm, fm_close = parse_frontmatter(text)
    slug = fm.get("slug", page_path.stem)
    title = fm.get("title", "")
    body = "\n".join(text.splitlines()[fm_close:])
    page_tokens = _tokens(slug + " " + title + " " + body)
    score = len(question_tokens & page_tokens)
    return score, text, slug


def _collect_wiki_pages(wiki_dir: Path) -> list[Path]:
    """Collect substantive wiki pages.

    Skip schema docs, scripts, log, and the navigational pages
    (``index.md`` at the wiki root and ``overview.md``). These
    pages link to every other page and would dominate keyword-
    overlap scoring without contributing a real answer.
    """
    return sorted(
        p
        for p in wiki_dir.rglob("*.md")
        if "schema" not in p.relative_to(wiki_dir).parts
        and "scripts" not in p.relative_to(wiki_dir).parts
        and not (p.parent == wiki_dir and p.name in ("index.md", "log.md", "overview.md"))
    )


def query_wiki(
    question: str,
    wiki_dir: Path,
    *,
    raw_dir: Path | None = None,
    rlm: Any | None = None,
    top_k: int = 3,
) -> WikiQueryResult:
    """Answer ``question`` from the wiki, escalating to RLM on miss.

    ``rlm`` must be an :class:`rlmkit.core.rlm.RLM` instance (we do
    not import it here to avoid a hard cycle on RLMKit's optional
    layers; duck-typing the ``run`` / ``client`` attributes is
    enough).
    """
    qtokens = _tokens(question)
    pages = _collect_wiki_pages(wiki_dir)
    scored = [_score_page(qtokens, p) for p in pages]
    ranked = sorted(
        zip(pages, scored, strict=True),
        key=lambda item: item[1][0],
        reverse=True,
    )
    top = [(p, score, text, slug) for p, (score, text, slug) in ranked[:top_k] if score > 0]

    if not top or rlm is None:
        # No wiki match at all — escalate immediately if we can.
        if rlm is not None and raw_dir is not None:
            return _rlm_fallback(question, raw_dir, rlm, pages_consulted=[])
        if not top:
            return WikiQueryResult(answer=INSUFFICIENT, pages_consulted=[])
        # rlm is None but we have wiki hits — return the top page text.
        slugs = [slug for _p, _s, _t, slug in top]
        joined = "\n\n---\n\n".join(text for _p, _s, text, _slug in top)
        return WikiQueryResult(answer=joined, pages_consulted=slugs)

    # Have wiki hits AND an rlm client — ask the model.
    pages_consulted = [slug for _p, _s, _t, slug in top]
    context = "\n\n---\n\n".join(text for _p, _s, text, _slug in top)
    answer = _ask_llm_from_wiki(rlm, question, context)
    if INSUFFICIENT in answer.upper() and raw_dir is not None:
        return _rlm_fallback(question, raw_dir, rlm, pages_consulted)
    return WikiQueryResult(answer=answer, pages_consulted=pages_consulted)


def _ask_llm_from_wiki(rlm: Any, question: str, context: str) -> str:
    """Ask ``rlm.client`` to answer from the assembled wiki context."""
    system = (
        "You are answering from a curated wiki. Use ONLY the wiki pages "
        "provided as context. If the answer is not present, respond with "
        f"the literal token '{INSUFFICIENT}' on its own line."
    )
    user = f"Wiki context:\n{context}\n\nQuestion: {question}"
    response: str = rlm.client.complete(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return response


def _rlm_fallback(
    question: str,
    raw_dir: Path,
    rlm: Any,
    pages_consulted: list[str],
) -> WikiQueryResult:
    """Run the recursive controller against the raw-source corpus."""
    if raw_dir.is_file():
        corpus = raw_dir.read_text(encoding="utf-8")
    else:
        chunks = []
        for child in sorted(raw_dir.rglob("*")):
            if not child.is_file():
                continue
            try:
                chunks.append(
                    f"=== {child.relative_to(raw_dir)} ===\n" + child.read_text(encoding="utf-8")
                )
            except (UnicodeDecodeError, OSError):
                continue
        corpus = "\n\n".join(chunks)
    rlm_result = rlm.run(prompt=corpus, query=question)
    return WikiQueryResult(
        answer=rlm_result.answer,
        pages_consulted=pages_consulted,
        fell_back_to_rlm=True,
        rlm_trace=rlm_result.trace,
    )

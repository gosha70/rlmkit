# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Load existing wiki state for the multi-page ingest prompt.

Algorithm (borrowed from code-copilot-team/scripts/wiki_ingest/wiki_state.py):

  1. Read ``index.md`` and ``log.md`` verbatim.
  2. List all wiki pages excluding ``schema/``, ``scripts/`` and
     the wiki-root meta pages (index.md, log.md, overview.md).
  3. Score each page by token-overlap with the source content +
     source path. Token = ``[a-z0-9][a-z0-9-]*``, minus a small
     stopword list. Page signal tokens = slug + path tokens +
     first 400 chars of body.
  4. Take the top-N (default 10) pages with score > 0,
     deterministic ordering (highest score, then path).

Trade-off: lexical only, no embeddings. Karpathy's gist makes
the explicit point that the wiki works at moderate scale
without a vector DB.
"""

from __future__ import annotations

import re
from pathlib import Path

from .entities import WikiState

DEFAULT_MAX_CANDIDATES = 10

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "which", "with",
    }
)


def read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def tokenise(text: str) -> set[str]:
    if not text:
        return set()
    return {
        tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS
    }


def list_wiki_pages(wiki_dir: Path) -> list[Path]:
    if not wiki_dir.is_dir():
        return []
    excluded_dirs = {"schema", "scripts"}
    excluded_stems = {"index", "log", "overview"}
    pages: list[Path] = []
    for p in sorted(wiki_dir.rglob("*.md")):
        rel_parts = p.relative_to(wiki_dir).parts
        if rel_parts and rel_parts[0] in excluded_dirs:
            continue
        if p.stem in excluded_stems and p.parent == wiki_dir:
            continue
        pages.append(p)
    return pages


def _score_page(page_path: Path, source_tokens: set[str]) -> int:
    if not source_tokens:
        return 0
    content = read_text_or_empty(page_path)
    if not content:
        return 0
    signal = {page_path.stem.lower()}
    signal |= tokenise(str(page_path).replace("/", " ").replace("-", " "))
    signal |= tokenise(content[:400])
    return len(source_tokens & signal)


def load_wiki_state(
    wiki_dir: Path,
    source_path: Path,
    source_content: str,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> WikiState:
    """Read index/log + a relevance-ranked candidate page set."""
    index_md = read_text_or_empty(wiki_dir / "index.md")
    log_md = read_text_or_empty(wiki_dir / "log.md")

    source_tokens = tokenise(source_content) | tokenise(str(source_path))

    scored: list[tuple[int, Path]] = []
    for page in list_wiki_pages(wiki_dir):
        score = _score_page(page, source_tokens)
        if score > 0:
            scored.append((score, page))

    scored.sort(key=lambda pair: (-pair[0], str(pair[1])))
    selected = scored[:max_candidates]

    candidate_pages: dict[str, str] = {}
    for _score, page in selected:
        rel = str(page.relative_to(wiki_dir))
        candidate_pages[rel] = read_text_or_empty(page)

    return WikiState(
        index_md=index_md,
        log_md=log_md,
        candidate_pages=candidate_pages,
    )

# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Index-first wiki query.

Algorithm (borrowed from cct/scripts/wiki_ingest/querier.py):
  1. Read index.md.
  2. Extract every wiki-relative .md link from index.md.
  3. Score each linked page by token-overlap with the question.
  4. Take the top-N (default 5).
  5. Compose a query prompt with index.md + the loaded pages.
  6. Backend returns {answer, citations[]}.
  7. Append the (question, pages_loaded) pair to
     doc_internal/wiki-query-log.jsonl for audit.

Key invariant: pages NOT linked from index.md are unreachable.
That is intentional — it makes the index the table of contents
and makes orphans visible to the linter.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

from .backends import WikiBackend
from .entities import Citation, QueryAnswer
from .errors import ContractViolationError
from .prompts import compose_query_prompt, load_schema_files
from .wiki_state import read_text_or_empty, tokenise

DEFAULT_MAX_PAGES = 5

_MD_LINK_RE = re.compile(r"\]\(([^)]+\.md)(?:#[^)]*)?\)")


def _extract_index_links(index_md: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for m in _MD_LINK_RE.finditer(index_md):
        target = m.group(1).strip()
        if not target or target.startswith(("http://", "https://", "mailto:", "../")):
            continue
        if target in seen:
            continue
        seen.add(target)
        paths.append(target)
    return paths


def _select_pages(
    wiki_dir: Path, question: str, max_pages: int
) -> tuple[str, list[str], dict[str, str]]:
    index_md = read_text_or_empty(wiki_dir / "index.md")
    if not index_md:
        return "", [], {}
    candidates = _extract_index_links(index_md)
    q_tokens = tokenise(question)
    scored: list[tuple[int, str]] = []
    for rel in candidates:
        page_path = wiki_dir / rel
        if not page_path.exists():
            continue
        content = read_text_or_empty(page_path)
        if not content:
            continue
        signal = {Path(rel).stem.lower()}
        signal |= tokenise(rel.replace("/", " ").replace("-", " "))
        signal |= tokenise(content[:400])
        score = len(q_tokens & signal)
        if score > 0:
            scored.append((score, rel))
    pos = {rel: i for i, rel in enumerate(candidates)}
    scored.sort(key=lambda pair: (-pair[0], pos.get(pair[1], 1_000_000)))
    selected = [rel for _s, rel in scored[:max_pages]]
    pages = {rel: read_text_or_empty(wiki_dir / rel) for rel in selected}
    return index_md, selected, pages


def _parse_query_response(raw: dict) -> tuple[str, list[Citation]]:
    if raw.get("version") != 1:
        raise ContractViolationError(
            f"query response version must be 1, got {raw.get('version')!r}"
        )
    answer = raw.get("answer")
    if not isinstance(answer, str):
        raise ContractViolationError("query response.answer must be a string")
    cits_raw = raw.get("citations") or []
    if not isinstance(cits_raw, list):
        raise ContractViolationError("query response.citations must be a list")
    cits: list[Citation] = []
    for c in cits_raw:
        if not isinstance(c, dict):
            continue
        page = c.get("page", "")
        fragment = c.get("fragment", "")
        if isinstance(page, str) and isinstance(fragment, str):
            cits.append(Citation(page=page, fragment=fragment))
    return answer, cits


def _append_audit_log(
    log_path: Path,
    question: str,
    pages_loaded: list[str],
    answer_len: int,
) -> None:
    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": question,
        "pages_loaded": pages_loaded,
        "answer_chars": answer_len,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        # Audit logging is best-effort — never fail the query.
        return


def query(
    question: str,
    wiki_dir: Path,
    backend: WikiBackend,
    max_pages: int = DEFAULT_MAX_PAGES,
    audit_log_path: Path | None = None,
) -> QueryAnswer:
    index_md, pages_loaded, pages = _select_pages(wiki_dir, question, max_pages)
    schema = load_schema_files()
    prompt = compose_query_prompt(question, index_md, pages, schema)
    raw = backend.invoke(prompt)
    answer, citations = _parse_query_response(raw)
    if audit_log_path is not None:
        _append_audit_log(audit_log_path, question, pages_loaded, len(answer))
    return QueryAnswer(
        answer=answer,
        citations=tuple(citations),
        pages_loaded=tuple(pages_loaded),
    )

# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki schema constants and a stdlib-only frontmatter parser.

The page-type taxonomy and frontmatter shape are borrowed verbatim
from ``code-copilot-team/knowledge/wiki/schema/``. The parser here
mirrors the awk-style logic the cct ``lint-wiki.sh`` uses, ported
to Python so RLMKit's pipeline does not need ``pyyaml``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

VALID_PAGE_TYPES: frozenset[str] = frozenset(
    {
        "concept",
        "workflow",
        "incident",
        "decision",
        "playbook",
        "glossary",
        "open-question",
        "index",
        "log",
        "overview",
    }
)

# page_type → directory (relative to wiki root) where pages of that
# type must live. ``"."`` means the wiki root.
PAGE_TYPE_DIRS: dict[str, str] = {
    "concept": "concepts",
    "workflow": "workflows",
    "incident": "incidents",
    "decision": "decisions",
    "playbook": "playbooks",
    "glossary": "glossary",
    "open-question": "open-questions",
    "index": ".",
    "log": ".",
    "overview": ".",
}

REQUIRED_FRONTMATTER_KEYS: tuple[str, ...] = (
    "page_type",
    "slug",
    "title",
    "status",
    "last_reviewed",
)

# pages exempt from `sources:` requirement (cct lint-rules §1).
SOURCES_EXEMPT_TYPES: frozenset[str] = frozenset({"index", "log"})

KEBAB_CASE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def is_kebab_case(value: str) -> bool:
    """True if value is lowercase kebab-case (per cct slug rule)."""
    return bool(KEBAB_CASE.match(value))


def expected_slug_for(file_path: Path, wiki_root: Path) -> str:
    """The slug a file at ``file_path`` ought to declare.

    Matches cct's special-case: ``<dir>/index.md`` takes the parent
    directory's name (e.g., ``glossary/index.md`` → ``glossary``);
    everything else takes the filename stem.
    """
    stem = file_path.stem
    if stem == "index":
        parent = file_path.parent
        if parent.resolve() != wiki_root.resolve():
            return parent.name
    return stem


def parse_frontmatter(text: str) -> tuple[dict[str, Any], int]:
    """Extract the YAML frontmatter block from a markdown document.

    Returns ``(frontmatter_dict, fm_close_line_number)`` where the
    close line number is 1-indexed (the line carrying the closing
    ``---``). If the document has no valid frontmatter, returns
    ``({}, 0)``.

    This is a deliberately small parser — only the keys we use:
    scalar string values, plus the ``sources:`` list whose entries
    are flat ``key: value`` mappings. Nested objects, anchors,
    multi-line scalars are out of scope.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, 0

    close_index: int | None = None
    for idx in range(1, min(len(lines), 50)):
        if lines[idx].strip() == "---":
            close_index = idx
            break
    if close_index is None:
        return {}, 0

    fm: dict[str, Any] = {}
    sources: list[dict[str, Any]] | None = None
    in_sources = False
    current_entry: dict[str, Any] | None = None

    for raw in lines[1:close_index]:
        if not raw.strip():
            in_sources = in_sources and True  # blank line keeps state
            continue

        if in_sources:
            stripped = raw.lstrip()
            indent = len(raw) - len(stripped)
            if indent == 0:
                in_sources = False
                current_entry = None
                # fall through to top-level parse below
            elif stripped.startswith("- "):
                current_entry = {}
                assert sources is not None
                sources.append(current_entry)
                kv = stripped[2:].strip()
                _absorb_scalar(current_entry, kv)
                continue
            elif current_entry is not None:
                _absorb_scalar(current_entry, stripped)
                continue

        if not in_sources:
            if raw.rstrip() == "sources:":
                sources = []
                fm["sources"] = sources
                in_sources = True
                current_entry = None
                continue

            if ":" in raw:
                key, _, value = raw.partition(":")
                fm[key.strip()] = _strip_quotes(value.strip())

    return fm, close_index + 1


def _absorb_scalar(entry: dict[str, Any], kv: str) -> None:
    """Parse a single ``key: value`` pair into a sources entry."""
    if ":" not in kv:
        return
    k, _, v = kv.partition(":")
    entry[k.strip()] = _strip_quotes(v.strip())


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def has_sources(frontmatter: dict[str, Any]) -> bool:
    """True if frontmatter declares at least one source entry."""
    sources = frontmatter.get("sources")
    return isinstance(sources, list) and len(sources) > 0

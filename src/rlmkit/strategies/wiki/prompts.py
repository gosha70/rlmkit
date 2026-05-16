# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Prompt composition for the three LLM-driven wiki tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .entities import WikiState

_SCHEMA_NAMES = ("ingest-rules", "page-types", "citation-rules", "lint-rules")
_SCHEMA_DIR = Path(__file__).resolve().parent / "schema"


def load_schema_files() -> dict[str, str]:
    """Load the bundled schema files. Raises FileNotFoundError if absent."""
    out: dict[str, str] = {}
    for name in _SCHEMA_NAMES:
        out[name] = (_SCHEMA_DIR / f"{name}.md").read_text(encoding="utf-8")
    return out


_PATCH_SET_SCHEMA = {
    "type": "object",
    "required": ["version", "rationale", "edits"],
    "properties": {
        "version": {"type": "integer", "const": 1},
        "rationale": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "action", "new_content"],
                "properties": {
                    "path": {"type": "string"},
                    "action": {
                        "enum": [
                            "create", "update", "append-log", "append-index"
                        ]
                    },
                    "new_content": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}

_QUERY_SCHEMA = {
    "type": "object",
    "required": ["version", "answer", "citations"],
    "properties": {
        "version": {"type": "integer", "const": 1},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["page", "fragment"],
                "properties": {
                    "page": {"type": "string"},
                    "fragment": {"type": "string"},
                },
            },
        },
    },
}


def compose_ingest_prompt(
    source_path: Path,
    source_content: str,
    wiki_state: WikiState,
    schema: dict[str, str] | None = None,
) -> dict[str, Any]:
    schema = schema or load_schema_files()
    return {
        "version": 1,
        "task": "ingest-multi",
        "system_instructions": (
            "You are the wiki curator. The wiki is a persistent, "
            "compounding artifact maintained over time; the existing "
            "wiki state is your working memory. Produce a multi-page "
            "WikiPatchSet that integrates the new source: update "
            "existing pages where the source extends or refines them, "
            "create new pages only when no existing page covers the "
            "topic, append a one-line dated entry to log.md, and update "
            "index.md with a link to any new page. Emit exactly one "
            "JSON object that matches the response schema; no prose, "
            "no markdown fences."
        ),
        "schema_excerpts": {
            "ingest_rules": schema.get("ingest-rules", ""),
            "page_types": schema.get("page-types", ""),
            "citation_rules": schema.get("citation-rules", ""),
        },
        "source": {
            "kind": "file",
            "path": str(source_path),
            "content": source_content,
        },
        "wiki_state": {
            "index_md": wiki_state.index_md,
            "log_md": wiki_state.log_md,
            "candidate_pages": dict(wiki_state.candidate_pages),
        },
        "response_schema": json.dumps(_PATCH_SET_SCHEMA),
    }


def compose_query_prompt(
    question: str,
    index_md: str,
    pages: dict[str, str],
    schema: dict[str, str] | None = None,
) -> dict[str, Any]:
    schema = schema or load_schema_files()
    return {
        "version": 1,
        "task": "query",
        "system_instructions": (
            "You are answering a question against a curated project "
            "wiki. Use ONLY the index and pages provided — do not "
            "fabricate wiki contents and do not draw on outside "
            "knowledge. If the answer is not in the provided "
            "material, return an empty answer string with one "
            "citation pointing at index.md. Always cite the pages "
            "you used; quote a short fragment from each."
        ),
        "question": question,
        "index_md": index_md,
        "pages": dict(pages),
        "schema_excerpts": {
            "page_types": schema.get("page-types", ""),
            "citation_rules": schema.get("citation-rules", ""),
        },
        "response_schema": json.dumps(_QUERY_SCHEMA),
    }


def compose_health_prompt(
    candidate_pairs: list[tuple[str, str, str, str]],
    schema: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Health-lint contradictions prompt.

    ``candidate_pairs`` is ``[(path_a, content_a, path_b, content_b)]``.
    """
    schema = schema or load_schema_files()
    pairs = [
        {
            "path_a": a,
            "content_a": ca,
            "path_b": b,
            "content_b": cb,
        }
        for a, ca, b, cb in candidate_pairs
    ]
    return {
        "version": 1,
        "task": "lint-health",
        "system_instructions": (
            "You are auditing a wiki for contradictions. For each "
            "page pair below, decide whether the two pages make "
            "directly conflicting claims about the same entity. "
            "Return a JSON object {version: 1, contradictions: "
            "[{page_a, page_b, description}]}. Omit pairs that "
            "merely restate the same claim or address different "
            "entities."
        ),
        "pairs": pairs,
        "response_schema": json.dumps(
            {
                "type": "object",
                "required": ["version", "contradictions"],
                "properties": {
                    "version": {"type": "integer"},
                    "contradictions": {"type": "array"},
                },
            }
        ),
    }

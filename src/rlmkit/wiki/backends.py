# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Ingest backends.

Three concrete implementations:

* ``TestBackend`` — deterministic, derives a proposal from the
  source's first H1. Used by the test suite and ``--backend test``.
* ``LLMBackend`` — single LLM completion against any object that
  satisfies the :class:`rlmkit.core.rlm.LLMClient` Protocol.
* ``RLMBackend`` — wraps a :class:`rlmkit.core.rlm.RLM` controller
  so ingest can use recursive code-execution to navigate large
  corpora. This is the wiring that makes "wiki + rlm" mode (issue
  gosha70/rlmkit#37 §A) work.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import BackendFailure
from .proposal import IngestProposal, IngestRequest

H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@runtime_checkable
class IngestBackend(Protocol):
    """Anything that can turn a source into an :class:`IngestProposal`."""

    name: str

    def ingest(self, request: IngestRequest, schema_excerpts: dict[str, str]) -> IngestProposal: ...


def _slugify(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return out or "untitled"


def _read_corpus(source_path: Path) -> str:
    """Concatenate file content (single file or directory).

    Directories are walked deterministically (sorted by relative
    path) and concatenated with ``=== <relpath> ===`` separators
    so the model — or the test backend — can address each source.
    """
    if source_path.is_file():
        return source_path.read_text(encoding="utf-8")
    parts: list[str] = []
    for child in sorted(source_path.rglob("*")):
        if not child.is_file():
            continue
        try:
            text = child.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = child.relative_to(source_path)
        parts.append(f"=== {rel} ===\n{text}")
    return "\n\n".join(parts)


def _first_h1(text: str) -> str | None:
    m = H1_RE.search(text)
    return m.group(1).strip() if m else None


@dataclass
class TestBackend:
    """Deterministic backend used by the test suite.

    Always classifies the source as a ``concept`` page, derives the
    slug from the first H1 (or filename), and emits a minimal but
    schema-valid draft body. No LLM call.
    """

    # Tell pytest not to collect this class — the "Test" prefix is
    # part of its public role (deterministic test backend), not a
    # signal that pytest should treat it as a test case.
    __test__ = False

    name: str = "test"

    def ingest(self, request: IngestRequest, schema_excerpts: dict[str, str]) -> IngestProposal:
        text = _read_corpus(request.source_path)
        title = _first_h1(text) or request.source_path.stem.replace("-", " ").title()
        slug = _slugify(title)
        sources = [{"path": str(request.source_path), "sha": "deadbee"}]
        draft = _build_concept_page(
            slug=slug,
            title=title,
            sources=sources,
            body_summary=(text or "(empty source)").strip().splitlines()[0][:200],
        )
        return IngestProposal(
            disposition="accept",
            reason="deterministic test backend always accepts",
            page_type="concept",
            slug=slug,
            title=title,
            draft_markdown=draft,
            sources=sources,
        )


@dataclass
class LLMBackend:
    """Single-call LLM backend.

    Composes a structured prompt from the schema excerpts plus the
    source content, asks the model to emit a JSON object matching
    :class:`IngestProposal`, parses the response. The Ingestor
    layer runs the two-layer validation against whatever this
    returns.

    ``client`` must satisfy :class:`rlmkit.core.rlm.LLMClient`.
    """

    client: Any
    name: str = "llm"

    def ingest(self, request: IngestRequest, schema_excerpts: dict[str, str]) -> IngestProposal:
        text = _read_corpus(request.source_path)
        prompt = _build_ingest_prompt(text, schema_excerpts, request.source_path)
        try:
            response = self.client.complete(
                [
                    {"role": "system", "content": "You are the wiki curator."},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:  # backend invocation failure
            raise BackendFailure(f"LLM backend complete() raised: {exc}") from exc
        return _parse_proposal_json(response)


@dataclass
class RLMBackend:
    """Recursive-controller backend (wiki + rlm mode).

    Loads the source corpus into the RLM REPL as the ``P``
    variable, instructs the model to use ``peek`` / ``grep`` /
    ``subcall`` to navigate it, then to emit a JSON
    :class:`IngestProposal` as ``FINAL:``. RLMKit's controller
    drives the loop and feeds intermediate execution results back.

    ``rlm`` is an :class:`rlmkit.core.rlm.RLM` instance. The
    backend is responsible for parsing the controller's final
    answer; intermediate prompt mechanics belong to RLMKit.
    """

    rlm: Any
    name: str = "rlm"

    def ingest(self, request: IngestRequest, schema_excerpts: dict[str, str]) -> IngestProposal:
        corpus = _read_corpus(request.source_path)
        query = _build_rlm_query(schema_excerpts, request.source_path)
        try:
            result = self.rlm.run(prompt=corpus, query=query)
        except Exception as exc:
            raise BackendFailure(f"RLM controller raised: {exc}") from exc
        if not result.success:
            raise BackendFailure(f"RLM controller failed: {result.error}")
        return _parse_proposal_json(result.answer)


# ── Prompt composition ─────────────────────────────────────────


def _ingest_instructions(source_label: str) -> str:
    return (
        "Apply the wiki curator's four-question gate to the source below. "
        "Reject the source if the answer to any of the four is no. "
        "On accept, draft a typed wiki page that satisfies the universal "
        "frontmatter, the page-type required H2 sections, and the citation "
        "rules.\n\n"
        f"Source: {source_label}\n\n"
        "Return EXACTLY ONE JSON object on stdout (no prose preamble, no "
        "afterword) with this shape:\n"
        "{\n"
        '  "disposition": "accept" | "reject",\n'
        '  "reason": "...",\n'
        '  "page_type": "concept" | "workflow" | ... | null,\n'
        '  "slug": "kebab-case" | null,\n'
        '  "title": "..." | null,\n'
        '  "draft_markdown": "---\\npage_type: ...\\n..." | null,\n'
        '  "sources": [{"path": "...", "sha": "..."} | {"issue": 1} | ...]\n'
        "}\n"
    )


def _build_ingest_prompt(text: str, schema_excerpts: dict[str, str], source_path: Path) -> str:
    parts = [
        _ingest_instructions(str(source_path)),
        "\n--- ingest-rules.md ---\n",
        schema_excerpts.get("ingest_rules", ""),
        "\n--- page-types.md ---\n",
        schema_excerpts.get("page_types", ""),
        "\n--- citation-rules.md ---\n",
        schema_excerpts.get("citation_rules", ""),
        "\n--- SOURCE CONTENT ---\n",
        text,
    ]
    return "\n".join(parts)


def _build_rlm_query(schema_excerpts: dict[str, str], source_path: Path) -> str:
    return (
        _ingest_instructions(str(source_path))
        + "\n\nThe corpus is loaded as variable `P` in your REPL. Use peek/grep "
        "to navigate it before drafting. When ready, return the JSON object "
        "above as your FINAL answer.\n\n"
        "--- ingest-rules.md ---\n"
        + schema_excerpts.get("ingest_rules", "")
        + "\n--- page-types.md ---\n"
        + schema_excerpts.get("page_types", "")
        + "\n--- citation-rules.md ---\n"
        + schema_excerpts.get("citation_rules", "")
    )


# ── JSON extraction (mirrors cct's strategy 1 + 2) ─────────────


def _parse_proposal_json(raw: str) -> IngestProposal:
    """Extract a JSON object from free-form text and build the proposal."""
    obj = _extract_json_object(raw)
    if obj is None:
        raise BackendFailure(
            f"backend returned no parseable JSON object; raw output (truncated): {raw[:512]!r}"
        )
    try:
        disposition = obj["disposition"]
        if disposition not in ("accept", "reject"):
            raise BackendFailure(f"invalid disposition: {disposition!r}")
        return IngestProposal(
            disposition=disposition,
            reason=obj.get("reason", ""),
            page_type=obj.get("page_type"),
            slug=obj.get("slug"),
            title=obj.get("title"),
            draft_markdown=obj.get("draft_markdown"),
            sources=obj.get("sources", []) or [],
        )
    except KeyError as exc:
        raise BackendFailure(f"backend JSON missing required key: {exc}") from exc


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first parseable JSON object out of free-form output.

    Strategy (mirroring cct):
    1. Look for a fenced ```json …``` block (or unfenced ```…```).
    2. Otherwise scan for the first balanced top-level ``{…}``
       block, skipping braces inside string literals.
    """
    for match in _FENCE_RE.finditer(text):
        candidate = match.group(1).strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        in_str = False
        escape = False
        for end in range(start, len(text)):
            ch = text[end]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        value = json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # try a later '{'
                    if isinstance(value, dict):
                        return value
                    break
        # if we never balanced, fall through to next start
    return None


# ── Page templates (used by TestBackend) ───────────────────────


def _build_concept_page(slug: str, title: str, sources: list[dict], body_summary: str) -> str:
    sources_yaml_lines: list[str] = []
    for src in sources:
        items = list(src.items())
        if not items:
            continue
        first_k, first_v = items[0]
        sources_yaml_lines.append(f"  - {first_k}: {first_v}")
        for k, v in items[1:]:
            sources_yaml_lines.append(f"    {k}: {v}")
    sources_yaml = "\n".join(sources_yaml_lines)
    return (
        f"---\n"
        f"page_type: concept\n"
        f"slug: {slug}\n"
        f"title: {title}\n"
        f"status: draft\n"
        f"last_reviewed: 2026-05-05\n"
        f"sources:\n{sources_yaml}\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"## Summary\n\n{body_summary}\n\n"
        f"## Key ideas\n\n- Drafted by the wiki test backend.\n\n"
        f"## Where this shows up\n\n- See cited sources.\n\n"
        f"## Related\n\n- (none)\n"
    )

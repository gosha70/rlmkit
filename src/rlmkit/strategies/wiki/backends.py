# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki-backend protocol + two adapters.

The wiki layer treats the LLM as a JSON-emitter for three tasks:

  - ``ingest-multi``  → produce a ``WikiPatchSet`` JSON
  - ``query``         → produce ``{answer, citations[]}`` JSON
  - ``lint-health``   → produce ``{contradictions[]}`` JSON

The cct substrate subprocesses out to ``claude -p`` /
``cursor-agent -p`` / ``codex exec``. RLMKit already has a rich
``LLMClient`` protocol with budget / retry / streaming, so we
diverge from cct and wrap that instead.

Two concrete backends:

  - ``LLMClientWikiBackend`` — wraps any ``LLMClient`` (LiteLLM
    adapter, OpenAI adapter, etc.). Composes the prompt JSON into
    a single chat message and parses the model's JSON reply.
  - ``DeterministicTestBackend`` — in-process, no network, used
    by the e2e test and ``--backend test``.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import BackendInvocationError, ContractViolationError


@runtime_checkable
class WikiBackend(Protocol):
    """Common interface for wiki backends."""

    @property
    def name(self) -> str: ...

    def invoke(self, prompt: dict[str, Any]) -> dict[str, Any]:
        """Send a wiki prompt and return the parsed JSON response."""
        ...


# ---------------------------------------------------------------------------
# LLMClient adapter
# ---------------------------------------------------------------------------


class LLMClientWikiBackend:
    """Adapt any rlmkit ``LLMClient`` to the ``WikiBackend`` protocol.

    The model is asked to emit exactly one JSON object per the
    response schema embedded in the prompt. We strip a single
    leading triple-backtick ``json`` fence if present (model
    habits) and parse the rest. Any deviation raises
    ``ContractViolationError``.
    """

    def __init__(self, client: Any, name: str = "llm") -> None:
        self.client = client
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _strip_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            # Drop the first line and the closing fence.
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        return text

    def invoke(self, prompt: dict[str, Any]) -> dict[str, Any]:
        system = prompt.get("system_instructions", "You are a wiki curator.")
        user_payload = json.dumps(prompt, indent=2)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_payload},
        ]
        try:
            raw = self.client.complete(messages)
        except Exception as exc:  # noqa: BLE001
            raise BackendInvocationError(
                f"LLM client {self._name} failed: {exc}"
            ) from exc

        cleaned = self._strip_fence(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ContractViolationError(
                f"backend {self._name} did not return valid JSON: {exc}\n"
                f"  first 400 chars of response: {cleaned[:400]!r}"
            ) from exc
        if not isinstance(data, dict):
            raise ContractViolationError(
                f"backend {self._name} returned non-object JSON: "
                f"{type(data).__name__}"
            )
        return data


# ---------------------------------------------------------------------------
# Deterministic test backend
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = _SLUG_RE.sub("-", text)
    return text.strip("-") or "untitled"


class DeterministicTestBackend:
    """In-process backend with fixed responses per task.

    Mirrors cct's ``backends/test.py`` shape so downstream tests
    can be ported without thinking. The responses are coherent
    enough to round-trip through the structural linter and the
    promoter — the e2e test exercises the full ingest → promote
    → query path with this backend alone.
    """

    @property
    def name(self) -> str:
        return "test"

    def invoke(self, prompt: dict[str, Any]) -> dict[str, Any]:
        task = prompt.get("task")
        if task == "ingest-multi":
            return self._ingest_multi(prompt)
        if task == "query":
            return self._query(prompt)
        if task == "lint-health":
            return self._lint_health(prompt)
        raise ContractViolationError(
            f"DeterministicTestBackend: unknown task {task!r}"
        )

    @staticmethod
    def _ingest_multi(prompt: dict[str, Any]) -> dict[str, Any]:
        source = prompt.get("source", {}) or {}
        path = source.get("path", "unknown")
        content = source.get("content", "") or ""
        # Pull a one-line headline from the source.
        headline = next(
            (ln.strip().lstrip("# ").strip() for ln in content.splitlines() if ln.strip()),
            "test source",
        )[:80] or "test source"
        slug = _slugify(Path(path).stem) or "test-source"
        title = headline
        date = datetime.date.today().isoformat()
        page_path = f"concepts/{slug}.md"
        page_md = (
            f"---\n"
            f"page_type: concept\n"
            f"slug: {slug}\n"
            f"title: {title}\n"
            f"status: draft\n"
            f"last_reviewed: {date}\n"
            f"sources:\n"
            f"  - path: {path}\n"
            f"---\n\n"
            f"## Summary\n\n"
            f"Auto-generated stub from the deterministic test backend "
            f"summarising '{title}'. This is a fixture, not real "
            f"distilled content.\n\n"
            f"## Key ideas\n\n"
            f"- Stub key idea: the source mentioned '{title}'.\n"
            f"- Stub key idea: round-trips through promoter cleanly.\n\n"
            f"## Where this shows up\n\n"
            f"- {path}\n\n"
            f"## Related\n\n"
            f"- [index](../index.md)\n"
        )
        return {
            "version": 1,
            "rationale": f"test backend stub for {path}",
            "edits": [
                {
                    "path": page_path,
                    "action": "create",
                    "new_content": page_md,
                    "rationale": "stub create",
                },
                {
                    "path": "log.md",
                    "action": "append-log",
                    "new_content": (
                        f"- {date} — add {slug} (concept): test-backend ingest"
                    ),
                    "rationale": "log",
                },
                {
                    "path": "index.md",
                    "action": "append-index",
                    "new_content": f"- [{title}]({page_path})",
                    "rationale": "index",
                },
            ],
        }

    @staticmethod
    def _query(prompt: dict[str, Any]) -> dict[str, Any]:
        question = prompt.get("question", "")
        pages = prompt.get("pages", {}) or {}
        # If we have pages, cite them; otherwise fall back to index.
        if pages:
            citations = [
                {
                    "page": page,
                    "fragment": (text.splitlines() or [""])[0][:80],
                }
                for page, text in list(pages.items())[:3]
            ]
            answer = (
                f"Test backend deterministic answer to: {question!r}. "
                f"Consulted {len(pages)} page(s)."
            )
        else:
            citations = [{"page": "index.md", "fragment": "wiki entry point"}]
            answer = ""
        return {"version": 1, "answer": answer, "citations": citations}

    @staticmethod
    def _lint_health(prompt: dict[str, Any]) -> dict[str, Any]:
        # Test backend never reports contradictions; structural and
        # heuristic checks live in pure-Python and don't need the LLM.
        return {"version": 1, "contradictions": []}

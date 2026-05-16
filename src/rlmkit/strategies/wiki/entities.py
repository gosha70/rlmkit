# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Frozen dataclasses for the wiki layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PageAction = Literal["create", "update", "append-log", "append-index"]
HealthKind = Literal[
    "contradiction", "stale-claim", "weak-orphan", "missing-cross-link"
]
HealthSeverity = Literal["warning", "error"]


@dataclass(frozen=True)
class PageEdit:
    """One write to the wiki.

    For ``create``/``update`` ``new_content`` is the full markdown.
    For ``append-log`` it is one log-line (e.g. ``- 2026-05-07 — add
    foo (concept): one-line why``). For ``append-index`` it is one
    bullet that should be inserted under the appropriate section in
    ``index.md``.
    """

    path: str
    action: PageAction
    new_content: str
    rationale: str = ""


@dataclass(frozen=True)
class WikiPatchSet:
    """A multi-page write plan returned by ingest."""

    edits: tuple[PageEdit, ...]
    source_path: str
    rationale: str

    @classmethod
    def of(
        cls, edits: list[PageEdit], source_path: str, rationale: str
    ) -> WikiPatchSet:
        return cls(edits=tuple(edits), source_path=source_path, rationale=rationale)


@dataclass(frozen=True)
class WikiState:
    """Snapshot of the live wiki passed into the ingest prompt."""

    index_md: str
    log_md: str
    candidate_pages: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    page: str
    fragment: str


@dataclass(frozen=True)
class QueryAnswer:
    answer: str
    citations: tuple[Citation, ...]
    pages_loaded: tuple[str, ...]


@dataclass(frozen=True)
class HealthFinding:
    kind: HealthKind
    severity: HealthSeverity
    pages: tuple[str, ...]
    description: str

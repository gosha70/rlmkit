# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Domain entities for the LLM Wiki backbone.

Pure stdlib. No filesystem, no LLM, no third-party imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class PageType(str, Enum):
    """Allowed wiki page taxonomy (issue #37 §B)."""

    CONCEPT = "concept"
    WORKFLOW = "workflow"
    INCIDENT = "incident"
    DECISION = "decision"
    PLAYBOOK = "playbook"
    GLOSSARY = "glossary"
    OVERVIEW = "overview"


# Lookup table from PageType -> wiki subdirectory.
# Overview is special — it lives at the wiki root, not in its own folder.
PAGE_TYPE_TO_DIR: dict[PageType, str] = {
    PageType.CONCEPT: "concepts",
    PageType.WORKFLOW: "workflows",
    PageType.INCIDENT: "incidents",
    PageType.DECISION: "decisions",
    PageType.PLAYBOOK: "playbooks",
    PageType.GLOSSARY: "glossary",
    PageType.OVERVIEW: "",  # placed at wiki/overview.md
}


class PageStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    STALE = "stale"


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


# ---------------------------------------------------------------------------
# Wiki page model
# ---------------------------------------------------------------------------


@dataclass
class WikiPage:
    """A single wiki page: machine-readable frontmatter + markdown body.

    The frontmatter fields are required (lint enforces this). The body is the
    rendered markdown content following the YAML block.
    """

    title: str
    slug: str
    type: PageType
    sources: list[str] = field(default_factory=list)
    status: PageStatus = PageStatus.DRAFT
    created: date | None = None
    updated: date | None = None
    body: str = ""

    def relative_path(self) -> str:
        """File path relative to the wiki root, e.g. ``concepts/chunking.md``."""
        sub = PAGE_TYPE_TO_DIR[self.type]
        if not sub:
            return f"{self.slug}.md"
        return f"{sub}/{self.slug}.md"


# ---------------------------------------------------------------------------
# Citations / query results
# ---------------------------------------------------------------------------


@dataclass
class WikiCitation:
    """A pointer from an answer back to a wiki page (and its raw sources)."""

    page_slug: str
    page_type: PageType
    raw_sources: list[str] = field(default_factory=list)
    score: float = 0.0


# ---------------------------------------------------------------------------
# Linting
# ---------------------------------------------------------------------------


@dataclass
class LintIssue:
    path: str
    severity: LintSeverity
    code: str
    message: str


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity is LintSeverity.ERROR]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity is LintSeverity.WARNING]

    @property
    def passed(self) -> bool:
        """A report passes when there are no errors. Warnings do not block."""
        return not self.errors


# ---------------------------------------------------------------------------
# Ingest / promote results
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    source_id: str
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    log_entry: str = ""


@dataclass
class PromoteResult:
    page_slug: str
    page_type: PageType
    created: bool
    log_entry: str = ""

# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""LLM Wiki backbone for RLMKit.

A curated markdown knowledge layer with a typed page schema, a
four-question ingest gate, a wiki-first query path, and RLMKit's
recursive controller wired in as the synthesis engine for large
corpora (issue gosha70/rlmkit#37).

The schema (page types, ingest rules, citation rules, lint rules)
is borrowed verbatim from gosha70/code-copilot-team. See
``DESIGN.md`` for the borrow/divergence ledger.
"""

from .errors import (
    BackendFailure,
    BackendNotFound,
    ContractViolation,
    OutputWriteFailure,
    SourceMissing,
    WikiError,
)
from .ingest import Ingestor
from .linter import LintReport, LintViolation, lint_wiki
from .proposal import IngestProposal, IngestRequest, render_proposal_file
from .query import WikiQueryResult, query_wiki
from .schema import PAGE_TYPE_DIRS, VALID_PAGE_TYPES, parse_frontmatter

__all__ = [
    "BackendFailure",
    "BackendNotFound",
    "ContractViolation",
    "Ingestor",
    "IngestProposal",
    "IngestRequest",
    "LintReport",
    "LintViolation",
    "OutputWriteFailure",
    "PAGE_TYPE_DIRS",
    "SourceMissing",
    "VALID_PAGE_TYPES",
    "WikiError",
    "WikiQueryResult",
    "lint_wiki",
    "parse_frontmatter",
    "query_wiki",
    "render_proposal_file",
]

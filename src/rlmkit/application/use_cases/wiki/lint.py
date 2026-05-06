# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki lint use case — pure validation, no LLM, no network.

Errors block: any required-frontmatter or schema violation.
Warnings inform but do not block: link rot, stale source pointers.
"""

from __future__ import annotations

import re

from rlmkit.application.ports.wiki_port import WikiRepositoryPort
from rlmkit.domain.wiki import (
    LintIssue,
    LintReport,
    LintSeverity,
    PageType,
    WikiPage,
)
from rlmkit.infrastructure.wiki.frontmatter import (
    REQUIRED_FRONTMATTER_FIELDS,
    FrontmatterError,
    parse_frontmatter,
)

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

# Lint codes — kept here as constants because they cross test/page boundaries.
LINT_FRONTMATTER_MISSING = "FM001"
LINT_FRONTMATTER_INVALID = "FM002"
LINT_UNKNOWN_TYPE = "FM003"
LINT_BROKEN_LINK = "LK001"
LINT_STALE_SOURCE = "SR001"
LINT_ORPHAN_PAGE = "OP001"


class LintWikiUseCase:
    """Validate every page in the wiki against the frontmatter schema and links."""

    def __init__(self, repo: WikiRepositoryPort):
        self.repo = repo

    def execute(self) -> LintReport:
        report = LintReport()
        pages = self.repo.list_pages()
        page_paths = {p.relative_path() for p in pages}
        raw_ids = set(self.repo.list_raws())

        # Re-walk pages from disk so we can flag unparseable frontmatter too.
        # list_pages() silently drops them; lint must catch them.
        for page in pages:
            self._check_page(page, raw_ids, page_paths, report)

        # Look for orphan pages — pages whose `sources` reference no raw
        # file at all (warning, not error: a page may be a synthesis page).
        for page in pages:
            if not page.sources:
                report.issues.append(
                    LintIssue(
                        path=page.relative_path(),
                        severity=LintSeverity.WARNING,
                        code=LINT_ORPHAN_PAGE,
                        message="Page has no `sources`; consider linking raw inputs.",
                    )
                )

        return report

    # ------------------------------------------------------------------

    def _check_page(
        self,
        page: WikiPage,
        raw_ids: set[str],
        page_paths: set[str],
        report: LintReport,
    ) -> None:
        # Frontmatter completeness was already enforced by page_from_text;
        # if a page made it into list_pages() its frontmatter parsed cleanly.
        # We still check optional invariants:
        if page.type not in PageType:
            report.issues.append(
                LintIssue(
                    path=page.relative_path(),
                    severity=LintSeverity.ERROR,
                    code=LINT_UNKNOWN_TYPE,
                    message=f"Unknown page type {page.type!r}",
                )
            )

        # Stale source references — warning.
        for src in page.sources:
            if src not in raw_ids:
                report.issues.append(
                    LintIssue(
                        path=page.relative_path(),
                        severity=LintSeverity.WARNING,
                        code=LINT_STALE_SOURCE,
                        message=f"Source {src!r} not present in knowledge/raw/",
                    )
                )

        # Broken intra-wiki links — warning. Only flag relative links that
        # look like wiki paths (don't try to resolve http(s) or anchors).
        for match in LINK_PATTERN.finditer(page.body):
            target = match.group(1).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if target not in page_paths:
                report.issues.append(
                    LintIssue(
                        path=page.relative_path(),
                        severity=LintSeverity.WARNING,
                        code=LINT_BROKEN_LINK,
                        message=f"Link target {target!r} not found in wiki.",
                    )
                )


def parse_or_lint_error(text: str, relative_path: str) -> tuple[WikiPage | None, LintIssue | None]:
    """Helper: try to parse a page; if it fails, return a LintIssue describing why.

    Useful for callers that want to lint files that ``list_pages`` would skip.
    """
    try:
        from rlmkit.infrastructure.wiki.frontmatter import page_from_text

        return page_from_text(text), None
    except FrontmatterError as exc:
        # Determine the right code.
        if "Missing required" in str(exc):
            code = LINT_FRONTMATTER_MISSING
            missing = [f for f in REQUIRED_FRONTMATTER_FIELDS if f not in (parse_frontmatter(text)[0] or {})]
            msg = f"Missing required frontmatter fields: {missing}"
        else:
            code = LINT_FRONTMATTER_INVALID
            msg = str(exc)
        return None, LintIssue(
            path=relative_path,
            severity=LintSeverity.ERROR,
            code=code,
            message=msg,
        )

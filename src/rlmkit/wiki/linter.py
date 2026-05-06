# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Python port of code-copilot-team's ``lint-wiki.sh``.

Runs the same six structural checks the cct linter enforces:

1. Frontmatter present, well-formed, with required keys.
2. ``page_type`` is one of the canonical values.
3. ``slug`` matches the filename stem (with the ``index.md``
   special case) and is unique across the wiki.
4. ``page_type`` matches the directory the page lives in.
5. Intra-wiki ``[text](path.md)`` links resolve to a real file.
6. Every page (except ``index`` / ``log``) is reachable from
   ``index.md`` via markdown links.

Behavior is intended to match cct rule-for-rule. Differences are a
bug.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .schema import (
    PAGE_TYPE_DIRS,
    REQUIRED_FRONTMATTER_KEYS,
    SOURCES_EXEMPT_TYPES,
    VALID_PAGE_TYPES,
    expected_slug_for,
    has_sources,
    parse_frontmatter,
)

LINK_RE = re.compile(r"\]\(([^)]+)\)")


@dataclass(frozen=True)
class LintViolation:
    page: str  # path relative to wiki root
    rule: str
    detail: str


@dataclass
class LintReport:
    pages_linted: int = 0
    violations: list[LintViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        return f"linted {self.pages_linted} pages, {len(self.violations)} violations"


def lint_wiki(wiki_dir: Path) -> LintReport:
    """Lint every wiki page under ``wiki_dir``.

    Excludes the ``schema/`` and ``scripts/`` subdirectories, per
    cct's rule (those are structural docs / tooling, not wiki
    content).
    """
    wiki_dir = wiki_dir.resolve()
    report = LintReport()

    pages = sorted(
        p
        for p in wiki_dir.rglob("*.md")
        if "schema" not in p.relative_to(wiki_dir).parts
        and "scripts" not in p.relative_to(wiki_dir).parts
    )

    slugs_seen: dict[str, str] = {}
    page_records: list[tuple[Path, dict]] = []

    for page in pages:
        report.pages_linted += 1
        rel = str(page.relative_to(wiki_dir))
        text = page.read_text(encoding="utf-8")
        fm, fm_close_line = parse_frontmatter(text)

        if not fm:
            report.violations.append(
                LintViolation(rel, "frontmatter", "missing or unparseable frontmatter")
            )
            page_records.append((page, fm))
            continue

        if fm_close_line > 50:
            report.violations.append(
                LintViolation(rel, "frontmatter", f"closing '---' beyond line 50 ({fm_close_line})")
            )

        for key in REQUIRED_FRONTMATTER_KEYS:
            if not fm.get(key):
                report.violations.append(
                    LintViolation(rel, "frontmatter", f"missing required key '{key}'")
                )

        page_type = fm.get("page_type", "")
        slug = fm.get("slug", "")

        if page_type and page_type not in VALID_PAGE_TYPES:
            report.violations.append(
                LintViolation(rel, "page_type", f"'{page_type}' is not a canonical page type")
            )

        if slug:
            expected = expected_slug_for(page, wiki_dir)
            if slug != expected:
                report.violations.append(
                    LintViolation(rel, "slug", f"slug '{slug}' should be '{expected}'")
                )
            if slug in slugs_seen:
                report.violations.append(
                    LintViolation(
                        rel, "slug", f"duplicate slug '{slug}' (also in {slugs_seen[slug]})"
                    )
                )
            else:
                slugs_seen[slug] = rel

        if page_type in PAGE_TYPE_DIRS:
            expected_dir = PAGE_TYPE_DIRS[page_type]
            actual_parent = str(page.parent.relative_to(wiki_dir))
            if actual_parent == "":
                actual_parent = "."
            if actual_parent != expected_dir:
                report.violations.append(
                    LintViolation(
                        rel,
                        "directory",
                        f"page_type '{page_type}' belongs in '{expected_dir}/' but found in '{actual_parent}/'",
                    )
                )

        if page_type and page_type not in SOURCES_EXEMPT_TYPES and not has_sources(fm):
            report.violations.append(
                LintViolation(rel, "sources", "missing 'sources:' (or empty list)")
            )

        page_records.append((page, fm))

    _check_links(wiki_dir, pages, report)
    _check_orphans(wiki_dir, pages, page_records, report)
    return report


def _intra_wiki_md_links(page: Path, wiki_dir: Path) -> list[Path]:
    """Resolve every intra-wiki ``[text](path.md)`` link in ``page``."""
    text = page.read_text(encoding="utf-8")
    out: list[Path] = []
    for match in LINK_RE.finditer(text):
        target = match.group(1).split()[0]  # drop optional title
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target_path = target.split("#", 1)[0]
        if not target_path or not target_path.endswith(".md"):
            continue
        resolved = (page.parent / target_path).resolve()
        try:
            resolved.relative_to(wiki_dir)
        except ValueError:
            continue  # link escapes the wiki tree
        out.append(resolved)
    return out


def _check_links(wiki_dir: Path, pages: list[Path], report: LintReport) -> None:
    for page in pages:
        rel = str(page.relative_to(wiki_dir))
        for target in _intra_wiki_md_links(page, wiki_dir):
            if not target.exists():
                report.violations.append(
                    LintViolation(
                        rel,
                        "link",
                        f"broken intra-wiki link → {target.relative_to(wiki_dir)}",
                    )
                )


def _check_orphans(
    wiki_dir: Path,
    pages: list[Path],
    page_records: list[tuple[Path, dict]],
    report: LintReport,
) -> None:
    index = wiki_dir / "index.md"
    if not index.exists():
        report.violations.append(LintViolation("index.md", "structure", "missing index.md"))
        return

    reached: set[Path] = {index.resolve()}
    queue: deque[Path] = deque([index.resolve()])
    while queue:
        cur = queue.popleft()
        for target in _intra_wiki_md_links(cur, wiki_dir):
            tgt = target.resolve()
            if tgt in reached or not tgt.exists():
                continue
            reached.add(tgt)
            queue.append(tgt)

    for page, fm in page_records:
        rel = str(page.relative_to(wiki_dir))
        if rel == "log.md":
            continue
        if fm.get("page_type") == "index":
            continue
        if page.resolve() not in reached:
            report.violations.append(
                LintViolation(rel, "orphan", "not reachable from index.md")
            )

# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Minimal structural linter — runs against any wiki tree (live or staged).

Checks:
  - frontmatter parses + required keys present
  - slug is kebab-case and matches filename stem (special case for
    ``<dir>/index.md`` whose slug is the parent directory name)
  - slug is unique across the wiki
  - page lives in the directory matching its page_type
  - intra-wiki markdown links resolve to a real file
  - every page (except index/log/overview) is reachable from index.md

Intentionally NOT a port of cct's full bash linter — this is the
gate the promoter runs against the staged tree and the gate the
``lint`` verb runs against the live tree. The cct linter has
more checks (sources frontmatter shape, etc.); we keep the
critical structural ones here.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .yaml_lite import parse_frontmatter

_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MD_LINK_RE = re.compile(r"\]\(([^)]+\.md)(?:#[^)]*)?\)")

_REQUIRED_KEYS = {"page_type", "slug", "title", "status", "last_reviewed"}

_PAGE_TYPE_DIR = {
    "concept": "concepts",
    "workflow": "workflows",
    "incident": "incidents",
    "decision": "decisions",
    "playbook": "playbooks",
    "glossary": "glossary",
    "open-question": "open-questions",
    "index": "",
    "log": "",
    "overview": "",
}

_ORPHAN_EXEMPT = {"index", "log", "overview"}


@dataclass(frozen=True)
class LintViolation:
    page: str
    rule: str
    message: str


def _list_pages(wiki_dir: Path) -> list[Path]:
    if not wiki_dir.is_dir():
        return []
    excluded = {"schema", "scripts"}
    out: list[Path] = []
    for p in sorted(wiki_dir.rglob("*.md")):
        rel = p.relative_to(wiki_dir).parts
        if rel and rel[0] in excluded:
            continue
        out.append(p)
    return out


def _expected_dir(page_type: str) -> str:
    return _PAGE_TYPE_DIR.get(page_type, "")


def _page_slug_for(rel_path: str) -> str | None:
    """Special-case for ``<dir>/index.md`` whose slug is the dir name."""
    parts = rel_path.split("/")
    if len(parts) == 2 and parts[1] == "index.md":
        return parts[0]
    if rel_path == "index.md":
        return "index"
    if rel_path == "log.md":
        return "log"
    if rel_path == "overview.md":
        return "overview"
    return Path(rel_path).stem


def lint(wiki_dir: Path) -> list[LintViolation]:
    """Run the structural lint pass; return a list of violations."""
    violations: list[LintViolation] = []
    pages = _list_pages(wiki_dir)
    if not pages:
        return [LintViolation("<wiki>", "missing", "wiki directory empty or missing")]

    seen_slugs: dict[str, str] = {}
    page_meta: dict[str, dict[str, Any]] = {}
    for p in pages:
        rel = str(p.relative_to(wiki_dir))
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            violations.append(LintViolation(rel, "io", f"unreadable: {exc}"))
            continue
        try:
            fm = parse_frontmatter(text)
        except Exception as exc:  # noqa: BLE001
            violations.append(
                LintViolation(rel, "frontmatter", f"parse failed: {exc}")
            )
            continue
        if not fm:
            violations.append(LintViolation(rel, "frontmatter", "missing or empty"))
            continue
        page_meta[rel] = fm

        # Required keys.
        missing = _REQUIRED_KEYS - set(fm.keys())
        if missing:
            violations.append(
                LintViolation(rel, "frontmatter",
                              f"missing keys: {sorted(missing)}")
            )

        # Slug shape + uniqueness + filename match.
        slug = fm.get("slug")
        if not isinstance(slug, str) or not _KEBAB_RE.match(slug):
            violations.append(
                LintViolation(rel, "slug", f"not kebab-case: {slug!r}")
            )
        else:
            expected_slug = _page_slug_for(rel)
            if expected_slug and slug != expected_slug:
                violations.append(
                    LintViolation(
                        rel, "slug",
                        f"slug {slug!r} does not match filename "
                        f"({expected_slug!r})",
                    )
                )
            if slug in seen_slugs:
                violations.append(
                    LintViolation(
                        rel, "slug",
                        f"duplicate slug {slug!r} (also in {seen_slugs[slug]})",
                    )
                )
            else:
                seen_slugs[slug] = rel

        # Page type → directory.
        page_type = fm.get("page_type")
        if not isinstance(page_type, str) or page_type not in _PAGE_TYPE_DIR:
            violations.append(
                LintViolation(rel, "page_type", f"invalid: {page_type!r}")
            )
        else:
            expected = _expected_dir(page_type)
            if expected:
                if not rel.startswith(f"{expected}/"):
                    violations.append(
                        LintViolation(
                            rel, "directory",
                            f"page_type {page_type!r} expects {expected}/, "
                            f"got {rel}",
                        )
                    )

    # Link integrity + orphan-from-index.
    edges: dict[str, set[str]] = defaultdict(set)
    for p in pages:
        rel = str(p.relative_to(wiki_dir))
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _MD_LINK_RE.finditer(text):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            resolved = (p.parent / target).resolve()
            try:
                resolved_rel = resolved.relative_to(wiki_dir.resolve()).as_posix()
            except ValueError:
                # escape link — not intra-wiki
                continue
            if not (wiki_dir / resolved_rel).exists():
                violations.append(
                    LintViolation(
                        rel, "broken-link",
                        f"link target does not exist: {target!r}",
                    )
                )
            else:
                edges[rel].add(resolved_rel)

    # Reachability from index.md.
    index_rel = "index.md"
    if (wiki_dir / index_rel).exists():
        reachable: set[str] = {index_rel}
        q: deque[str] = deque([index_rel])
        while q:
            cur = q.popleft()
            for nbr in edges.get(cur, ()):
                if nbr not in reachable:
                    reachable.add(nbr)
                    q.append(nbr)
        for p in pages:
            rel = str(p.relative_to(wiki_dir))
            stem = Path(rel).stem
            if rel == index_rel or stem in _ORPHAN_EXEMPT:
                continue
            if rel not in reachable:
                violations.append(
                    LintViolation(rel, "orphan", "not reachable from index.md")
                )

    return violations


def format_violations(violations: list[LintViolation]) -> str:
    if not violations:
        return "structural: 0 violations"
    lines = [f"structural: {len(violations)} violation(s)"]
    for v in violations:
        lines.append(f"  [{v.rule}] {v.page}: {v.message}")
    return "\n".join(lines)

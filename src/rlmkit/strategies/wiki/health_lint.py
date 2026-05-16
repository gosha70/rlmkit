# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Knowledge-health lint — semantic checks beyond structural lint.

Checks (borrowed from cct/scripts/wiki_ingest/health_lint.py):
  1. ``weak-orphan``        — pages reachable from ``index.md`` via
                              exactly one inbound link
  2. ``stale-claim``        — frontmatter ``sources[].path`` no
                              longer exists in the repo
  3. ``missing-cross-link`` — page slug appears in N≥3 pages but
                              fewer than 2 link to the canonical page
  4. ``contradiction``      — LLM-checked over candidate pairs
                              (skipped when no backend supplied)
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .backends import WikiBackend
from .entities import HealthFinding
from .prompts import compose_health_prompt
from .wiki_state import list_wiki_pages, read_text_or_empty
from .yaml_lite import parse_frontmatter

_MD_LINK_RE = re.compile(r"\]\(([^)]+\.md)(?:#[^)]*)?\)")


def _intra_links(page: Path, wiki_dir: Path) -> list[str]:
    out: list[str] = []
    text = read_text_or_empty(page)
    for m in _MD_LINK_RE.finditer(text):
        t = m.group(1).strip()
        if t.startswith(("http://", "https://", "mailto:", "#")):
            continue
        resolved = (page.parent / t).resolve()
        try:
            rel = resolved.relative_to(wiki_dir.resolve()).as_posix()
        except ValueError:
            continue
        out.append(rel)
    return out


def _check_weak_orphans(wiki_dir: Path) -> list[HealthFinding]:
    pages = list_wiki_pages(wiki_dir)
    inbound: dict[str, set[str]] = defaultdict(set)
    all_paths = {str(p.relative_to(wiki_dir)) for p in pages}
    all_paths |= {"index.md", "log.md", "overview.md"}
    sources = pages + [
        wiki_dir / "index.md",
        wiki_dir / "overview.md",
    ]
    for src in sources:
        if not src.exists():
            continue
        rel_src = str(src.relative_to(wiki_dir))
        for tgt in _intra_links(src, wiki_dir):
            if tgt in all_paths:
                inbound[tgt].add(rel_src)
    findings: list[HealthFinding] = []
    for p in pages:
        rel = str(p.relative_to(wiki_dir))
        if Path(rel).stem in {"index", "log", "overview"}:
            continue
        in_set = inbound.get(rel, set())
        if len(in_set) == 1 and "index.md" in in_set:
            findings.append(
                HealthFinding(
                    kind="weak-orphan",
                    severity="warning",
                    pages=(rel,),
                    description=(
                        f"page reachable via a single inbound link from "
                        f"'index.md'; if that hub changes, this page "
                        f"becomes a structural orphan."
                    ),
                )
            )
    return findings


def _check_stale_claims(
    wiki_dir: Path, repo_root: Path
) -> list[HealthFinding]:
    findings: list[HealthFinding] = []
    for p in list_wiki_pages(wiki_dir):
        text = read_text_or_empty(p)
        fm = parse_frontmatter(text) or {}
        sources = fm.get("sources") or []
        if not isinstance(sources, list):
            continue
        rel = str(p.relative_to(wiki_dir))
        for entry in sources:
            if not isinstance(entry, dict):
                continue
            src_path = entry.get("path")
            if not isinstance(src_path, str) or not src_path:
                continue
            full = (repo_root / src_path).resolve()
            if not full.exists():
                findings.append(
                    HealthFinding(
                        kind="stale-claim",
                        severity="warning",
                        pages=(rel,),
                        description=(
                            f"sources[].path {src_path!r} does not "
                            f"exist relative to repo root."
                        ),
                    )
                )
    return findings


def _check_missing_cross_links(wiki_dir: Path) -> list[HealthFinding]:
    pages = list_wiki_pages(wiki_dir)
    slug_to_path: dict[str, str] = {}
    page_text: dict[str, str] = {}
    for p in pages:
        rel = str(p.relative_to(wiki_dir))
        text = read_text_or_empty(p)
        page_text[rel] = text
        fm = parse_frontmatter(text) or {}
        slug = fm.get("slug")
        if isinstance(slug, str) and slug:
            slug_to_path[slug] = rel
    findings: list[HealthFinding] = []
    for slug, canonical in slug_to_path.items():
        # How many pages mention the slug as a word?
        slug_re = re.compile(rf"\b{re.escape(slug)}\b", re.IGNORECASE)
        mentions: list[str] = []
        for rel, text in page_text.items():
            if rel == canonical:
                continue
            if slug_re.search(text):
                mentions.append(rel)
        if len(mentions) < 3:
            continue
        link_re = re.compile(rf"\]\([^)]*{re.escape(slug)}\.md")
        link_count = sum(
            1 for rel in mentions if link_re.search(page_text[rel])
        )
        if link_count < 2:
            findings.append(
                HealthFinding(
                    kind="missing-cross-link",
                    severity="warning",
                    pages=(canonical,) + tuple(mentions[:5]),
                    description=(
                        f"slug {slug!r} mentioned in {len(mentions)} "
                        f"pages but only {link_count} link to "
                        f"{canonical}."
                    ),
                )
            )
    return findings


def _check_contradictions(
    wiki_dir: Path, backend: WikiBackend
) -> list[HealthFinding]:
    """LLM-driven contradiction check.

    Pairs candidate by shared slug-mention or shared frontmatter
    sources[].path. The LLM returns ``{contradictions: [...]}``.
    """
    pages = list_wiki_pages(wiki_dir)
    page_text: dict[str, str] = {
        str(p.relative_to(wiki_dir)): read_text_or_empty(p) for p in pages
    }
    pairs: list[tuple[str, str, str, str]] = []
    keys = list(page_text.keys())
    for i, a in enumerate(keys):
        fm_a = parse_frontmatter(page_text[a]) or {}
        for b in keys[i + 1:]:
            fm_b = parse_frontmatter(page_text[b]) or {}
            # Cheap pair-candidacy: shared slug-mention either way.
            slug_a = fm_a.get("slug") or ""
            slug_b = fm_b.get("slug") or ""
            shares_slug = (
                isinstance(slug_a, str) and slug_a in page_text[b]
            ) or (
                isinstance(slug_b, str) and slug_b in page_text[a]
            )
            if shares_slug:
                pairs.append((a, page_text[a], b, page_text[b]))
    if not pairs:
        return []
    prompt = compose_health_prompt(pairs[:10])  # cap pairs sent to LLM
    raw = backend.invoke(prompt)
    findings: list[HealthFinding] = []
    for c in raw.get("contradictions", []) or []:
        if not isinstance(c, dict):
            continue
        a = c.get("page_a", "")
        b = c.get("page_b", "")
        desc = c.get("description", "")
        if a and b:
            findings.append(
                HealthFinding(
                    kind="contradiction",
                    severity="warning",
                    pages=(a, b),
                    description=desc or "contradiction reported by backend",
                )
            )
    return findings


def lint_health(
    wiki_dir: Path,
    repo_root: Path | None = None,
    backend: WikiBackend | None = None,
) -> list[HealthFinding]:
    repo_root = repo_root or wiki_dir.parent.parent
    findings: list[HealthFinding] = []
    findings.extend(_check_weak_orphans(wiki_dir))
    findings.extend(_check_stale_claims(wiki_dir, repo_root))
    findings.extend(_check_missing_cross_links(wiki_dir))
    if backend is not None:
        findings.extend(_check_contradictions(wiki_dir, backend))
    return findings


def format_findings(findings: list[HealthFinding]) -> str:
    if not findings:
        return "knowledge-health: 0 findings"
    out = [f"knowledge-health: {len(findings)} finding(s)"]
    for f in findings:
        pages = ", ".join(f.pages)
        out.append(f"  [{f.kind} / {f.severity}] {pages}: {f.description}")
    return "\n".join(out)

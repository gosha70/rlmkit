# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Schema + linter unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlmkit.wiki.linter import lint_wiki
from rlmkit.wiki.schema import (
    PAGE_TYPE_DIRS,
    VALID_PAGE_TYPES,
    has_sources,
    is_kebab_case,
    parse_frontmatter,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI = REPO_ROOT / "knowledge" / "wiki"


def test_seeded_wiki_lints_clean() -> None:
    report = lint_wiki(WIKI)
    assert report.ok, "\n".join(f"{v.page}: [{v.rule}] {v.detail}" for v in report.violations)


def test_parse_frontmatter_extracts_required_keys() -> None:
    text = (WIKI / "overview.md").read_text(encoding="utf-8")
    fm, close = parse_frontmatter(text)
    assert close > 0
    assert fm["page_type"] == "overview"
    assert fm["slug"] == "overview"
    assert has_sources(fm)


def test_parse_frontmatter_rejects_missing_fence() -> None:
    fm, close = parse_frontmatter("# no frontmatter here\n")
    assert fm == {} and close == 0


def test_kebab_case_predicate() -> None:
    assert is_kebab_case("good-slug")
    assert is_kebab_case("a")
    assert not is_kebab_case("BadSlug")
    assert not is_kebab_case("trailing-")
    assert not is_kebab_case("under_score")


def test_page_type_dirs_cover_valid_types() -> None:
    assert set(PAGE_TYPE_DIRS) == VALID_PAGE_TYPES


def test_linter_flags_orphan(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "index.md").write_text(
        "---\npage_type: index\nslug: index\ntitle: Index\n"
        "status: stable\nlast_reviewed: 2026-05-05\n---\n# Index\n",
        encoding="utf-8",
    )
    # Orphan: not linked from index.md.
    (wiki / "concepts" / "lonely.md").write_text(
        "---\npage_type: concept\nslug: lonely\ntitle: Lonely\n"
        "status: draft\nlast_reviewed: 2026-05-05\n"
        "sources:\n  - issue: 1\n---\n# Lonely\n",
        encoding="utf-8",
    )
    report = lint_wiki(wiki)
    assert any(v.rule == "orphan" for v in report.violations)


def test_linter_flags_directory_mismatch(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "index.md").write_text(
        "---\npage_type: index\nslug: index\ntitle: Index\n"
        "status: stable\nlast_reviewed: 2026-05-05\n---\n# Index\n[a](a.md)\n",
        encoding="utf-8",
    )
    # concept page in the wrong directory (root, not concepts/).
    (wiki / "a.md").write_text(
        "---\npage_type: concept\nslug: a\ntitle: A\n"
        "status: draft\nlast_reviewed: 2026-05-05\n"
        "sources:\n  - issue: 1\n---\n# A\n",
        encoding="utf-8",
    )
    report = lint_wiki(wiki)
    assert any(v.rule == "directory" for v in report.violations)


def test_linter_flags_missing_sources(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "index.md").write_text(
        "---\npage_type: index\nslug: index\ntitle: Index\n"
        "status: stable\nlast_reviewed: 2026-05-05\n---\n# Index\n[a](concepts/a.md)\n",
        encoding="utf-8",
    )
    (wiki / "concepts" / "a.md").write_text(
        "---\npage_type: concept\nslug: a\ntitle: A\n"
        "status: draft\nlast_reviewed: 2026-05-05\n---\n# A\n",
        encoding="utf-8",
    )
    report = lint_wiki(wiki)
    assert any(v.rule == "sources" for v in report.violations)

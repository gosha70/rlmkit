# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""On-disk markdown implementation of WikiRepositoryPort.

Layout::

    <root>/
      raw/<source-id>.md
      wiki/
        index.md
        log.md
        overview.md
        concepts/<slug>.md
        workflows/<slug>.md
        ...
"""

from __future__ import annotations

from pathlib import Path

from rlmkit.domain.wiki import PAGE_TYPE_TO_DIR, PageType, WikiPage

from .frontmatter import page_from_text, serialize_page

WIKI_SUBDIR = "wiki"
RAW_SUBDIR = "raw"
INDEX_FILE = "index.md"
LOG_FILE = "log.md"


class MarkdownWikiRepository:
    """Concrete WikiRepositoryPort backed by a directory tree.

    The directory is created on first write; reads of missing files raise
    ``FileNotFoundError``.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.wiki_dir = self.root / WIKI_SUBDIR
        self.raw_dir = self.root / RAW_SUBDIR

    # -- raw sources ---------------------------------------------------

    def write_raw(self, source_id: str, content: str) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{source_id}.md"
        path.write_text(content, encoding="utf-8")

    def read_raw(self, source_id: str) -> str:
        return (self.raw_dir / f"{source_id}.md").read_text(encoding="utf-8")

    def list_raws(self) -> list[str]:
        if not self.raw_dir.exists():
            return []
        return sorted(p.stem for p in self.raw_dir.glob("*.md"))

    # -- wiki pages ----------------------------------------------------

    def write_page(self, page: WikiPage) -> bool:
        sub = PAGE_TYPE_TO_DIR[page.type]
        target_dir = self.wiki_dir if not sub else self.wiki_dir / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{page.slug}.md"
        is_new = not path.exists()
        path.write_text(serialize_page(page), encoding="utf-8")
        return is_new

    def read_page(self, relative_path: str) -> WikiPage:
        path = self.wiki_dir / relative_path
        return page_from_text(path.read_text(encoding="utf-8"))

    def list_pages(self) -> list[WikiPage]:
        if not self.wiki_dir.exists():
            return []
        pages: list[WikiPage] = []
        for path in sorted(self.wiki_dir.rglob("*.md")):
            name = path.name
            if name in {INDEX_FILE, LOG_FILE}:
                continue
            try:
                pages.append(page_from_text(path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 — list_pages skips unparseable files
                # Linting will surface the error; listing must not crash.
                continue
        return pages

    def page_exists(self, relative_path: str) -> bool:
        return (self.wiki_dir / relative_path).exists()

    # -- index / log ---------------------------------------------------

    def write_index(self, content: str) -> None:
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        (self.wiki_dir / INDEX_FILE).write_text(content, encoding="utf-8")

    def read_index(self) -> str:
        return (self.wiki_dir / INDEX_FILE).read_text(encoding="utf-8")

    def append_log(self, entry: str) -> None:
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        path = self.wiki_dir / LOG_FILE
        line = entry.rstrip("\n") + "\n"
        if path.exists():
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        else:
            path.write_text(line, encoding="utf-8")

    def read_log(self) -> str:
        path = self.wiki_dir / LOG_FILE
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # -- helpers (not part of the port) --------------------------------

    def page_path(self, page_type: PageType, slug: str) -> Path:
        sub = PAGE_TYPE_TO_DIR[page_type]
        return self.wiki_dir / (f"{slug}.md" if not sub else f"{sub}/{slug}.md")

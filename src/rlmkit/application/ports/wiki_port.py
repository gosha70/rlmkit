# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Repository port for the wiki backbone.

Application use cases speak to the wiki only through this Protocol —
the on-disk markdown layout is an infrastructure concern.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rlmkit.domain.wiki import WikiPage


@runtime_checkable
class WikiRepositoryPort(Protocol):
    """Persistence contract for the wiki layer."""

    # -- raw sources (knowledge/raw/) ----------------------------------

    def write_raw(self, source_id: str, content: str) -> None: ...

    def read_raw(self, source_id: str) -> str: ...

    def list_raws(self) -> list[str]: ...

    # -- wiki pages (knowledge/wiki/<type>/<slug>.md) ------------------

    def write_page(self, page: WikiPage) -> bool:
        """Write the page to disk. Returns True if the page is new,
        False if it overwrote an existing one."""

    def read_page(self, relative_path: str) -> WikiPage: ...

    def list_pages(self) -> list[WikiPage]: ...

    def page_exists(self, relative_path: str) -> bool: ...

    # -- index / log (knowledge/wiki/{index,log}.md) -------------------

    def write_index(self, content: str) -> None: ...

    def read_index(self) -> str: ...

    def append_log(self, entry: str) -> None: ...

    def read_log(self) -> str: ...

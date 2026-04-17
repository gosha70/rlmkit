"""Docs endpoint — allowlisted markdown loader for the Learn tab.

``docs/`` is authoritative. Learn does not fork content; it reads
repo markdown at runtime via this allowlisted endpoint. An explicit
slug → filename mapping (``_DOCS_ALLOWLIST``) prevents arbitrary
file reads, and a resolved-path containment check provides
defense-in-depth against path traversal via symlinks.

See ``doc_internal/specs/learn-tab/pitch-learn-tab.md`` §Backend.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from rlmkit.server.models import DocResponse

router = APIRouter()


# Project root: resolve up from …/src/rlmkit/server/routes/docs.py to
# the repo root four parents above. parents[0]=routes, parents[1]=server,
# parents[2]=rlmkit, parents[3]=src, parents[4]=<repo root>.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
_DOCS_DIR: Path = _PROJECT_ROOT / "docs"

# Slug must match this regex before any lookup. Defense in depth against
# an accidentally loose allowlist; keeps slugs URL-clean.
_SLUG_PATTERN = re.compile(r"^[a-z0-9-]+$")

# Explicit slug → filename allowlist. Files live under _DOCS_DIR.
# Later steps (Cookbook) may extend this with hosts/<provider>.md
# entries; when they do, _SLUG_PATTERN should be updated alongside.
_DOCS_ALLOWLIST: dict[str, str] = {
    "rlm-concepts": "rlm-concepts.md",
    "rlm-prompt-tuning": "rlm-prompt-tuning.md",
    "rlm-studio-guide": "rlm-studio-guide.md",
    "rlmkit-design-document": "RLMKit_Design_Document.md",
    "lessons-from-ai-copilots": "lessons-from-ai-copilots.md",
    # Cookbook provider guides. Slug uses flat "hosts-<name>" form so
    # the slug regex ^[a-z0-9-]+$ stays simple; the filename resolves
    # into the nested docs/hosts/ directory.
    "hosts-ollama": "hosts/ollama.md",
    "hosts-lmstudio": "hosts/lmstudio.md",
    "hosts-vllm": "hosts/vllm.md",
    "hosts-dgx-spark": "hosts/dgx-spark.md",
    "hosts-openai": "hosts/openai.md",
    "hosts-anthropic": "hosts/anthropic.md",
    # hosts-groq intentionally omitted: the app's provider catalog does
    # not currently support a `groq` backend. When Groq support lands,
    # re-add this entry and the matching frontend catalog entry.
}


@router.get("/api/docs/{slug}")
async def get_doc(slug: str) -> DocResponse:
    """Return the markdown body of an allowlisted doc.

    Rejects anything not in the allowlist with 404 so the endpoint
    never leaks whether an unlisted file exists on disk. Also rejects
    resolved paths that escape ``docs/`` (symlink / traversal guard).
    """
    if not _SLUG_PATTERN.fullmatch(slug):
        raise HTTPException(status_code=404, detail="Doc not found")

    filename = _DOCS_ALLOWLIST.get(slug)
    if filename is None:
        raise HTTPException(status_code=404, detail="Doc not found")

    target = (_DOCS_DIR / filename).resolve()
    docs_root = _DOCS_DIR.resolve()
    # Containment check: target must be a descendant of docs_root. Blocks
    # resolution via symlinks that point outside the docs/ tree.
    try:
        target.relative_to(docs_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Doc not found") from exc

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Doc not found")

    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to read doc") from exc

    return DocResponse(slug=slug, content=content)

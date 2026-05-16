# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Atomic patch-set application — the ONLY writer to the live wiki.

Algorithm (borrowed from cct):
  1. Copy the live wiki to a temp staging tree.
  2. Apply each PageEdit to the staging tree.
  3. Run the structural linter against the staging tree.
  4. On success: move staged files into the live wiki and archive
     the proposal dir under ``.applied/``.
  5. On failure: throw away the staging tree; the live wiki is
     untouched.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .entities import PageEdit, WikiPatchSet
from .errors import PromoteApplyError, PromoteValidationError
from .ingestor import load_patch_from_dir
from .structural_lint import lint, format_violations

_INDEX_INSERT_HEADING_RE = re.compile(r"^##\s+", re.MULTILINE)


@dataclass(frozen=True)
class PromoteResult:
    applied_paths: tuple[str, ...]
    proposal_dir: Path
    archived_dir: Path | None
    dry_run: bool


def _apply_edit(staging_dir: Path, edit: PageEdit) -> None:
    target = staging_dir / edit.path
    target.parent.mkdir(parents=True, exist_ok=True)
    if edit.action == "create":
        if target.exists():
            raise PromoteValidationError(
                f"create target already exists: {edit.path}"
            )
        target.write_text(edit.new_content, encoding="utf-8")
    elif edit.action == "update":
        target.write_text(edit.new_content, encoding="utf-8")
    elif edit.action == "append-log":
        log_path = staging_dir / "log.md"
        existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        line = edit.new_content.rstrip()
        if not existing:
            new_log = f"# Wiki log\n\n{line}\n"
        else:
            new_log = (
                existing if existing.endswith("\n") else existing + "\n"
            ) + line + "\n"
        log_path.write_text(new_log, encoding="utf-8")
    elif edit.action == "append-index":
        index_path = staging_dir / "index.md"
        if not index_path.exists():
            index_path.write_text(
                "# Wiki index\n\n## Pages\n\n"
                + edit.new_content.rstrip() + "\n",
                encoding="utf-8",
            )
            return
        existing = index_path.read_text(encoding="utf-8")
        line = edit.new_content.rstrip()
        # Append to the last ## section, or fall back to end of file.
        matches = list(_INDEX_INSERT_HEADING_RE.finditer(existing))
        if matches:
            # Insert just before the next heading after the last match,
            # or at end of file if last match is the last heading.
            last = matches[-1]
            after = existing[last.end():]
            if not after.endswith("\n"):
                after = after + "\n"
            new_index = existing[:last.end()] + after.rstrip("\n") + "\n" + line + "\n"
        else:
            new_index = (
                existing if existing.endswith("\n") else existing + "\n"
            ) + line + "\n"
        index_path.write_text(new_index, encoding="utf-8")


def _materialise_staging(wiki_dir: Path, staging_dir: Path) -> None:
    """Copy the live wiki to the staging dir."""
    if wiki_dir.is_dir():
        shutil.copytree(wiki_dir, staging_dir, dirs_exist_ok=True)
    else:
        staging_dir.mkdir(parents=True, exist_ok=True)


def promote(
    proposal_dir: Path,
    wiki_dir: Path,
    archive_root: Path | None = None,
    dry_run: bool = False,
) -> PromoteResult:
    if not proposal_dir.exists():
        raise PromoteValidationError(f"proposal dir missing: {proposal_dir}")
    patch = load_patch_from_dir(proposal_dir)
    return promote_patch(
        patch,
        wiki_dir,
        proposal_dir=proposal_dir,
        archive_root=archive_root,
        dry_run=dry_run,
    )


def promote_patch(
    patch: WikiPatchSet,
    wiki_dir: Path,
    proposal_dir: Path | None = None,
    archive_root: Path | None = None,
    dry_run: bool = False,
) -> PromoteResult:
    with tempfile.TemporaryDirectory(prefix="rlmkit-wiki-stage-") as tmp:
        staging = Path(tmp) / "wiki"
        _materialise_staging(wiki_dir, staging)
        for edit in patch.edits:
            try:
                _apply_edit(staging, edit)
            except PromoteValidationError:
                raise
            except OSError as exc:
                raise PromoteApplyError(
                    f"failed to apply edit {edit.path!r}: {exc}"
                ) from exc

        violations = lint(staging)
        if violations:
            raise PromoteValidationError(
                "staged tree failed structural lint:\n"
                + format_violations(violations)
            )

        if dry_run:
            return PromoteResult(
                applied_paths=tuple(e.path for e in patch.edits),
                proposal_dir=proposal_dir or Path("<patch>"),
                archived_dir=None,
                dry_run=True,
            )

        # Commit: copy each touched file from staging into the live wiki.
        applied: list[str] = []
        wiki_dir.mkdir(parents=True, exist_ok=True)
        touched: set[str] = set()
        for edit in patch.edits:
            if edit.action in ("create", "update"):
                touched.add(edit.path)
            elif edit.action == "append-log":
                touched.add("log.md")
            elif edit.action == "append-index":
                touched.add("index.md")
        for rel in sorted(touched):
            src = staging / rel
            dst = wiki_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            applied.append(rel)

        archived: Path | None = None
        if proposal_dir is not None and archive_root is not None:
            archive_root.mkdir(parents=True, exist_ok=True)
            archived = archive_root / proposal_dir.name
            if archived.exists():
                shutil.rmtree(archived)
            shutil.move(str(proposal_dir), str(archived))

        return PromoteResult(
            applied_paths=tuple(applied),
            proposal_dir=proposal_dir or Path("<patch>"),
            archived_dir=archived,
            dry_run=False,
        )

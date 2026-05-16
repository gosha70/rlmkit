# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Multi-page ingest orchestrator.

  source_path → load WikiState → compose prompt → invoke backend
  → parse WikiPatchSet → write proposal dir

The proposal directory layout matches cct so curators have one
mental model:

  doc_internal/proposals/<date>-<source-stem>/
    plan.json              # the patch-set, edits with preview pointers
    preview/<rel-path>     # one file per create/update edit
    log.txt                # one log-append line
    index.txt              # one index-append line
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, replace
from pathlib import Path

from .backends import WikiBackend
from .entities import PageEdit, WikiPatchSet
from .errors import (
    ContractViolationError,
    OutputDirError,
    SourceMissingError,
)
from .prompts import compose_ingest_prompt, load_schema_files
from .wiki_state import load_wiki_state


def _slugify_stem(path: Path) -> str:
    s = path.stem.lower()
    out = []
    prev_dash = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-") or "source"


def _parse_patch_set(raw: dict, source_path: Path) -> WikiPatchSet:
    if raw.get("version") != 1:
        raise ContractViolationError(
            f"WikiPatchSet response version must be 1, got {raw.get('version')!r}"
        )
    edits_raw = raw.get("edits")
    if not isinstance(edits_raw, list):
        raise ContractViolationError(
            "WikiPatchSet.edits must be a list"
        )
    edits: list[PageEdit] = []
    for i, e in enumerate(edits_raw):
        if not isinstance(e, dict):
            raise ContractViolationError(
                f"WikiPatchSet.edits[{i}] must be an object"
            )
        for key in ("path", "action", "new_content"):
            if key not in e:
                raise ContractViolationError(
                    f"WikiPatchSet.edits[{i}] missing key {key!r}"
                )
        if e["action"] not in (
            "create", "update", "append-log", "append-index"
        ):
            raise ContractViolationError(
                f"WikiPatchSet.edits[{i}].action invalid: {e['action']!r}"
            )
        edits.append(
            PageEdit(
                path=e["path"],
                action=e["action"],
                new_content=e["new_content"],
                rationale=e.get("rationale", ""),
            )
        )
    return WikiPatchSet.of(
        edits=edits,
        source_path=str(source_path),
        rationale=str(raw.get("rationale", "")),
    )


def _validate_set(patch: WikiPatchSet) -> None:
    """Set-level invariants (no duplicate creates, etc.)."""
    seen_creates: set[str] = set()
    for e in patch.edits:
        if e.action == "create":
            if e.path in seen_creates:
                raise ContractViolationError(
                    f"duplicate create for path {e.path!r}"
                )
            seen_creates.add(e.path)


def write_proposal_dir(
    patch: WikiPatchSet,
    source_path: Path,
    proposals_root: Path,
) -> Path:
    """Materialise the patch-set on disk; return the proposal dir."""
    date = datetime.date.today().isoformat()
    slug = _slugify_stem(source_path)
    out_dir = proposals_root / f"{date}-{slug}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preview").mkdir(exist_ok=True)
    except OSError as exc:
        raise OutputDirError(f"could not create proposal dir: {exc}") from exc

    plan_edits = []
    for e in patch.edits:
        if e.action in ("create", "update"):
            preview_rel = f"preview/{e.path}"
            preview_path = out_dir / preview_rel
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview_path.write_text(e.new_content, encoding="utf-8")
            plan_edits.append(
                {
                    "path": e.path,
                    "action": e.action,
                    "preview": preview_rel,
                    "rationale": e.rationale,
                }
            )
        else:
            plan_edits.append(
                {
                    "path": e.path,
                    "action": e.action,
                    "new_content": e.new_content,
                    "rationale": e.rationale,
                }
            )

    plan = {
        "version": 1,
        "source_path": patch.source_path,
        "rationale": patch.rationale,
        "edits": plan_edits,
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir


def ingest(
    source_path: Path,
    wiki_dir: Path,
    proposals_root: Path,
    backend: WikiBackend,
) -> tuple[WikiPatchSet, Path]:
    """End-to-end ingest: returns (patch_set, proposal_dir)."""
    if not source_path.exists():
        raise SourceMissingError(f"source not found: {source_path}")
    source_content = source_path.read_text(encoding="utf-8")
    wiki_state = load_wiki_state(wiki_dir, source_path, source_content)
    schema = load_schema_files()
    prompt = compose_ingest_prompt(source_path, source_content, wiki_state, schema)
    raw = backend.invoke(prompt)
    patch = _parse_patch_set(raw, source_path)
    _validate_set(patch)
    proposal_dir = write_proposal_dir(patch, source_path, proposals_root)
    return patch, proposal_dir


def load_patch_from_dir(proposal_dir: Path) -> WikiPatchSet:
    """Reconstruct a WikiPatchSet from a proposals dir on disk."""
    plan_path = proposal_dir / "plan.json"
    if not plan_path.exists():
        raise ContractViolationError(
            f"plan.json missing in {proposal_dir}"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    edits: list[PageEdit] = []
    for raw in plan.get("edits", []):
        if raw["action"] in ("create", "update"):
            preview_rel = raw["preview"]
            preview_path = proposal_dir / preview_rel
            if not preview_path.exists():
                raise ContractViolationError(
                    f"preview file missing: {preview_path}"
                )
            content = preview_path.read_text(encoding="utf-8")
        else:
            content = raw["new_content"]
        edits.append(
            PageEdit(
                path=raw["path"],
                action=raw["action"],
                new_content=content,
                rationale=raw.get("rationale", ""),
            )
        )
    return WikiPatchSet.of(
        edits=edits,
        source_path=plan.get("source_path", ""),
        rationale=plan.get("rationale", ""),
    )

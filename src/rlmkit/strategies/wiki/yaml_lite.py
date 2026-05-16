# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Hand-rolled frontmatter parser — stdlib only.

Borrowed-shape from code-copilot-team/scripts/wiki_ingest/yaml_lite.py.
We do not pull in PyYAML to keep the wiki layer dep-free.

Supports:
  - top-level scalar keys: ``key: value``
  - lists of dicts under a key:
        sources:
          - path: foo.md
            sha: abc123
          - url: https://x
            retrieved: 2026-05-07

Anything more exotic (anchors, multi-line scalars, flow style)
is not supported on purpose — wiki frontmatter is a constrained
dialect.
"""

from __future__ import annotations

from typing import Any


def parse_frontmatter(markdown: str) -> dict[str, Any] | None:
    """Extract and parse the YAML frontmatter from a markdown string.

    Returns the parsed dict, or ``None`` if no frontmatter is present.
    """
    if not markdown.startswith("---\n") and not markdown.startswith("---\r\n"):
        return None
    # Find closing fence.
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    closing = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing = i
            break
    if closing == -1:
        return None
    return _parse_block(lines[1:closing])


def _unquote(s: str) -> str:
    s = s.strip()
    if (
        len(s) >= 2
        and s[0] == s[-1]
        and s[0] in ("'", '"')
    ):
        return s[1:-1]
    return s


def _parse_scalar(s: str) -> Any:
    s = _unquote(s)
    if s == "":
        return ""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "null":
        return None
    try:
        if "." not in s:
            return int(s)
    except ValueError:
        pass
    return s


def _parse_block(lines: list[str]) -> dict[str, Any]:
    """Parse the YAML body (between the --- fences)."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        # Top-level key.
        if raw.startswith(" "):
            i += 1
            continue
        if ":" not in raw:
            i += 1
            continue
        key, _, val = raw.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            # Could be a list of dicts on subsequent lines.
            sub_items: list[Any] = []
            j = i + 1
            while j < len(lines):
                sub = lines[j]
                if not sub.strip():
                    j += 1
                    continue
                if not sub.startswith(" "):
                    break
                stripped = sub.lstrip()
                indent = len(sub) - len(stripped)
                if stripped.startswith("- "):
                    item: dict[str, Any] = {}
                    rest = stripped[2:]
                    if ":" in rest:
                        k, _, v = rest.partition(":")
                        item[k.strip()] = _parse_scalar(v)
                    j += 1
                    while j < len(lines):
                        cont = lines[j]
                        if not cont.strip():
                            j += 1
                            continue
                        cont_stripped = cont.lstrip()
                        cont_indent = len(cont) - len(cont_stripped)
                        if cont_indent <= indent or cont_stripped.startswith("- "):
                            break
                        if ":" in cont_stripped:
                            k, _, v = cont_stripped.partition(":")
                            item[k.strip()] = _parse_scalar(v)
                        j += 1
                    sub_items.append(item)
                else:
                    break
            if sub_items:
                out[key] = sub_items
            else:
                out[key] = ""
            i = j
        else:
            out[key] = _parse_scalar(val)
            i += 1
    return out

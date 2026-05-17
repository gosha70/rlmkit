"""Regression guards tied to first-party assumptions about transitive deps.

Each test here documents a security invariant — typically of the form
"no first-party code calls X" — that, if broken, would make the project
reachable to a vulnerability in a transitive dependency. The guards stand
on their own as defense-in-depth, independent of whether the matching CVE
is currently fixed at the lock level or covered by a pip-audit ignore.
If a guard fails, evaluate whether the new code path is safe against the
underlying class of vulnerability before removing or relaxing it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = (REPO_ROOT / "src", REPO_ROOT / "tests")
SELF_PATH = Path(__file__).resolve()

# Built piecewise so this file does not match its own scan pattern.
_DOTENV_SET_KEY_RE = re.compile(r"\b" + "set" + r"_key\s*\(")


def test_no_dotenv_set_key_calls() -> None:
    """python-dotenv CVE-2026-28684 (set_key symlink-follow).

    As of litellm 1.83.14 the lock resolves python-dotenv 1.2.2 (the
    patched version), so the pip-audit ignore for CVE-2026-28684 was
    removed in af85660. This guard is retained as defense-in-depth:
    nothing prevents a future litellm release from re-pinning an
    older python-dotenv, and as long as no first-party code calls
    ``set_key()`` the symlink-follow path is unreachable regardless
    of the transitive's version. This test fails if anyone introduces
    a set_key() call, forcing re-evaluation of the assumption.
    """
    offenders: list[str] = []
    for root in SCAN_DIRS:
        for path in root.rglob("*.py"):
            if path.resolve() == SELF_PATH:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if _DOTENV_SET_KEY_RE.search(line):
                    offenders.append(f"  {path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")
    assert not offenders, (
        "Found dotenv.set_key( call(s) in first-party code.\n"
        "RLMKit's defense-in-depth invariant for python-dotenv "
        "CVE-2026-28684 is that no first-party code calls set_key(), "
        "so the symlink-follow path stays unreachable even if a "
        "future litellm release regresses its python-dotenv pin. "
        "Either remove the call, or remove this guard and accept "
        "reliance on the lock pinning python-dotenv >= 1.2.2.\n" + "\n".join(offenders)
    )

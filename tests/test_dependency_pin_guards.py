"""Regression guards tied to dependency-pin / pip-audit-ignore decisions.

Each test here documents a security invariant whose validity depends on
the current dependency graph. If the invariant is broken, the matching
``--ignore-vuln`` flag in ``.github/workflows/ci.yml`` must be revisited
before merging.
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

    litellm 1.83.7 hard-pins python-dotenv==1.0.1, which is below the
    1.2.2 fix. We ignore the CVE in pip-audit on the basis that no
    first-party code calls ``set_key()``. This test fails if anyone
    introduces such a call, forcing the ignore to be re-evaluated.
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
        "python-dotenv CVE-2026-28684 is currently ignored in "
        ".github/workflows/ci.yml on the basis that we never call "
        "set_key(). Either remove the call, or remove the "
        "--ignore-vuln CVE-2026-28684 flag and address the CVE.\n" + "\n".join(offenders)
    )

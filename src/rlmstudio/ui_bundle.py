"""Locate the bundled RLM Studio UI shipped inside the rlmstudio wheel.

When `rlm-studio` is installed via PyPI (or via ``pip install -e .`` after
``npm run build:bundle``), the Next.js static export lives under
``src/rlmstudio/_ui/``. The server mounts it at ``/studio`` and the CLI
``rlm-studio studio`` command opens a browser to it.

When `rlm-studio` is installed from a source checkout that has *not* run the
frontend build, ``_ui/`` is either missing or empty. The server runs
normally; the CLI prints a helpful pointer at the dev workflow.

Returning ``None`` from `get_ui_directory()` is the canonical signal that
"no bundled UI is available." Every consumer must handle that case.
"""

from __future__ import annotations

from pathlib import Path


def get_ui_directory() -> Path | None:
    """Return the path to the bundled Studio UI, or None if not bundled.

    The check requires both the directory to exist and an ``index.html``
    file to be present. A directory that exists but is empty (which can
    happen with stale source checkouts) is treated the same as missing.
    """
    import rlmstudio

    pkg_root = Path(rlmstudio.__file__).parent
    ui_dir = pkg_root / "_ui"
    if ui_dir.is_dir() and (ui_dir / "index.html").is_file():
        return ui_dir
    return None


def is_ui_bundled() -> bool:
    """Cheap boolean wrapper around `get_ui_directory()`."""
    return get_ui_directory() is not None

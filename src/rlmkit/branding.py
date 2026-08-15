"""Single source of truth for product identity.

Every module that needs the product name, the environment-variable prefix,
or the on-disk state directory imports it from here instead of spelling the
string out.  Renaming the product is then a change to the constants below
plus a legacy entry for the old value — never a repo-wide search-and-replace
in runtime code.

Stdlib only: this module is imported from every layer, including
``application/``, so it must not pull in anything heavier.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Identity ---------------------------------------------------------------

PRODUCT_NAME = "RLMKit"
"""Human-facing product name (UI, logs, docs)."""

DIST_NAME = "rlmkit"
"""Distribution name on PyPI (``pip install <DIST_NAME>``)."""

PACKAGE_NAME = "rlmkit"
"""Top-level import package name."""

CLI_NAME = "rlmkit"
"""Console-script name registered in ``[project.scripts]``."""

# --- Environment variables --------------------------------------------------

ENV_PREFIX = "RLMKIT_"
"""Canonical prefix for every environment variable the product reads."""

LEGACY_ENV_PREFIXES: tuple[str, ...] = ()
"""Older prefixes still honoured (with a one-time deprecation warning)."""

# --- On-disk state ----------------------------------------------------------

STATE_DIR_NAME = ".rlmkit"
"""Directory under ``$HOME`` that holds config, secrets and telemetry."""

LEGACY_STATE_DIR_NAMES: tuple[str, ...] = ()
"""Older state-directory names; contents are copied forward on first boot."""

STATE_DIR_ENV_SUFFIX = "DIR"
"""``<ENV_PREFIX>DIR`` overrides the state directory location."""

CONFIG_FILE_STEM = "rlmkit_config"
"""Base name of the CWD-local config file (``./<stem>.yaml`` / ``.json``)."""


# --- Environment access -----------------------------------------------------

_warned_legacy_env: set[str] = set()


def env_name(suffix: str) -> str:
    """Return the canonical environment-variable name for ``suffix``.

    ``env_name("PORT")`` → ``"RLMKIT_PORT"`` (whatever the current prefix is).
    """
    return f"{ENV_PREFIX}{suffix}"


def env(suffix: str, default: str | None = None) -> str | None:
    """Read ``<ENV_PREFIX><suffix>``, falling back to legacy prefixes.

    The canonical name always wins.  When only a legacy name is set, its
    value is returned and a deprecation warning is logged once per suffix.
    """
    canonical = env_name(suffix)
    value = os.environ.get(canonical)
    if value is not None:
        return value
    for prefix in LEGACY_ENV_PREFIXES:
        legacy = f"{prefix}{suffix}"
        value = os.environ.get(legacy)
        if value is not None:
            if suffix not in _warned_legacy_env:
                _warned_legacy_env.add(suffix)
                logger.warning(
                    "%s is deprecated; set %s instead (legacy name honoured for this release)",
                    legacy,
                    canonical,
                )
            return value
    return default


# --- State directory --------------------------------------------------------

_migration_attempted: set[Path] = set()


def state_dir(*, create: bool = False) -> Path:
    """Return the product's state directory.

    Resolution order:

    1. ``<ENV_PREFIX>DIR`` (or a legacy-prefixed equivalent) if set.
    2. ``$HOME/<STATE_DIR_NAME>``.

    When the canonical directory does not exist yet but a legacy-named one
    does, the legacy contents are **copied** (never moved or deleted) into the
    canonical location the first time this function is called in a process.
    Pass ``create=True`` to ``mkdir -p`` the resolved directory.
    """
    override = env(STATE_DIR_ENV_SUFFIX)
    if override:
        target = Path(override).expanduser()
    else:
        target = Path.home() / STATE_DIR_NAME
        _migrate_legacy_state(target)
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def _migrate_legacy_state(target: Path) -> None:
    """Copy a legacy state directory forward into ``target`` once per process."""
    if target in _migration_attempted:
        return
    _migration_attempted.add(target)
    if target.exists():
        return
    for legacy_name in LEGACY_STATE_DIR_NAMES:
        legacy_dir = target.parent / legacy_name
        if not legacy_dir.is_dir():
            continue
        try:
            shutil.copytree(legacy_dir, target, dirs_exist_ok=True)
        except OSError as exc:  # pragma: no cover - defensive; surfaced in logs
            logger.warning("Could not copy legacy state from %s to %s: %s", legacy_dir, target, exc)
            return
        logger.info(
            "Copied %s state from %s to %s (the old directory was left untouched)",
            PRODUCT_NAME,
            legacy_dir,
            target,
        )
        return

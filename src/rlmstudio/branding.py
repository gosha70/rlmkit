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

PRODUCT_NAME = "RLM Studio"
"""Human-facing product name (UI, logs, docs)."""

DIST_NAME = "rlm-studio"
"""Distribution name on PyPI (``pip install <DIST_NAME>``)."""

PACKAGE_NAME = "rlmstudio"
"""Top-level import package name."""

CLI_NAME = "rlm-studio"
"""Console-script name registered in ``[project.scripts]``."""

# --- Environment variables --------------------------------------------------

ENV_PREFIX = "RLM_STUDIO_"
"""Canonical prefix for every environment variable the product reads."""

LEGACY_ENV_PREFIXES: tuple[str, ...] = ("RLMKIT_",)
"""Older prefixes still honoured (with a one-time deprecation warning)."""

# --- On-disk state ----------------------------------------------------------

STATE_DIR_NAME = ".rlm-studio"
"""Directory under ``$HOME`` that holds config, secrets and telemetry."""

LEGACY_STATE_DIR_NAMES: tuple[str, ...] = (".rlmkit",)
"""Older state-directory names; contents are copied forward on first boot."""

DEFAULT_HOST = "127.0.0.1"
"""Default bind address for the API server (override: ``<ENV_PREFIX>HOST``)."""

DEFAULT_PORT = 8000
"""Default bind port for the API server (override: ``<ENV_PREFIX>PORT``)."""

STATE_DIR_ENV_SUFFIX = "DIR"
"""``<ENV_PREFIX>DIR`` overrides the state directory location."""

CONFIG_FILE_STEM = "rlm_studio_config"
"""Base name of the CWD-local config file (``./<stem>.yaml`` / ``.json``)."""

LEGACY_CONFIG_FILE_STEMS: tuple[str, ...] = ("rlmkit_config",)
"""Older CWD-local config stems, searched after the canonical one."""

KEYRING_SERVICE = "rlm-studio"
"""Service name under which provider API keys are stored in the OS keyring."""

LEGACY_KEYRING_SERVICES: tuple[str, ...] = ("rlmkit",)
"""Older keyring service names; read as a fallback and re-saved under the new one."""

SANDBOX_IMAGE_NAME = "rlm-studio-sandbox"
"""Default Docker image tag for the sandbox (built from docker/Dockerfile.sandbox)."""


# --- Environment access -----------------------------------------------------

_warned_legacy_env: set[str] = set()


def env_name(suffix: str) -> str:
    """Return the canonical environment-variable name for ``suffix``.

    ``env_name("PORT")`` → ``"RLM_STUDIO_PORT"`` (whatever the current prefix is).
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


def state_dir() -> Path:
    """Return the product's state directory. **Pure**: no filesystem access.

    Resolution order:

    1. ``<ENV_PREFIX>DIR`` (or a legacy-prefixed equivalent) if set.
    2. ``$HOME/<STATE_DIR_NAME>``.

    Nothing is created or copied here — callers ``mkdir`` when they write,
    and the one-time legacy migration is a separate, explicit step
    (:func:`migrate_legacy_state`) that only real application boot paths
    call.  Importing a module, running tests, ``rlm-studio version`` or
    using the library passively must never mutate the filesystem.
    """
    override = env(STATE_DIR_ENV_SUFFIX)
    if override:
        return Path(override).expanduser()
    return Path.home() / STATE_DIR_NAME


def migrate_legacy_state() -> Path | None:
    """Copy a legacy state directory forward into :func:`state_dir` once.

    Intended to be called from application boot paths only (server
    lifespan startup — which covers ``rlm-studio studio``,
    ``python -m rlmstudio.server``, uvicorn and Docker).  It is a no-op
    when the canonical directory already exists, when the location is
    overridden via ``<ENV_PREFIX>DIR``, or when no legacy directory is
    present.  Legacy contents are **copied**, never moved or deleted.

    Returns:
        The legacy directory that was copied, or ``None`` if nothing was
        migrated.
    """
    if env(STATE_DIR_ENV_SUFFIX):
        return None
    target = state_dir()
    if target.exists():
        return None
    for legacy_name in LEGACY_STATE_DIR_NAMES:
        legacy_dir = target.parent / legacy_name
        if not legacy_dir.is_dir():
            continue
        try:
            shutil.copytree(legacy_dir, target, dirs_exist_ok=True)
        except OSError as exc:  # pragma: no cover - defensive; surfaced in logs
            logger.warning("Could not copy legacy state from %s to %s: %s", legacy_dir, target, exc)
            return None
        logger.info(
            "Copied %s state from %s to %s (the old directory was left untouched)",
            PRODUCT_NAME,
            legacy_dir,
            target,
        )
        return legacy_dir
    return None

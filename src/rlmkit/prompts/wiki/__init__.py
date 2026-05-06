# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki prompt loader — versioned YAML, never inline strings."""

from __future__ import annotations

import importlib.resources as pkg_resources
from functools import lru_cache

import yaml


@lru_cache(maxsize=8)
def get_wiki_prompt(name: str) -> str:
    """Return the ``system`` field of ``prompts/wiki/<name>.yaml``."""
    pkg = pkg_resources.files("rlmkit.prompts.wiki")
    raw = (pkg / f"{name}.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(raw) or {}
    if "system" not in doc:
        raise ValueError(f"Wiki prompt {name!r} missing `system` field.")
    return str(doc["system"]).rstrip()

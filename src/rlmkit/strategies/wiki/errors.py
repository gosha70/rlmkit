# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Wiki-specific exception taxonomy.

Mirrors the cct exit-code taxonomy so existing operator habits
(``2 = backend not found``, ``4 = contract violation`` etc.)
carry over cleanly.
"""

from __future__ import annotations


class WikiError(Exception):
    """Base for all wiki-layer errors. Exit code 1 by default."""

    exit_code: int = 1


class BackendInvocationError(WikiError):
    exit_code = 3


class ContractViolationError(WikiError):
    exit_code = 4


class SourceMissingError(WikiError):
    exit_code = 5


class OutputDirError(WikiError):
    exit_code = 6


class PromoteValidationError(WikiError):
    exit_code = 4


class PromoteApplyError(WikiError):
    exit_code = 6

# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Exception hierarchy for the wiki module.

Each exception class corresponds to a documented CLI exit code so
callers can translate raised errors into stable exit semantics.
"""

from __future__ import annotations


class WikiError(Exception):
    """Base for every wiki-pipeline error. Exit code 1."""

    exit_code: int = 1


class BackendNotFound(WikiError):
    """Requested backend was not registered. Exit code 2."""

    exit_code = 2


class BackendFailure(WikiError):
    """Backend raised or exited non-zero. Exit code 3."""

    exit_code = 3


class ContractViolation(WikiError):
    """Backend output failed shape or semantic validation. Exit code 4."""

    exit_code = 4


class SourceMissing(WikiError):
    """The ingest source path does not exist or is unreadable. Exit code 5."""

    exit_code = 5


class OutputWriteFailure(WikiError):
    """The proposal output directory could not be written. Exit code 6."""

    exit_code = 6

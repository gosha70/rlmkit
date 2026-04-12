"""Named constants for sandbox variable names and config-extra keys.

These identifiers cross module boundaries (use cases set them,
infrastructure skips them on return), so they live here as the single
source of truth rather than as magic strings scattered across files.
"""

# ---------------------------------------------------------------------------
# Sandbox variable names — bound via sandbox.set_variable(), read by model
# code inside the subprocess, and excluded from the child → parent return
# to avoid O(payload-size) round-trips.
# ---------------------------------------------------------------------------

SANDBOX_VAR_DOCUMENT = "P"
"""The full document content variable.  Set by RunRLMUseCase."""

SANDBOX_VAR_FILE_INDEX = "_FILE_INDEX"
"""Multi-file index (forward reservation).  Excluded from return."""

SANDBOX_VAR_HISTORY = "history"
"""Conversation history list.  Set by RunRLMUseCase when conversation
memory is enabled; the model reads it via ``print(history[-1])``."""

# The set of variable names that should NOT round-trip child → parent
# after each sandbox execution.  Used by both the child-side
# ``_child_worker`` (skip_return) and the parent-side ``execute``
# (skip_keys) in ``subprocess_sandbox.py``.
SANDBOX_SKIP_RETURN_VARS = frozenset(
    {SANDBOX_VAR_DOCUMENT, SANDBOX_VAR_FILE_INDEX, SANDBOX_VAR_HISTORY}
)

# ---------------------------------------------------------------------------
# RunConfigDTO.extra keys — carry data from the routing layer to use cases
# ---------------------------------------------------------------------------

EXTRA_KEY_HISTORY_VARIABLE = "history_variable"
"""Key in ``RunConfigDTO.extra`` that carries the ``history`` list from
``_prepare_history_context`` in ``chat.py`` to ``RunRLMUseCase``."""

EXTRA_KEY_HISTORY_MESSAGES = "history_messages"
"""Key in ``RunConfigDTO.extra`` that carries prior-turn chat messages
(native user/assistant pairs) for Direct, Compare, and RAG modes."""

# ---------------------------------------------------------------------------
# History context path identifiers — returned in history_info["path"]
# ---------------------------------------------------------------------------

HISTORY_PATH_DISABLED = "disabled"
HISTORY_PATH_EMPTY = "empty"
HISTORY_PATH_INPROMPT = "inprompt"
HISTORY_PATH_REPL_VARIABLE = "repl_variable"

# ---------------------------------------------------------------------------
# Execution mode identifiers
# ---------------------------------------------------------------------------

MODE_DIRECT = "direct"
MODE_RLM = "rlm"
MODE_RAG = "rag"
MODE_COMPARE = "compare"
MODE_AUTO = "auto"

# Modes that use RLM internally (RLM run config applies)
MODES_RLM_INTERNAL = frozenset({MODE_RLM, MODE_AUTO, MODE_COMPARE})

# ---------------------------------------------------------------------------
# Mode groupings — which execution modes use which history path
# ---------------------------------------------------------------------------

MODES_INPROMPT = frozenset({MODE_DIRECT, MODE_COMPARE, MODE_RAG})
"""Modes that carry history as native user/assistant chat messages."""

MODES_REPL_VARIABLE = frozenset({MODE_RLM, MODE_AUTO})
"""Modes that carry history as a sandbox ``history`` Python variable."""

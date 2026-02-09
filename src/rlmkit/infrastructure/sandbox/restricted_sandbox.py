"""Restricted sandbox adapter using RestrictedPython.

Compiles LLM-generated code through RestrictedPython's AST transformer which
blocks dangerous patterns at compile time:
- Attribute access to dunder methods (``__class__``, ``__bases__``, etc.)
- Direct ``import`` of modules not in the allow-list
- ``exec`` / ``eval`` / ``compile`` (not present in ``safe_builtins``)
- File I/O (``open`` not present in ``safe_builtins``)
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Optional

from RestrictedPython import compile_restricted, safe_builtins, safe_globals
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector

from rlmkit.application.dto import ExecutionResultDTO
from rlmkit.domain.exceptions import SecurityViolationError


# Modules considered safe for LLM-generated code (subset of rlmkit defaults)
_DEFAULT_ALLOWED_MODULES = frozenset({
    "json", "re", "math", "datetime", "time", "calendar",
    "decimal", "fractions", "statistics", "random",
    "string", "textwrap", "unicodedata",
    "itertools", "functools", "operator", "collections",
    "heapq", "bisect", "array", "copy", "pprint",
    "enum", "dataclasses", "typing",
})


def _make_safe_import(allowed_modules: frozenset[str]):
    """Return a restricted ``__import__`` that only allows *allowed_modules*."""
    import builtins as _builtins

    def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        base = name.split(".")[0]
        if base not in allowed_modules:
            raise SecurityViolationError(
                f"Import of module '{name}' is not allowed. "
                f"Allowed modules: {', '.join(sorted(allowed_modules)[:10])}..."
            )
        return _builtins.__import__(name, *args, **kwargs)

    return _safe_import


class RestrictedSandboxAdapter:
    """Sandbox that uses RestrictedPython for compile-time and runtime safety.

    This adapter provides stronger isolation than the basic
    :class:`LocalSandboxAdapter` by using RestrictedPython's AST transformer
    to block dangerous patterns *before* the code runs.

    Args:
        allowed_modules: Set of module names that ``import`` may load.
        max_stdout_chars: Maximum characters captured from stdout/stderr.
    """

    def __init__(
        self,
        allowed_modules: Optional[frozenset[str]] = None,
        max_stdout_chars: int = 10_000,
    ) -> None:
        self._allowed_modules = allowed_modules or _DEFAULT_ALLOWED_MODULES
        self._max_stdout_chars = max_stdout_chars
        self._namespace: dict[str, Any] = {}
        self._build_globals()

    # ------------------------------------------------------------------
    # SandboxPort interface
    # ------------------------------------------------------------------

    def execute(self, code: str) -> ExecutionResultDTO:
        """Compile and execute *code* under RestrictedPython.

        Args:
            code: Python source code to execute.

        Returns:
            :class:`ExecutionResultDTO` with captured output.
        """
        # 1. Compile through RestrictedPython
        try:
            byte_code = compile_restricted(code, "<sandbox>", "exec")
        except SyntaxError as exc:
            return ExecutionResultDTO(exception=f"SyntaxError: {exc}")

        if byte_code is None:
            return ExecutionResultDTO(exception="RestrictedPython compilation failed")

        # 2. Execute in the restricted namespace
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exception_msg: Optional[str] = None
        truncated = False

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(byte_code, self._globals, self._namespace)  # noqa: S102
        except SecurityViolationError as exc:
            exception_msg = f"SecurityViolationError: {exc}"
        except Exception as exc:
            exception_msg = f"{type(exc).__name__}: {exc}"

        # Collect printed output from PrintCollector
        printer = self._namespace.pop("_print", None)
        printed_text = ""
        if printer is not None:
            printed_text = printer()

        stdout = printed_text + stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        if len(stdout) > self._max_stdout_chars:
            stdout = stdout[: self._max_stdout_chars] + "\n... (truncated)"
            truncated = True

        return ExecutionResultDTO(
            stdout=stdout,
            stderr=stderr,
            exception=exception_msg,
            truncated=truncated,
        )

    def reset(self) -> None:
        """Reset the execution namespace to a clean state."""
        self._namespace.clear()
        self._build_globals()

    def is_healthy(self) -> bool:
        """The restricted sandbox is always healthy if RestrictedPython is importable."""
        return True

    def set_variable(self, name: str, value: Any) -> None:
        """Inject a variable into the sandbox namespace.

        Args:
            name: Variable name.
            value: Variable value.
        """
        self._namespace[name] = value

    def get_variable(self, name: str) -> Optional[Any]:
        """Retrieve a variable from the sandbox namespace.

        Args:
            name: Variable name.

        Returns:
            The variable's value, or ``None`` if not found.
        """
        return self._namespace.get(name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_globals(self) -> None:
        """(Re)build the restricted globals dict."""
        self._globals = safe_globals.copy()

        # RestrictedPython guard hooks
        self._globals["_print_"] = PrintCollector
        self._globals["_getattr_"] = safer_getattr
        self._globals["_getiter_"] = default_guarded_getiter
        self._globals["_getitem_"] = default_guarded_getitem
        self._globals["_iter_unpack_sequence_"] = guarded_unpack_sequence

        # Provide a restricted __import__
        builtins_dict = dict(safe_builtins)
        builtins_dict["__import__"] = _make_safe_import(self._allowed_modules)
        self._globals["__builtins__"] = builtins_dict

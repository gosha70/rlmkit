"""Subprocess sandbox adapter: out-of-process execution with hard timeouts.

Runs each code cell in a short-lived child process communicating via pipes.
The child process inherits persistent environment state serialized as JSON,
executes with a hard ``signal.alarm`` + ``SIGKILL`` timeout chain, and returns
updated globals + stdout/stderr.

This adapter is the safe default for non-main-thread contexts (FastAPI, etc.)
where signal-based timeouts on the local sandbox don't work.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import signal
from functools import partial
from typing import Any

from rlmkit.application.dto import ExecutionResultDTO

logger = logging.getLogger(__name__)

# Types that survive JSON round-tripping without loss.
_SERIALIZABLE_TYPES = (str, int, float, bool, type(None), list, dict)


def _is_json_serializable(value: Any) -> bool:
    """Check whether *value* can survive a JSON round-trip."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _serialize_namespace(ns: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of *ns*, skipping non-serializable entries."""
    safe: dict[str, Any] = {}
    for key, value in ns.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, _SERIALIZABLE_TYPES):
            logger.debug("Skipping non-serializable variable %r (%s)", key, type(value).__name__)
            continue
        if _is_json_serializable(value):
            safe[key] = value
        else:
            logger.debug("Skipping variable %r: JSON serialization failed", key)
    return safe


def _child_worker(
    conn: mp.connection.Connection,
    code: str,
    namespace_json: str,
    timeout_s: float,
    max_stdout_chars: int,
    safe_mode: bool = False,
    allowed_imports: list[str] | None = None,
) -> None:
    """Execute *code* in the child process and send results back via *conn*.

    This function runs as the ``target`` of a :class:`multiprocessing.Process`.
    It is the main thread of its own process, so ``signal.alarm`` works.
    """
    import io
    from contextlib import redirect_stderr, redirect_stdout

    # -- Deserialize namespace --
    try:
        namespace: dict[str, Any] = json.loads(namespace_json)
    except json.JSONDecodeError:
        namespace = {}

    # -- Build execution globals (restricted or unrestricted) --
    if safe_mode:
        from rlmkit.envs.sandbox import create_safe_globals

        exec_globals = create_safe_globals(allowed_imports=allowed_imports)
    else:
        exec_globals = {"__builtins__": __builtins__}

    # -- Re-inject content-navigation tools --
    try:
        from rlmkit.tools.content import (
            chunk,
            grep,
            grep_file,
            outline_file,
            peek,
            peek_file,
            select,
        )

        tool_globals: dict[str, Any] = {
            "peek": peek,
            "grep": grep,
            "grep_file": grep_file,
            "outline_file": outline_file,
            "peek_file": peek_file,
            "chunk": chunk,
            "select": select,
        }

        # Bind tools to content variable P if present.
        content = namespace.get("P")
        if isinstance(content, str):
            tool_globals["peek"] = partial(peek, content)
            tool_globals["peek_file"] = partial(peek_file, content)
            tool_globals["grep"] = partial(grep, content)
            tool_globals["grep_file"] = partial(grep_file, content)
            tool_globals["outline_file"] = partial(outline_file, content)
            tool_globals["chunk"] = partial(chunk, content)
            tool_globals["select"] = partial(select, content)
    except ImportError:
        tool_globals = {}

    exec_globals.update(tool_globals)
    exec_globals.update(namespace)

    # -- Set up signal-based timeout (we ARE the main thread) --
    timed_out = False

    if hasattr(signal, "SIGALRM"):

        def _alarm_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Code execution timed out after {timeout_s}s")

        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(max(1, timeout_s)))

    # -- Execute --
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exception_msg: str | None = None

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, exec_globals)  # noqa: S102
    except TimeoutError:
        timed_out = True
        exception_msg = f"Code execution timed out after {timeout_s}s"
    except Exception as exc:
        exception_msg = f"{type(exc).__name__}: {exc}"
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    # -- Collect updated namespace (primitives only) --
    # Exclude parent-injected content keys (P, _FILE_INDEX) to avoid
    # round-tripping potentially large document payloads back.
    skip_return = {"P", "_FILE_INDEX"}
    updated_ns: dict[str, Any] = {}
    for key, value in exec_globals.items():
        if key.startswith("_") or key == "__builtins__":
            continue
        if key in skip_return:
            continue
        if callable(value):
            continue
        if _is_json_serializable(value):
            updated_ns[key] = value

    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()
    truncated = False
    if len(stdout) > max_stdout_chars:
        stdout = stdout[:max_stdout_chars] + "\n... (truncated)"
        truncated = True

    # -- Send result back --
    result = json.dumps(
        {
            "stdout": stdout,
            "stderr": stderr,
            "exception": exception_msg,
            "timeout": timed_out,
            "truncated": truncated,
            "namespace": updated_ns,
        }
    )
    try:
        conn.send(result)
    except Exception:  # nosec B110 — pipe may be broken if parent timed out; nothing to do
        pass  # noqa: S110
    finally:
        conn.close()


class SubprocessSandboxAdapter:
    """Sandbox that executes code in a child process with hard timeout enforcement.

    Each :meth:`execute` call spawns a short-lived child process. The persistent
    namespace is serialized as JSON (primitives only), sent to the child, and
    the updated namespace is merged back after execution.

    Args:
        safe_mode: When True, the child process uses restricted builtins and
            import controls (same restrictions as ``PyReplEnv(safe_mode=True)``).
        allowed_imports: Modules allowed in safe mode.
        max_exec_time_s: Per-execution timeout in seconds.
        max_stdout_chars: Maximum characters captured from stdout.
    """

    def __init__(
        self,
        safe_mode: bool = False,
        allowed_imports: list[str] | None = None,
        max_exec_time_s: float = 5.0,
        max_stdout_chars: int = 10_000,
    ) -> None:
        self._safe_mode = safe_mode
        self._allowed_imports = allowed_imports
        self._max_exec_time_s = max_exec_time_s
        self._max_stdout_chars = max_stdout_chars
        self._namespace: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # SandboxPort interface
    # ------------------------------------------------------------------

    def execute(self, code: str) -> ExecutionResultDTO:
        """Execute *code* in a child process with hard timeout.

        Args:
            code: Python source code to execute.

        Returns:
            :class:`ExecutionResultDTO` with captured output.
        """
        namespace_json = json.dumps(_serialize_namespace(self._namespace))

        parent_conn, child_conn = mp.Pipe(duplex=False)

        process = mp.Process(
            target=_child_worker,
            args=(
                child_conn,
                code,
                namespace_json,
                self._max_exec_time_s,
                self._max_stdout_chars,
                self._safe_mode,
                self._allowed_imports,
            ),
            daemon=True,
        )
        process.start()
        child_conn.close()  # Parent doesn't write to child's end.

        # Wait for the child to finish or timeout.
        # Give a small grace period beyond the signal timeout for cleanup.
        join_timeout = self._max_exec_time_s + 2.0
        has_result = parent_conn.poll(join_timeout)

        if has_result:
            try:
                raw = parent_conn.recv()
                result = json.loads(raw)
            except Exception as exc:
                result = {
                    "stdout": "",
                    "stderr": "",
                    "exception": f"Failed to read child result: {exc}",
                    "timeout": False,
                    "truncated": False,
                    "namespace": {},
                }
        else:
            # Timed out — kill the child.
            result = {
                "stdout": "",
                "stderr": "",
                "exception": f"Code execution timed out after {self._max_exec_time_s}s",
                "timeout": True,
                "truncated": False,
                "namespace": {},
            }

        parent_conn.close()

        # Ensure the child process is cleaned up.
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        else:
            process.join(timeout=1.0)

        # Replace serializable portion of namespace with child's result.
        # This correctly reflects deletions (`del x`) and variables that
        # became non-serializable (e.g. reassigned to a lambda).  Keys that
        # were never sent to the child (non-serializable from the start,
        # e.g. lambdas) are preserved as-is.  Content keys (P, _FILE_INDEX)
        # are intentionally excluded from the child return to avoid O(doc-size)
        # round-trips, so we also exclude them from deletion detection.
        updated_ns: dict[str, Any] = result.get("namespace", {})
        skip_keys = {"P", "_FILE_INDEX"}
        sent_keys = set(_serialize_namespace(self._namespace).keys()) - skip_keys
        for key in sent_keys - set(updated_ns.keys()):
            self._namespace.pop(key, None)
        self._namespace.update(updated_ns)

        return ExecutionResultDTO(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            exception=result.get("exception"),
            timeout=result.get("timeout", False),
            truncated=result.get("truncated", False),
        )

    def reset(self) -> None:
        """Clear the persistent namespace."""
        self._namespace.clear()

    def is_healthy(self) -> bool:
        """The subprocess sandbox has no external dependencies."""
        return True

    def set_variable(self, name: str, value: Any) -> None:
        """Inject a variable into the persistent namespace.

        Args:
            name: Variable name.
            value: Variable value.
        """
        self._namespace[name] = value

    def get_variable(self, name: str) -> Any | None:
        """Retrieve a variable from the persistent namespace.

        Args:
            name: Variable name.

        Returns:
            The variable's value, or ``None`` if not found.
        """
        return self._namespace.get(name)

    async def execute_async(self, code: str) -> ExecutionResultDTO:
        """Execute code asynchronously by delegating to the sync method."""
        import asyncio

        return await asyncio.to_thread(self.execute, code)

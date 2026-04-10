"""Tests for SubprocessSandboxAdapter."""

from __future__ import annotations

import time

import pytest

from rlmkit.infrastructure.sandbox.subprocess_sandbox import SubprocessSandboxAdapter


@pytest.fixture
def sandbox() -> SubprocessSandboxAdapter:
    """Create a fresh subprocess sandbox with a short timeout."""
    return SubprocessSandboxAdapter(max_exec_time_s=3.0, max_stdout_chars=5000)


class TestBasicExecution:
    """Basic code execution tests."""

    def test_print_hello(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute('print("hello")')
        assert result.success
        assert result.stdout.strip() == "hello"
        assert result.exception is None
        assert not result.timeout

    def test_arithmetic(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("x = 2 + 3")
        assert result.success
        assert sandbox.get_variable("x") == 5

    def test_multiline_code(self, sandbox: SubprocessSandboxAdapter) -> None:
        code = "a = 1\nb = 2\nc = a + b\nprint(c)"
        result = sandbox.execute(code)
        assert result.success
        assert result.stdout.strip() == "3"
        assert sandbox.get_variable("c") == 3

    def test_syntax_error(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("def f(:\n  pass")
        assert not result.success
        assert result.exception is not None
        assert "SyntaxError" in result.exception

    def test_runtime_error(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("1 / 0")
        assert not result.success
        assert result.exception is not None
        assert "ZeroDivisionError" in result.exception

    def test_name_error(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("print(undefined_var)")
        assert not result.success
        assert "NameError" in (result.exception or "")


class TestVariablePersistence:
    """Variable persistence across executions."""

    def test_set_then_get(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("x", 1)
        result = sandbox.execute("x += 1")
        assert result.success
        assert sandbox.get_variable("x") == 2

    def test_variable_persists_across_calls(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.execute("counter = 10")
        sandbox.execute("counter += 5")
        assert sandbox.get_variable("counter") == 15

    def test_get_variable_not_found(self, sandbox: SubprocessSandboxAdapter) -> None:
        assert sandbox.get_variable("nonexistent") is None

    def test_set_variable_dict(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("data", {"key": "value", "count": 42})
        result = sandbox.execute('print(data["key"])')
        assert result.success
        assert result.stdout.strip() == "value"

    def test_set_variable_list(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("items", [1, 2, 3])
        result = sandbox.execute("items.append(4)\nprint(len(items))")
        assert result.success
        assert result.stdout.strip() == "4"
        assert sandbox.get_variable("items") == [1, 2, 3, 4]


class TestTimeoutEnforcement:
    """Timeout enforcement with hard kill."""

    def test_infinite_loop_times_out(self) -> None:
        sandbox = SubprocessSandboxAdapter(max_exec_time_s=1.0)
        start = time.monotonic()
        result = sandbox.execute("while True: pass")
        elapsed = time.monotonic() - start

        assert result.timeout
        assert not result.success
        assert "timed out" in (result.exception or "").lower()
        # Should complete within timeout + 2s grace period
        assert elapsed < 1.0 + 3.0, f"Took too long: {elapsed:.1f}s"

    def test_sleep_within_timeout(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("import time; time.sleep(0.1); print('done')")
        assert result.success
        assert result.stdout.strip() == "done"


class TestToolFunctions:
    """Content-navigation tool availability in the child process."""

    def test_peek_with_content(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("P", "hello world, this is a test")
        result = sandbox.execute("print(peek(start=0, end=5))")
        assert result.success
        assert "hello" in result.stdout

    def test_grep_with_content(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("P", "line one\nline two\nline three\n")
        result = sandbox.execute('print(grep("two"))')
        assert result.success
        assert "two" in result.stdout

    def test_chunk_with_content(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("P", "abcdefghij" * 10)
        result = sandbox.execute("chunks = chunk(size=20)\nprint(len(chunks))")
        assert result.success
        assert int(result.stdout.strip()) == 5

    def test_tools_without_content(self, sandbox: SubprocessSandboxAdapter) -> None:
        # Tools should still be importable; they just need content arg explicitly
        result = sandbox.execute("from rlmkit.tools.content import peek\nprint(peek('abc', 0, 2))")
        assert result.success
        assert "ab" in result.stdout


class TestNonSerializableVariables:
    """Non-serializable variables are skipped gracefully."""

    def test_lambda_skipped(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("fn", lambda x: x + 1)
        sandbox.set_variable("x", 10)
        result = sandbox.execute("y = x + 1\nprint(y)")
        assert result.success
        assert result.stdout.strip() == "11"
        # fn is not available in child (non-serializable), but no crash
        assert sandbox.get_variable("y") == 11

    def test_class_instance_skipped(self, sandbox: SubprocessSandboxAdapter) -> None:
        class Custom:
            pass

        sandbox.set_variable("obj", Custom())
        sandbox.set_variable("num", 5)
        result = sandbox.execute("print(num)")
        assert result.success
        assert result.stdout.strip() == "5"


class TestReset:
    """Reset clears the namespace."""

    def test_reset_clears_variables(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("x", 42)
        assert sandbox.get_variable("x") == 42
        sandbox.reset()
        assert sandbox.get_variable("x") is None

    def test_reset_then_execute(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.execute("x = 100")
        sandbox.reset()
        result = sandbox.execute("print(x)")
        assert not result.success  # x is undefined after reset
        assert "NameError" in (result.exception or "")


class TestHealthCheck:
    """is_healthy always returns True."""

    def test_is_healthy(self, sandbox: SubprocessSandboxAdapter) -> None:
        assert sandbox.is_healthy() is True


class TestStdoutTruncation:
    """Stdout truncation when output exceeds max_stdout_chars."""

    def test_large_output_truncated(self) -> None:
        sandbox = SubprocessSandboxAdapter(max_exec_time_s=3.0, max_stdout_chars=100)
        result = sandbox.execute("print('x' * 500)")
        assert result.truncated
        assert len(result.stdout) < 200  # truncated + marker


class TestStderr:
    """Stderr capture."""

    def test_stderr_captured(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = sandbox.execute("import sys; sys.stderr.write('warning\\n')")
        assert "warning" in result.stderr


class TestNamespaceDeletionSemantics:
    """P1 fix: deletions and non-serializable reassignments are reflected."""

    def test_del_removes_variable(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("x", 42)
        result = sandbox.execute("del x")
        assert result.success
        assert sandbox.get_variable("x") is None

    def test_reassign_to_non_serializable_clears(self, sandbox: SubprocessSandboxAdapter) -> None:
        sandbox.set_variable("x", 10)
        # Child reassigns x to a lambda — not serializable, so parent
        # should see x removed (not stale value 10).
        result = sandbox.execute("x = lambda: 1")
        assert result.success
        assert sandbox.get_variable("x") is None

    def test_non_serializable_parent_var_preserved(self, sandbox: SubprocessSandboxAdapter) -> None:
        # Non-serializable vars that were never sent to child stay intact.
        sandbox.set_variable("fn", lambda: 99)
        sandbox.set_variable("y", 5)
        result = sandbox.execute("y += 1")
        assert result.success
        assert sandbox.get_variable("y") == 6
        # fn was never serialized/sent, so it should survive.
        assert callable(sandbox.get_variable("fn"))


class TestContentNotRoundTripped:
    """P2 fix: P is not returned from child, avoiding O(doc-size) overhead."""

    def test_large_p_not_in_child_return(self, sandbox: SubprocessSandboxAdapter) -> None:
        big_content = "x" * 100_000
        sandbox.set_variable("P", big_content)
        sandbox.set_variable("counter", 0)
        result = sandbox.execute("counter = 1")
        assert result.success
        assert sandbox.get_variable("counter") == 1
        # P should still be present in the parent namespace (never removed).
        assert sandbox.get_variable("P") == big_content


class TestServerSandboxSelection:
    """P0 fix: server does NOT override 'restricted' to 'subprocess'."""

    def test_restricted_type_passes_through(self) -> None:
        from unittest.mock import patch

        from rlmkit.server.dependencies import AppState

        state = AppState(load_from_disk=False)
        state.config.sandbox.type = "restricted"
        with patch("rlmkit.server.dependencies.create_sandbox") as mock_create:
            state.create_sandbox()
            mock_create.assert_called_once_with(sandbox_type="restricted")

    def test_local_type_becomes_subprocess(self) -> None:
        from unittest.mock import patch

        from rlmkit.server.dependencies import AppState

        state = AppState(load_from_disk=False)
        state.config.sandbox.type = "local"
        with patch("rlmkit.server.dependencies.create_sandbox") as mock_create:
            state.create_sandbox()
            mock_create.assert_called_once_with(sandbox_type="subprocess")


class TestSafeMode:
    """safe_mode restricts builtins and imports in the child process."""

    @pytest.fixture
    def safe_sandbox(self) -> SubprocessSandboxAdapter:
        return SubprocessSandboxAdapter(safe_mode=True, max_exec_time_s=3.0)

    def test_safe_mode_blocks_os_import(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("import os")
        assert not result.success
        assert result.exception is not None
        assert "not allowed" in result.exception.lower() or "Security" in result.exception

    def test_safe_mode_blocks_subprocess(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("import subprocess")
        assert not result.success
        assert result.exception is not None

    def test_safe_mode_blocks_open(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("f = open('/etc/passwd')")
        assert not result.success
        assert result.exception is not None

    def test_safe_mode_allows_safe_imports(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("import json\nprint(json.dumps([1,2]))")
        assert result.success
        assert "[1, 2]" in result.stdout

    def test_safe_mode_allows_math(self, safe_sandbox: SubprocessSandboxAdapter) -> None:
        result = safe_sandbox.execute("import math\nprint(math.pi)")
        assert result.success
        assert "3.14" in result.stdout

    def test_unsafe_mode_allows_os(self, sandbox: SubprocessSandboxAdapter) -> None:
        # Default fixture has safe_mode=False
        result = sandbox.execute("import os; print(os.getpid())")
        assert result.success
        assert result.stdout.strip().isdigit()

    def test_factory_forwards_safe_mode(self) -> None:
        """Factory passes safe_mode through to SubprocessSandboxAdapter."""
        from rlmkit.infrastructure.sandbox.sandbox_factory import create_sandbox

        sb = create_sandbox(sandbox_type="subprocess", safe_mode=True)
        result = sb.execute("import os")
        assert not result.success
        assert result.exception is not None
        assert "not allowed" in result.exception.lower() or "Security" in result.exception


class TestAsyncExecution:
    """Async execution wrapper."""

    @pytest.mark.asyncio
    async def test_execute_async(self, sandbox: SubprocessSandboxAdapter) -> None:
        result = await sandbox.execute_async('print("async hello")')
        assert result.success
        assert "async hello" in result.stdout
